"""
Module Name: loggers

Description: This module implements all logging functionality

Classes & Methods:
- RunningAvg: keeps a running average score
    - update: updates the current total and increases the number of elements to reflect the change
- FloatingMetricSummary: keeps a list of results for a metric (glorified list)
    - add_result: adds a result to the list
    - get_summary: calculates the average and STD of the results logged for the metric
- RunLogger: keeps track of all metrics and all predictions made during the course of a single run
    - add_fold_results: appends the groundtruths, predictions and prediction scores of a single fold
    - add_fold_loss: appends 
    - dump_to_file: saves the already logged groundtruths, predictions and prediction scores to a pickle file in the specified directory
    - get_results_per_fold: calculates and returns the segment-level and speaker-level results for each fold
    - get_results_run: calculates and returns the segment-level and speaker-level results "averaged" over all folds
- BatchLogger: implements a logger that holds the logger objects of all runs in a particular batch
    - add_run: adds a RunLogger instance to the batch of runs
    - dump_to_file: saves the dictionary containing RunLogger objects to disk
    - get_results_per_run: retrieves the segment-level and speaker-level results for each run in the batch
    - get_results_batch: retrieves the segment-level and speaker-level results for each run in the batch
"""

import os
from typing import Tuple, List

import wandb
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from evaluation_utils import calculate_metrics_with_cis, calculate_metrics
from preprocessing_utils import label_encoder
from utils import flatten_dict, flat_nested_dict


class RunningAvg:
    """Implements a running average object."""

    def __init__(self):
        self.num_elements = 0
        self.current_total = 0

    def update(self, val: float):
        """Updates the current total and increases the number of elements to reflect the change.

        Args:
            val (float): the value to add to the current total
        """
        self.current_total += val
        self.num_elements += 1

    def __call__(self) -> float:
        """Returns the average at that specific moment

        Returns:
            float: the current average
        """
        return self.current_total / self.num_elements


class FloatingMetricSummary:
    def __init__(self):
        self.results = []

    def add_result(self, res: float):
        self.results.append(res)

    def get_summary(self):
        return (np.average(self.results), np.std(self.results))


class RunLogger:
    """Implements a logger that holds the groundtruths, predictions and prediction scores for each fold in a run and per client. On request
    this logger calculates the metrics per client across folds and across clients across folds.
    """

    def __init__(self, run_id: str, client_ids: List[str]):
        """Initializes a RunHistoryLogger object with empty dictionaries of per fold groundtruths, predictions, prediction scores, and optionally, speaker IDs

        Args:
            run_id (str): an identifier for the run
            client_ids (List[str]): the IDs of the clients on whose data we evaluate in this run
        """
        self.run_id = run_id

        # a dictionary containing per client results
        self.client_results = {}
        # a dictionary containing per client test results
        self.client_test_results = {}

        for client_id in client_ids:
            self.client_results[client_id] = {}
            self.client_results[client_id]["groundtruths_per_fold"] = {}
            self.client_results[client_id]["predictions_per_fold"] = {}
            self.client_results[client_id]["prediction_scores_per_fold"] = {}

        self.label2id, self.id2label = label_encoder()

    def add_client_fold_results(
        self,
        client_id: int,
        fold_id: int,
        groundtruths: np.ndarray,
        predictions: np.ndarray,
        prediction_scores: np.ndarray,
    ):
        """Appends the groundtruths, predictions and prediction scores of a single fold to the client's results.

        Args:
            client_id (int): the id of the client for whome we are adding fold results
            fold_id (int): the id of the fold for which we are adding results
            groundtruths (np.ndarray): a list of groundtruth labels for the fold
            predictions (np.ndarray): a list of predicted labels for the fold
            prediction_scores (np.ndarray): a 2d array of prediction scores for the fold
        """

        assert len(groundtruths) == len(predictions) == prediction_scores.shape[0]

        self.client_results[client_id]["groundtruths_per_fold"][
            f"fold_{fold_id}"
        ] = groundtruths.copy()
        self.client_results[client_id]["predictions_per_fold"][
            f"fold_{fold_id}"
        ] = predictions.copy()
        self.client_results[client_id]["prediction_scores_per_fold"][
            f"fold_{fold_id}"
        ] = prediction_scores.copy()

    def add_client_test_results(
        self,
        client_id: int,
        groundtruths: np.ndarray,
        predictions: np.ndarray,
        prediction_scores: np.ndarray,
    ):

        assert len(groundtruths) == len(predictions) == prediction_scores.shape[0]

        self.client_test_results[client_id] = {}
        self.client_test_results[client_id]["groundtruths"] = groundtruths.copy()
        self.client_test_results[client_id]["predictions"] = predictions.copy()
        self.client_test_results[client_id][
            "prediction_scores"
        ] = prediction_scores.copy()

    def dump_to_file(self, output_dir: str):
        """Saves the already logged client groundtruths, predictions and prediction scores to a pickle file in the specified directory

        Args:
            output_dir (str): a path to the directory where the results should be saved
        """

        with open(f"{output_dir}/{self.run_id}_client_result_logs.pkl", "wb") as f:
            pickle.dump(self.client_results, f)

        if self.client_test_results:
            with open(
                f"{output_dir}/{self.run_id}_client_test_result_logs.pkl", "wb"
            ) as f:
                pickle.dump(self.client_test_results, f)

    def _concatenate_client_cv_results(self):
        concatenated_folds_per_client = {}

        for client_id, client_results in self.client_results.items():

            concatenated_folds_per_client[client_id] = {
                "groundtruths": None,
                "predictions": None,
                "prediction_scores": None,
            }

            for fold_id in client_results["groundtruths_per_fold"].keys():
                fold_gts = client_results["groundtruths_per_fold"][fold_id]
                fold_scores = client_results["prediction_scores_per_fold"][fold_id]

                if concatenated_folds_per_client[client_id]["groundtruths"] is None:
                    concatenated_folds_per_client[client_id][
                        "groundtruths"
                    ] = fold_gts.copy()
                    concatenated_folds_per_client[client_id][
                        "prediction_scores"
                    ] = fold_scores.copy()
                else:
                    concatenated_folds_per_client[client_id]["groundtruths"] = (
                        np.concatenate(
                            [
                                concatenated_folds_per_client[client_id][
                                    "groundtruths"
                                ],
                                fold_gts,
                            ]
                        )
                    )
                    concatenated_folds_per_client[client_id]["prediction_scores"] = (
                        np.concatenate(
                            [
                                concatenated_folds_per_client[client_id][
                                    "prediction_scores"
                                ],
                                fold_scores,
                            ],
                            axis=0,
                        )
                    )

        return concatenated_folds_per_client

    def _concatenate_test_results(self):
        concatenated_test_results = {
            "groundtruths": None,
            "predictions": None,
            "prediction_scores": None,
            "client_ids": None,
        }

        for client_id, client_results in self.client_test_results.items():
            if concatenated_test_results["groundtruths"] is not None:
                concatenated_test_results["groundtruths"] = np.concatenate(
                    [
                        concatenated_test_results["groundtruths"],
                        client_results["groundtruths"],
                    ],
                    axis=0,
                )
                concatenated_test_results["predictions"] = np.concatenate(
                    [
                        concatenated_test_results["predictions"],
                        client_results["predictions"],
                    ],
                    axis=0,
                )
                concatenated_test_results["prediction_scores"] = np.concatenate(
                    [
                        concatenated_test_results["prediction_scores"],
                        client_results["prediction_scores"],
                    ],
                    axis=0,
                )
                concatenated_test_results["client_ids"] = np.concatenate(
                    [
                        concatenated_test_results["client_ids"],
                        np.array([client_id] * client_results["groundtruths"].shape[0]),
                    ],
                    axis=0,
                )
            else:
                concatenated_test_results["groundtruths"] = client_results[
                    "groundtruths"
                ].copy()
                concatenated_test_results["predictions"] = client_results[
                    "predictions"
                ].copy()
                concatenated_test_results["prediction_scores"] = client_results[
                    "prediction_scores"
                ].copy()
                concatenated_test_results["client_ids"] = np.array(
                    [client_id] * client_results["groundtruths"].shape[0]
                )

        return concatenated_test_results

    def get_results_run(self, output_dir: str = None) -> Tuple[dict, dict]:
        """Calculates and returns client-level results averaged over all folds as well as results averaged over clients and folds.
           The client-level results aren't really averaged since groundtruths and predictions across folds are first concatenated together.

        Args:
            output_dir (str, optional): the directory where output files will be saved. If None, no files are saved.

        Returns:
            Tuple[dict, dict]: a tuple containing three dictionaries. The first dictionary contains the client-level results across folds. The second dictionary contains the speaker-level results across folds.
        """

        concatenated_client_cv_results = self._concatenate_client_cv_results()

        # calculate CV and test results per client
        ########################################################################
        metrics_per_client = {}

        for client_id, concat_client_results in concatenated_client_cv_results.items():
            metrics_per_client[client_id] = calculate_metrics(
                concat_client_results["groundtruths"],
                concat_client_results["prediction_scores"],
                self.id2label,
            )

        # do the same for the test metrics
        test_metrics_per_client = {}
        test_all_metrics_per_client = {}

        if self.client_test_results:

            for client_id, client_test_results in self.client_test_results.items():
                test_metrics_per_client[client_id] = calculate_metrics_with_cis(
                    client_test_results["groundtruths"],
                    client_test_results["predictions"],
                )
                
                test_all_metrics_per_client[client_id] = calculate_metrics(
                    client_test_results["groundtruths"],
                    client_test_results["prediction_scores"],
                    self.id2label,
                )

        ########################################################################
        ########################################################################

        ########################################################################
        # * calculate and summarize the CV results across clients
        ########################################################################
        # these are just the averages and standard deviations across clients of all per-client metrics
        ########################################################################
        metrics_across_clients = {}

        # iterate over all metrics saved for the first client (since every client should have the same metrics)
        first_clients_metrics = metrics_per_client[list(metrics_per_client.keys())[0]]
        for metric_name, metric_value in first_clients_metrics.items():
            # iterate over all clients
            for client_id in metrics_per_client.keys():
                # extract the value of the metric for the current client
                client_metric_value = metrics_per_client[client_id][metric_name]
                # check if it is a float
                if isinstance(metric_value, float):
                    # if it is not present in the metrics_across_clients dictionary, introduce it as a FloatingMetricSummary object
                    if metric_name not in metrics_across_clients:
                        metrics_across_clients[metric_name] = FloatingMetricSummary()
                    metrics_across_clients[metric_name].add_result(client_metric_value)
                # check if it is a numpy array (for CMs)
                elif isinstance(metric_value, np.ndarray):
                    # if it is not present in the metrics_across_clients dictionary, introduce it as a list object
                    if metric_name not in metrics_across_clients:
                        metrics_across_clients[metric_name] = []
                    metrics_across_clients[metric_name].append(client_metric_value)

        # * extract the average and standard deviation of all the metrics recorded in the `metrics_across_clients` dictionary
        ########################################################################
        across_clients_summary = {}

        for metric_name, metric_summary in metrics_across_clients.items():
            if isinstance(metric_summary, FloatingMetricSummary):
                metric_mean, metric_std = metric_summary.get_summary()
                across_clients_summary[f"{metric_name}_mean"] = metric_mean
                across_clients_summary[f"{metric_name}_std"] = metric_std
            else:
                # it can only be a list of ndarrays
                stacked_arrays = np.stack(metric_summary, axis=2)

                mean_cm = np.mean(stacked_arrays, axis=2)
                std_cm = np.std(stacked_arrays, axis=2)

                inter = mean_cm.sum(axis=1)[:, np.newaxis]
                inter[inter == 0] = np.inf
                norm_cm = mean_cm / inter

                across_clients_summary["mean_cm"] = mean_cm
                across_clients_summary["std_cm"] = std_cm
                across_clients_summary["norm_cm"] = norm_cm

        ########################################################################
        ########################################################################

        ########################################################################
        # * calculate and summarize the test results as well
        ########################################################################
        across_clients_test_summary = None
        test_across_clients_all_summary = {}
        #test_all_metrics_per_client = {}

        if self.client_test_results:
            concatenated_test_results = self._concatenate_test_results()

            across_clients_test_summary = calculate_metrics_with_cis(
                concatenated_test_results["groundtruths"],
                concatenated_test_results["predictions"],
                concatenated_test_results["client_ids"],
            )

            #for client_id in concatenated_test_results["client_ids"]:
#
            #    test_all_metrics_per_client[client_id] = calculate_metrics(
            #        concatenated_test_results["groundtruths"],
            #        concatenated_test_results["prediction_scores"],
            #        self.id2label,
            #    )
               
            ###################################################################
            # * calculate and summarize the test results across clients

            test_metrics_across_clients = {}

            # iterate over all metrics saved for the first client (since every client should have the same metrics)
            first_clients_metrics = test_all_metrics_per_client[
                list(test_all_metrics_per_client.keys())[0]
            ]
            for metric_name, metric_value in first_clients_metrics.items():
                # iterate over all clients
                for client_id in test_all_metrics_per_client.keys():
                    # extract the value of the metric for the current client
                    client_metric_value = test_all_metrics_per_client[client_id][
                        metric_name
                    ]

                    # check if it is a float
                    if isinstance(metric_value, float):
                        # if it is not present in the test_metrics_across_clients dictionary, introduce it as a FloatingMetricSummary object
                        if metric_name not in test_metrics_across_clients:
                            test_metrics_across_clients[metric_name] = (
                                FloatingMetricSummary()
                            )
                        test_metrics_across_clients[metric_name].add_result(
                            client_metric_value
                        )
                    # check if it is a numpy array (for CMs)
                    elif isinstance(metric_value, np.ndarray):
                        # if it is not present in the test_metrics_across_clients dictionary, introduce it as a list object
                        if metric_name not in test_metrics_across_clients:
                            test_metrics_across_clients[metric_name] = []
                        test_metrics_across_clients[metric_name].append(
                            client_metric_value
                        )

            # * extract the average and standard deviation of all the metrics recorded in the `test_metrics_across_clients` dictionary
            ########################################################################

            for metric_name, metric_summary in test_metrics_across_clients.items():
                if isinstance(metric_summary, FloatingMetricSummary):
                    metric_mean, metric_std = metric_summary.get_summary()
                    test_across_clients_all_summary[f"{metric_name}_mean"] = metric_mean
                    test_across_clients_all_summary[f"{metric_name}_std"] = metric_std
                else:
                    # it can only be a list of ndarrays
                    stacked_arrays = np.stack(metric_summary, axis=2)

                    mean_cm = np.mean(stacked_arrays, axis=2)
                    std_cm = np.std(stacked_arrays, axis=2)

                    inter = mean_cm.sum(axis=1)[:, np.newaxis]
                    inter[inter == 0] = np.inf
                    norm_cm = mean_cm / inter

                    test_across_clients_all_summary["mean_cm"] = mean_cm
                    test_across_clients_all_summary["std_cm"] = std_cm
                    test_across_clients_all_summary["norm_cm"] = norm_cm

        ########################################################################
        ########################################################################

        if output_dir is not None:
            with open(f"{output_dir}/metrics_per_client.pkl", "wb") as f:
                pickle.dump(metrics_per_client, f)

            if self.test_metrics_per_client:
                with open(f"{output_dir}/test_metrics_per_client.pkl", "wb") as f:
                    pickle.dump(test_metrics_per_client, f)

            if self.test_across_clients_all_summary:
                with open(
                    f"{output_dir}/test_all_metrics_across_clients_summary", "wb"
                ) as f:
                    pickle.dump(test_across_clients_all_summary, f)

            with open(f"{output_dir}/across_clients_summary.pkl", "wb") as f:
                pickle.dump(across_clients_summary, f)

            if across_clients_test_summary is not None:
                with open(f"{output_dir}/across_clients_test_summary.pkl", "wb") as f:
                    pickle.dump(across_clients_test_summary, f)

        return (
            metrics_per_client,
            test_metrics_per_client,
            test_all_metrics_per_client,
            test_across_clients_all_summary,
            across_clients_summary,
            across_clients_test_summary,
        )


class BatchLogger:
    """Implements a logger that holds the logger objects of all runs in a particular batch. On request this logger calculates the metrics per client across runs and across clients across runs."""

    def __init__(self, batch_name: str, id2dataset: dict, class_distributions):
        """Initializes a BatchLogger object (group of runs) with an empty dictionary (ment to house individual RunLoggers)

        Args:
            batch_name (str): a name for this group of runs
            id2dataset (dict): a mapping between dataset ids and their names
            class_distributions (dict): the number of examples per class in each of the datasets
        """
        self.batch_name = batch_name

        self.class_distributions = class_distributions
        self.id2dataset = id2dataset
        self.label2id, self.id2label = label_encoder()

        self.runs_in_batch = {}

    def add_run(self, run_logger: RunLogger):
        """Adds a RunLogger instance to the batch of runs.

        Args:
            run_logger (RunLogger): the logger instance associated with the run we want to add to the batch
        """

        self.runs_in_batch[run_logger.run_id] = run_logger

    def dump_to_file(self, output_dir: str):
        """Saves the dictionary containing RunLogger objects to disk.

        Args:
            outdir (str): path to the output directory on disk
        """

        with open(f"{output_dir}/{self.batch_name}_run_dump.pkl", "wb") as f:
            pickle.dump(self.runs_in_batch, f)

    def __plot_cms__(
        self, client_summary_across_runs, summary_across_clients_across_runs
    ) -> Tuple[plt.figure, plt.figure]:
        """Plots confusion matrices on a per client bases and across clients and across runs.

        Args:
            client_summary_across_runs (_type_): the per client summary metrics
            summary_across_clients_across_runs (_type_): the summary metrics across clients and runs

        Returns:
            Tuple[plt.figure, plt.figure]: the figures that contain the per client and across clients plots, respectively
        """

        per_client_fig = plt.figure(constrained_layout=True, figsize=(15, 15))
        per_client_subfigs = per_client_fig.subfigures(
            nrows=len(self.client_ids_in_run), ncols=1
        )

        for client_id, client_subfig in zip(self.client_ids_in_run, per_client_subfigs):

            client_mean_f1_score = client_summary_across_runs[
                f"client {self.id2dataset[client_id]} cv_f1_macro_mean"
            ]
            client_std_f1_score = client_summary_across_runs[
                f"client {self.id2dataset[client_id]} cv_f1_macro_std"
            ]

            client_subfig.suptitle(
                f"Client: {self.id2dataset[client_id]} --- CV Macro F1-score: {client_mean_f1_score:.3f} ({client_std_f1_score:.3f})"
            )
            client_axs = client_subfig.subplots(nrows=1, ncols=4)

            _ = sns.heatmap(
                client_summary_across_runs[
                    f"client {self.id2dataset[client_id]} cv_mean_cm"
                ],
                annot=True,
                cmap="crest",
                xticklabels=[v for _, v in self.id2label.items()],
                yticklabels=[v for _, v in self.id2label.items()],
                linewidth=0.5,
                fmt="g",
                ax=client_axs[0],
            )
            _ = sns.heatmap(
                client_summary_across_runs[
                    f"client {self.id2dataset[client_id]} cv_std_cm"
                ],
                annot=True,
                cmap="crest",
                xticklabels=[v for _, v in self.id2label.items()],
                yticklabels=[v for _, v in self.id2label.items()],
                linewidth=0.5,
                fmt="g",
                ax=client_axs[1],
            )
            _ = sns.heatmap(
                client_summary_across_runs[
                    f"client {self.id2dataset[client_id]} cv_norm_cm"
                ],
                annot=True,
                cmap="crest",
                xticklabels=[v for _, v in self.id2label.items()],
                yticklabels=[v for _, v in self.id2label.items()],
                linewidth=0.5,
                fmt="g",
                ax=client_axs[2],
            )

            _ = sns.countplot(
                data=pd.DataFrame(
                    self.class_distributions[self.id2dataset[client_id]],
                    columns=["label"],
                ),
                x="label",
                ax=client_axs[3],
            )

            client_axs[0].set_title(f"Mean CM")
            client_axs[1].set_title(f"Std CM")
            client_axs[2].set_title(f"Norm CM")
            client_axs[3].set_title(f"Class distribution")

        # plot and save the confusion matrices across clients and runs
        across_clients_fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        across_clients_fig.suptitle("CV Confusion matrices across clients and runs")

        _ = sns.heatmap(
            summary_across_clients_across_runs["cv_mean_cm"],
            annot=True,
            cmap="crest",
            xticklabels=[v for _, v in self.id2label.items()],
            yticklabels=[v for _, v in self.id2label.items()],
            linewidth=0.5,
            fmt="g",
            ax=ax[0],
        )
        _ = sns.heatmap(
            summary_across_clients_across_runs["cv_std_cm"],
            annot=True,
            cmap="crest",
            xticklabels=[v for _, v in self.id2label.items()],
            yticklabels=[v for _, v in self.id2label.items()],
            linewidth=0.5,
            fmt="g",
            ax=ax[1],
        )
        _ = sns.heatmap(
            summary_across_clients_across_runs["cv_norm_cm"],
            annot=True,
            cmap="crest",
            xticklabels=[v for _, v in self.id2label.items()],
            yticklabels=[v for _, v in self.id2label.items()],
            linewidth=0.5,
            fmt="g",
            ax=ax[2],
        )

        ax[0].set_title("Mean CM")
        ax[1].set_title("Std CM")
        ax[2].set_title("Norm CM")

        return per_client_fig, across_clients_fig

    def __plot_cms__test__(
        self,
        client_test_all_summary_across_runs,
        test_all_metrics_across_clients_across_runs,
    ) -> Tuple[plt.figure, plt.figure]:
        """Plots confusion matrices on a per client bases and across clients and across runs for the test set

        Args:
            client_test_metrics_across_runs (_type_): the per client summary metrics
        Returns:
            Tuple[plt.figure, plt.figure]: the figures that contain the per client and across clients plots, respectively
        """

        per_client_fig = plt.figure(constrained_layout=True, figsize=(15, 15))
        per_client_subfigs = per_client_fig.subfigures(
            nrows=len(self.client_ids_in_run), ncols=1
        )
        
        for client_id, client_subfig in zip(self.client_ids_in_run, per_client_subfigs):

            client_mean_f1_score = test_all_metrics_across_clients_across_runs[
                f"client {self.id2dataset[client_id]} test_f1_macro_mean"
            ]
           
            client_std_f1_score = test_all_metrics_across_clients_across_runs[
                f"client {self.id2dataset[client_id]} test_f1_macro_std"
            ]

            client_subfig.suptitle(
                f"Client: {self.id2dataset[client_id]} --- Test Macro F1-score: {client_mean_f1_score:.3f} ({client_std_f1_score:.3f})"
            )
            client_axs = client_subfig.subplots(nrows=1, ncols=4)

            _ = sns.heatmap(
                test_all_metrics_across_clients_across_runs[
                    f"client {self.id2dataset[client_id]} test_mean_cm"
                ],
                annot=True,
                cmap="crest",
                xticklabels=[v for _, v in self.id2label.items()],
                yticklabels=[v for _, v in self.id2label.items()],
                linewidth=0.5,
                fmt="g",
                ax=client_axs[0],
            )
            _ = sns.heatmap(
                test_all_metrics_across_clients_across_runs[
                    f"client {self.id2dataset[client_id]} test_std_cm"
                ],
                annot=True,
                cmap="crest",
                xticklabels=[v for _, v in self.id2label.items()],
                yticklabels=[v for _, v in self.id2label.items()],
                linewidth=0.5,
                fmt="g",
                ax=client_axs[1],
            )
            _ = sns.heatmap(
                test_all_metrics_across_clients_across_runs[
                    f"client {self.id2dataset[client_id]} test_norm_cm"
                ],
                annot=True,
                cmap="crest",
                xticklabels=[v for _, v in self.id2label.items()],
                yticklabels=[v for _, v in self.id2label.items()],
                linewidth=0.5,
                fmt="g",
                ax=client_axs[2],
            )

            _ = sns.countplot(
                data=pd.DataFrame(
                    self.class_distributions[self.id2dataset[client_id]],
                    columns=["label"],
                ),
                x="label",
                ax=client_axs[3],
            )

            client_axs[0].set_title(f"Mean CM")
            client_axs[1].set_title(f"Std CM")
            client_axs[2].set_title(f"Norm CM")
            client_axs[3].set_title(f"Class distribution")

            # plot and save the confusion matrices across clients and runs
        #across_clients_fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        #across_clients_fig.suptitle("Test Confusion matrices across clients and runs")
#
        #_ = sns.heatmap(
        #    client_test_all_summary_across_runs["mean_cm"],
        #    annot=True,
        #    cmap="crest",
        #    xticklabels=[v for _, v in self.id2label.items()],
        #    yticklabels=[v for _, v in self.id2label.items()],
        #    linewidth=0.5,
        #    fmt="g",
        #    ax=ax[0],
        #)
        #_ = sns.heatmap(
        #    client_test_all_summary_across_runs["std_cm"],
        #    annot=True,
        #    cmap="crest",
        #    xticklabels=[v for _, v in self.id2label.items()],
        #    yticklabels=[v for _, v in self.id2label.items()],
        #    linewidth=0.5,
        #    fmt="g",
        #    ax=ax[1],
        #)
        #_ = sns.heatmap(
        #    client_test_all_summary_across_runs["norm_cm"],
        #    annot=True,
        #    cmap="crest",
        #    xticklabels=[v for _, v in self.id2label.items()],
        #    yticklabels=[v for _, v in self.id2label.items()],
        #    linewidth=0.5,
        #    fmt="g",
        #    ax=ax[2],
        #)
#
        #ax[0].set_title("Mean CM")
        #ax[1].set_title("Std CM")
        #ax[2].set_title("Norm CM")
        #return per_client_fig, across_clients_fig
        return per_client_fig
#
    def _concatenate_client_test_results_across_runs(self):

        concatenated_client_test_results_across_runs = {}

        # iterate over the runs in this batch
        for _, run_logger in self.runs_in_batch.items():
            for (
                client_id,
                client_test_results,
            ) in run_logger.client_test_results.items():

                if client_id not in concatenated_client_test_results_across_runs:
                    concatenated_client_test_results_across_runs[client_id] = {
                        "groundtruths": None,
                        "predictions": None,
                        "prediction_scores": None,
                    }

                if (
                    concatenated_client_test_results_across_runs[client_id][
                        "groundtruths"
                    ]
                    is None
                ):
                    concatenated_client_test_results_across_runs[client_id][
                        "groundtruths"
                    ] = client_test_results["groundtruths"].copy()
                    concatenated_client_test_results_across_runs[client_id][
                        "predictions"
                    ] = client_test_results["predictions"].copy()
                    concatenated_client_test_results_across_runs[client_id][
                        "prediction_scores"
                    ] = client_test_results["prediction_scores"].copy()
                else:
                    concatenated_client_test_results_across_runs[client_id][
                        "groundtruths"
                    ] = np.concatenate(
                        [
                            concatenated_client_test_results_across_runs[client_id][
                                "groundtruths"
                            ],
                            client_test_results["groundtruths"],
                        ],
                        axis=0,
                    )
                    concatenated_client_test_results_across_runs[client_id][
                        "predictions"
                    ] = np.concatenate(
                        [
                            concatenated_client_test_results_across_runs[client_id][
                                "predictions"
                            ],
                            client_test_results["predictions"],
                        ],
                        axis=0,
                    )
                    concatenated_client_test_results_across_runs[client_id][
                        "prediction_scores"
                    ] = np.concatenate(
                        [
                            concatenated_client_test_results_across_runs[client_id][
                                "prediction_scores"
                            ],
                            client_test_results["prediction_scores"],
                        ],
                        axis=0,
                    )

        return concatenated_client_test_results_across_runs

    def __concat_client_test_all_summary_results_across(self):

        test_client_all_metrics = {}

        # iterate over the runs in this batch
        for _, run_logger in self.runs_in_batch.items():
            _, _, test_across_clients_all_summary, _, _, _ = run_logger.get_results_run(
                output_dir=None
            )

            if self.client_ids_in_run is None:
                self.client_ids_in_run = list(test_across_clients_all_summary.keys())
            # iterate over the clients
            for client_id, client_results in test_across_clients_all_summary.items():

                if client_id not in test_client_all_metrics:
                    test_client_all_metrics[client_id] = {}
                # iterate over the metrics for the client
                for metric_name, metric_value in client_results.items():
                    if isinstance(metric_value, float):
                        # if it is not present in the  test_client_all_metrics[client_id] dictionary, introduce it as a FloatingMetricSummary object
                        if metric_name not in test_client_all_metrics[client_id]:
                            test_client_all_metrics[client_id][
                                f"{metric_name}"
                            ] = FloatingMetricSummary()
                        test_client_all_metrics[client_id][f"{metric_name}"].add_result(
                            metric_value
                        )
                    elif isinstance(metric_value, np.ndarray):
                        # if it is not present in the  test_client_all_metrics[client_id] dictionary, introduce it as a list object
                        if metric_name not in test_client_all_metrics[client_id]:
                            test_client_all_metrics[client_id][f"{metric_name}"] = []
                        test_client_all_metrics[client_id][f"{metric_name}"].append(
                            metric_value
                        )
        # * extract the average and standard deviation of all the per client metrics recorded in the `client_cv_metrics` dictionary
        ########################################################################
        # this should be a dictionary with only one level so metrics will be renamed using the following rule: new_name = "client X cv_{metric_name}_(mean/std)"
        client_test_all_summary_across_runs = {}

        # iterate over the clients
        for client_id, client_metrics_across_runs in test_client_all_metrics.items():
            # iterate over the client's metrics
            for metric_name, metric_summary in client_metrics_across_runs.items():
                if isinstance(metric_summary, FloatingMetricSummary):
                    metric_mean, metric_std = metric_summary.get_summary()
                    client_test_all_summary_across_runs[
                        f"client {self.id2dataset[client_id]} test_{metric_name}_mean"
                    ] = metric_mean
                    client_test_all_summary_across_runs[
                        f"client {self.id2dataset[client_id]} test_{metric_name}_std"
                    ] = metric_std
                else:
                    # it can only be a list of ndarrays
                    stacked_arrays = np.stack(metric_summary, axis=2)

                    mean_cm = np.mean(stacked_arrays, axis=2)
                    std_cm = np.std(stacked_arrays, axis=2)

                    inter = mean_cm.sum(axis=1)[:, np.newaxis]
                    inter[inter == 0] = np.inf
                    norm_cm = mean_cm / inter

                    client_test_all_summary_across_runs[
                        f"client {self.id2dataset[client_id]} test_mean_cm"
                    ] = mean_cm
                    client_test_all_summary_across_runs[
                        f"client {self.id2dataset[client_id]} test_std_cm"
                    ] = std_cm
                    client_test_all_summary_across_runs[
                        f"client {self.id2dataset[client_id]} test_norm_cm"
                    ] = norm_cm
        ########################################################################
        ########################################################################

        ########################################################################
        # * Calculate and summarize the results across clients and across runs
        ########################################################################
        # these are just the averages and standard deviations of the mean values of the metrics recorded per client across runs
        ########################################################################
        test_all_metrics_across_clients_across_runs = {}

        # iterate through the summary metrics
        for (
            full_metric_name,
            metric_value,
        ) in client_test_all_summary_across_runs.items():
            # only consider a summary metric if it is the mean value
            if "mean" in full_metric_name:
                if isinstance(metric_value, float):
                    metric_name = "_".join(
                        full_metric_name.split(" ")[2].split("_")[:-1]
                    )

                    if metric_name not in test_all_metrics_across_clients_across_runs:
                        test_all_metrics_across_clients_across_runs[metric_name] = (
                            FloatingMetricSummary()
                        )

                    test_all_metrics_across_clients_across_runs[metric_name].add_result(
                        metric_value
                    )
                elif isinstance(metric_value, np.ndarray):
                    metric_name = "cm"

                    if metric_name not in test_all_metrics_across_clients_across_runs:
                        test_all_metrics_across_clients_across_runs[metric_name] = []

                    test_all_metrics_across_clients_across_runs[metric_name].append(
                        metric_value
                    )

        return (
            client_test_all_summary_across_runs,
            test_all_metrics_across_clients_across_runs,
        )

    def get_results_batch(
        self,
        output_dir: str = None,
        save_plots: bool = False,
        upload_to_wandb: bool = False,
        eval_on_test: bool = False,
    ) -> Tuple[dict, dict]:
        """Retrieves and summarizes the per client and across clients resuts across all runs.
        Args:
            output_dir (str, optional): the directory where the results for each run should be stored. If this is None, the results are not stored
            save_plots (bool, optional): if True, plots are saved
            upload_to_wandb (bool, optional): if True, uploads all metrics to W&B as well

        Returns:
            tuple[dict, dict]: A tuple containing three dictionaries. The first dictionary contains the summarized
            results per client and across runs. The second dictionary contains the summarized results across clients and across runs.
        """

        self.client_ids_in_run = None

        if output_dir is not None:
            if not os.path.exists(f"{output_dir}/{self.batch_name}"):
                os.mkdir(f"{output_dir}/{self.batch_name}")

        ########################################################################
        # * calculate and summarize the per client CV results across runs
        ########################################################################
        # these are just the averages and standard deviations across runs of all per-client metrics
        ########################################################################
        client_cv_metrics = {}

        # iterate over the runs in this batch
        for _, run_logger in self.runs_in_batch.items():
            run_metrics_per_client, _, _, _, _, _ = run_logger.get_results_run(
                output_dir=None
            )

            if self.client_ids_in_run is None:
                self.client_ids_in_run = list(run_metrics_per_client.keys())
            # iterate over the clients
            for client_id, client_results in run_metrics_per_client.items():
                if client_id not in client_cv_metrics:
                    client_cv_metrics[client_id] = {}
                # iterate over the metrics for the client
                for metric_name, metric_value in client_results.items():
                    if isinstance(metric_value, float):
                        # if it is not present in the  client_cv_metrics[client_id] dictionary, introduce it as a FloatingMetricSummary object
                        if metric_name not in client_cv_metrics[client_id]:
                            client_cv_metrics[client_id][
                                f"{metric_name}"
                            ] = FloatingMetricSummary()
                        client_cv_metrics[client_id][f"{metric_name}"].add_result(
                            metric_value
                        )
                    elif isinstance(metric_value, np.ndarray):
                        # if it is not present in the  client_cv_metrics[client_id] dictionary, introduce it as a list object
                        if metric_name not in client_cv_metrics[client_id]:
                            client_cv_metrics[client_id][f"{metric_name}"] = []
                        client_cv_metrics[client_id][f"{metric_name}"].append(
                            metric_value
                        )

        # * extract the average and standard deviation of all the per client metrics recorded in the `client_cv_metrics` dictionary
        ########################################################################
        # this should be a dictionary with only one level so metrics will be renamed using the following rule: new_name = "client X cv_{metric_name}_(mean/std)"
        client_cv_summary_across_runs = {}

        # iterate over the clients
        for client_id, client_metrics_across_runs in client_cv_metrics.items():
            # iterate over the client's metrics
            for metric_name, metric_summary in client_metrics_across_runs.items():
                if isinstance(metric_summary, FloatingMetricSummary):
                    metric_mean, metric_std = metric_summary.get_summary()
                    client_cv_summary_across_runs[
                        f"client {self.id2dataset[client_id]} cv_{metric_name}_mean"
                    ] = metric_mean
                    client_cv_summary_across_runs[
                        f"client {self.id2dataset[client_id]} cv_{metric_name}_std"
                    ] = metric_std
                else:
                    # it can only be a list of ndarrays
                    stacked_arrays = np.stack(metric_summary, axis=2)

                    mean_cm = np.mean(stacked_arrays, axis=2)
                    std_cm = np.std(stacked_arrays, axis=2)

                    inter = mean_cm.sum(axis=1)[:, np.newaxis]
                    inter[inter == 0] = np.inf
                    norm_cm = mean_cm / inter

                    client_cv_summary_across_runs[
                        f"client {self.id2dataset[client_id]} cv_mean_cm"
                    ] = mean_cm
                    client_cv_summary_across_runs[
                        f"client {self.id2dataset[client_id]} cv_std_cm"
                    ] = std_cm
                    client_cv_summary_across_runs[
                        f"client {self.id2dataset[client_id]} cv_norm_cm"
                    ] = norm_cm
        ########################################################################
        ########################################################################

        ########################################################################
        # * Calculate and summarize the results across clients and across runs
        ########################################################################
        # these are just the averages and standard deviations of the mean values of the metrics recorded per client across runs
        ########################################################################
        cv_metrics_across_clients_across_runs = {}

        # iterate through the summary metrics
        for full_metric_name, metric_value in client_cv_summary_across_runs.items():
            # only consider a summary metric if it is the mean value
            if "mean" in full_metric_name:
                if isinstance(metric_value, float):
                    metric_name = "_".join(
                        full_metric_name.split(" ")[2].split("_")[:-1]
                    )

                    if metric_name not in cv_metrics_across_clients_across_runs:
                        cv_metrics_across_clients_across_runs[metric_name] = (
                            FloatingMetricSummary()
                        )

                    cv_metrics_across_clients_across_runs[metric_name].add_result(
                        metric_value
                    )
                elif isinstance(metric_value, np.ndarray):
                    metric_name = "cm"

                    if metric_name not in cv_metrics_across_clients_across_runs:
                        cv_metrics_across_clients_across_runs[metric_name] = []

                    cv_metrics_across_clients_across_runs[metric_name].append(
                        metric_value
                    )

        # * extract the average and standard deviation of all the metrics recorded in the `cv_metrics_across_clients_across_runs` dictionary
        ########################################################################
        # the names of the metrics in this dictionary are build according to the following rule: "{metric_name}_mean"
        cv_summary_across_clients_across_runs = {}

        # iterate over the metrics, extract the summaries
        for (
            metric_name,
            metric_summary,
        ) in cv_metrics_across_clients_across_runs.items():
            if isinstance(metric_summary, FloatingMetricSummary):
                metric_mean, metric_std = metric_summary.get_summary()
                cv_summary_across_clients_across_runs[f"{metric_name}_mean"] = (
                    metric_mean
                )
                cv_summary_across_clients_across_runs[f"{metric_name}_std"] = metric_std
            else:
                # it can only be a list of ndarrays
                stacked_arrays = np.stack(metric_summary, axis=2)

                mean_cm = np.mean(stacked_arrays, axis=2)
                std_cm = np.std(stacked_arrays, axis=2)

                inter = mean_cm.sum(axis=1)[:, np.newaxis]
                inter[inter == 0] = np.inf
                norm_cm = mean_cm / inter

                cv_summary_across_clients_across_runs[f"cv_mean_cm"] = mean_cm
                cv_summary_across_clients_across_runs[f"cv_std_cm"] = std_cm
                cv_summary_across_clients_across_runs[f"cv_norm_cm"] = norm_cm

        # * extract a weighted macro F-score
        ########################################################################
        client_cv_f1_macro_scores = []
        client_cv_num_examples = []
        num_cv_examples_across_clients = 0.0

        for client_id in client_cv_metrics.keys():
            client_cv_f1_macro_scores.append(
                client_cv_summary_across_runs[
                    f"client {self.id2dataset[client_id]} cv_f1_macro_mean"
                ]
            )
            client_cv_num_examples.append(
                len(self.class_distributions[self.id2dataset[client_id]])
            )
            num_cv_examples_across_clients += len(
                self.class_distributions[self.id2dataset[client_id]]
            )

        client_cv_weights = (
            np.array(client_cv_num_examples) / num_cv_examples_across_clients
        )

        cv_summary_across_clients_across_runs["cv_weighted_f1_macro_mean"] = np.average(
            client_cv_f1_macro_scores, weights=client_cv_weights
        )
        cv_summary_across_clients_across_runs["cv_weighted_f1_macro_std"] = np.sqrt(
            np.average(
                (
                    client_cv_f1_macro_scores
                    - cv_summary_across_clients_across_runs["cv_weighted_f1_macro_mean"]
                )
                ** 2,
                weights=client_cv_weights,
            )
        )
        ########################################################################

        ########################################################################
        ########################################################################

        # calculate the client test metrics across runs
        ########################################################################
        client_test_results_across_runs = (
            self._concatenate_client_test_results_across_runs()
        )

        (
            test_all_metrics_across_clients_across_runs,
            client_test_all_summary_across_runs,
        ) = self.__concat_client_test_all_summary_results_across()

        client_test_metrics_across_runs = {}

        test_metrics_across_clients_and_runs = None

        if eval_on_test == True:

            test_results_across_runs = {
                "groundtruths": None,
                "predictions": None,
                "prediction_scores": None,
                "client_ids": None,
            }
            for (
                client_id,
                client_results_across_runs,
            ) in client_test_results_across_runs.items():
                client_test_metrics_across_runs[self.id2dataset[client_id]] = (
                    calculate_metrics_with_cis(
                        client_results_across_runs["groundtruths"],
                        client_results_across_runs["predictions"],
                    )
                )
            ########################################################################
            ########################################################################

            # calculate the test metrics across clients and across runs
            ########################################################################
            for (
                client_id,
                client_results_across_runs,
            ) in client_test_results_across_runs.items():

                if test_results_across_runs["groundtruths"] is None:
                    test_results_across_runs["groundtruths"] = (
                        client_results_across_runs["groundtruths"].copy()
                    )

                    test_results_across_runs["predictions"] = (
                        client_results_across_runs["predictions"].copy()
                    )
                    test_results_across_runs["prediction_scores"] = (
                        client_results_across_runs["prediction_scores"].copy()
                    )
                    test_results_across_runs["client_ids"] = [
                        client_id
                    ] * client_results_across_runs["groundtruths"].shape[0]
                else:
                    test_results_across_runs["groundtruths"] = np.concatenate(
                        [
                            test_results_across_runs["groundtruths"],
                            client_results_across_runs["groundtruths"],
                        ],
                        axis=0,
                    )
                    test_results_across_runs["predictions"] = np.concatenate(
                        [
                            test_results_across_runs["predictions"],
                            client_results_across_runs["predictions"],
                        ],
                        axis=0,
                    )
                    test_results_across_runs["prediction_scores"] = np.concatenate(
                        [
                            test_results_across_runs["prediction_scores"],
                            client_results_across_runs["prediction_scores"],
                        ],
                        axis=0,
                    )
                    test_results_across_runs["client_ids"] = np.concatenate(
                        [
                            test_results_across_runs["client_ids"],
                            [client_id]
                            * client_results_across_runs["groundtruths"].shape[0],
                        ],
                        axis=0,
                    )

            test_metrics_across_clients_and_runs = calculate_metrics_with_cis(
                test_results_across_runs["groundtruths"],
                test_results_across_runs["predictions"],
                conditions=test_results_across_runs["client_ids"],
            )

            test_metrics_across_clients_and_runs_flattened = flatten_dict(
                test_metrics_across_clients_and_runs
            )

            client_test_metrics_across_runs_flattened = flat_nested_dict(
                client_test_metrics_across_runs
            )

        # save the results to disk
        if output_dir is not None:
            if not os.path.exists(f"{output_dir}/{self.batch_name}"):
                os.mkdir(f"{output_dir}/{self.batch_name}")

            with open(
                f"{output_dir}/{self.batch_name}/client_cv_summary.pkl", "wb"
            ) as f:
                pickle.dump(client_cv_summary_across_runs, f)

            with open(
                f"{output_dir}/{self.batch_name}/across_clients_cv_summary.pkl", "wb"
            ) as f:
                pickle.dump(cv_summary_across_clients_across_runs, f)

            if eval_on_test == True:
                if client_test_metrics_across_runs:
                    with open(
                        f"{output_dir}/{self.batch_name}/client_test_summary.pkl", "wb"
                    ) as f:
                        pickle.dump(client_test_metrics_across_runs, f)

                    with open(
                        f"{output_dir}/{self.batch_name}/across_clients_test_summary.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(test_metrics_across_clients_and_runs, f)

            if save_plots:
                per_client_fig, across_clients_fig = self.__plot_cms__(
                    client_cv_summary_across_runs, cv_summary_across_clients_across_runs
                )

                per_client_fig.savefig(
                    f"{output_dir}/{self.batch_name}/per_client_cv_confusion_matrices.png",
                    dpi=400,
                )
                across_clients_fig.savefig(
                    f"{output_dir}/{self.batch_name}/across_clients_cv_confusion_matrices.png",
                    dpi=400,
                )

                if eval_on_test == True:
            
                    test_per_client_fig = (
                        self.__plot_cms__test__(
                            client_test_all_summary_across_runs,
                            test_all_metrics_across_clients_across_runs,
                        )
                    )

                    test_per_client_fig.savefig(
                        f"{output_dir}/{self.batch_name}/test_per_client_confusion_matrices.png",
                        dpi=400,
                    )

                    #test_across_clients_fig.savefig(
                    #    f"{output_dir}/{self.batch_name}/test_across_client_confusion_matrices.png",
                    #    dpi=400,
                    #)

        # upload the results to W&B as well
        if upload_to_wandb:
            wandb_summary = {}
            combined_summary = client_cv_summary_across_runs.copy()
            combined_summary.update(cv_summary_across_clients_across_runs)
            if eval_on_test == True:
                combined_summary.update(client_test_metrics_across_runs_flattened)
                combined_summary.update(test_metrics_across_clients_and_runs_flattened)

            for key, value in combined_summary.items():
                if "cm" not in key:
                    wandb_summary[key] = value

            per_client_fig, across_clients_fig = self.__plot_cms__(
                client_cv_summary_across_runs, cv_summary_across_clients_across_runs
            )

            wandb.log(
                {
                    "Per Client (CV) Confusion Matrix Summary": wandb.Image(
                        per_client_fig
                    )
                }
            )
            wandb.log(
                {
                    "Across Clients (CV) Confusion Matrix Summary": wandb.Image(
                        across_clients_fig
                    )
                }
            )

            if eval_on_test == True:
                test_per_client_fig = self.__plot_cms__test__(
                    client_test_all_summary_across_runs,
                    test_all_metrics_across_clients_across_runs,
                )
                wandb.log(
                    {
                        "Test Per Clients Confusion Matrix Summary": wandb.Image(
                            test_per_client_fig
                        )
                    }
                )

                #wandb.log(
                #    {
                #        "Test Across Clients Confusion Matrix Summary": wandb.Image(
                #            test_across_clients_fig
                #        )
                #    }
                #)

            # finally, log the wanb suitable summary
            wandb.log(wandb_summary)

        return (
            client_cv_summary_across_runs,
            cv_summary_across_clients_across_runs,
            client_test_metrics_across_runs,
            test_metrics_across_clients_and_runs,
        )
