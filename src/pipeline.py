"""
Module Name: pipeline.py
Description: The module defines an abstract pipeline class and two implementations of this class in the form of a non-federated and federated pipeline

Classes & Methods:
Pipeline: 

#! Notes/to-dos:
#! 1. make personalization rounds a separate config parameter
"""

import os
import random
import math
from abc import ABC, abstractclassmethod
from typing import List, Tuple, Dict, Callable
from types import SimpleNamespace

import numpy as np
import pandas as pd

import ray
import torch
import audeer
import wandb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import flwr as fl

import utils
import model_utils
import preprocessing_utils
import loggers
from dataset import CIDataset
from torch.utils.data import DataLoader
from federated_client import create_client_with_args

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"


class Pipeline(ABC):
    """An abstract pipeline class that implements a big part of the boilerplate."""

    def __init__(self, config: SimpleNamespace, tuning: bool):
        self.config = config
        self.tuning = tuning

        # * Intialize the device the pipeline will be using
        self._initialize_device(self.config.device_id)

        # * Load the data from disk
        self._load_data()

        # * Transform the 'dataset' column
        self._transform_dataset_column()

        self._separate_train_test()

        # * Extract class distributions per client
        self._extract_client_distributions()

        # * Encode the values in the label column
        self._transform_label_column()

        # * Initialize logging for this batch of runs by creating an output directory and a batch logger
        self._initialize_logging()

    def _initialize_device(self, device_id: str):
        """Initializes a device for use with Pytorch. Also initializes ray with the right amount of cpus.

        Args:
            device_id (str): the id of the gpu to use, if available
        """
        self.device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        if "SLURM_CPUS_PER_TASK" in os.environ:
            ray.init(
                num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK")),
                include_dashboard=False,
            )
        else:
            ray.init(num_cpus=4, include_dashboard=False)

    def _load_data(self):
        """Loads preprocessed data or preprocesses the data and saves it to disk."""

        # path_to_features can be empty if features should be extracted, otherwise, features are expected to already be there
        if hasattr(self.config, "path_to_features"):
            print("Loading preprocessed data...")
            (
                self.processed_data,
                self.config.input_size,
                self.feature_columns,
                self.config.num_unique_labels,
            ) = utils.load_data(
                None,
                self.config.path_to_features,
                self.config.target_class,
                self.device,
                False,
            )
        elif hasattr(self.config, "path_to_original"):
            # create the feature output directory
            feature_output_dir = audeer.mkdir(
                f"{self.config.feature_output_dir}/{self.config.target_class}_{self.config.feature_type}/"
            )

            # load the original data and preprocess it
            (
                self.processed_data,
                self.config.input_size,
                self.feature_columns,
                self.config.num_unique_labels,
            ) = utils.load_data(
                self.config.path_to_original,
                feature_output_dir,
                self.config.feature_type,
                self.config.target_class,
                self.device,
                self.config.number_variance_pca,
                self.config.number_components,
                self.config.pca,
                preprocess=True,
            )
        else:
            print(
                "[ERROR] The user needs to provide either the path to the original data and a path to where extracted features should be saved, or provide a path to where already extracted features are saved"
            )
            exit()

    def _transform_dataset_column(self):
        """Combines the Pitt corpus and the address dataset and encodes the dataset values."""
        # combine the `pitt_corpus` dataset and the adress dataset
        combine_datasests = {"pitt_corpus": "pitt_adress", "adress": "pitt_adress"}
        self.processed_data["dataset"] = (
            self.processed_data["dataset"]
            .map(combine_datasests)
            .fillna(self.processed_data["dataset"], downcast="infer")
        )
        self.processed_data = self.processed_data[
            self.processed_data["dataset"] != "baycrest"
        ]

        # construct a mapping dictionary between data set names and ids
        self.dataset2id = {}
        sorted_datasets = sorted(self.processed_data["dataset"].unique())

        for dataset_idx, dataset in enumerate(sorted_datasets):
            self.dataset2id[dataset] = str(dataset_idx)
        # construct the inverse mapping
        self.id2dataset = {id: speaker for speaker, id in self.dataset2id.items()}

        # apply the mapping to the 'dataset' column
        self.processed_data["dataset"] = self.processed_data["dataset"].map(
            self.dataset2id
        )

        print(
            "The unique datasets/clients encountered are:",
            self.processed_data["dataset"].unique(),
        )

    def _separate_train_test(self):
        """Separates the training and test data"""
        self.processed_train_data = self.processed_data[
            self.processed_data.split == "train"
        ]
        self.processed_test_data = self.processed_data[
            self.processed_data.split == "test"
        ]

        # delete the combined processed data as safety
        del self.processed_data

    def _extract_client_distributions(self):
        """Extracts the class distributions per data set/client."""
        self.class_distributions = {}

        for client in self.processed_train_data["dataset"].unique():
            dataset_labels = self.processed_train_data[
                self.processed_train_data["dataset"] == client
            ]["label"]
            self.class_distributions[self.id2dataset[client]] = dataset_labels

    def _transform_label_column(self):
        """Encodes the values in the label column."""
        self.label2id, self.id2label = preprocessing_utils.label_encoder()
        self.processed_train_data["label"] = (
            self.processed_train_data["label"].map(self.label2id).astype(int)
        )
        print(
            "\nUnique labels encountered after mapping in the training set:",
            np.unique(self.processed_train_data["label"], return_counts=True),
        )

        self.processed_test_data["label"] = (
            self.processed_test_data["label"].map(self.label2id).astype(int)
        )
        print(
            "\nUnique labels encountered after mapping in the test set:",
            np.unique(self.processed_train_data["label"], return_counts=True),
        )

    def _initialize_logging(self):
        """Creates an output directory and initializes a BatchLogger."""

        # create the basic output directory if it does not exist
        self.output_dir = f"{self.config.results_dir}/{self.config.target_class}_{self.config.feature_type}"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # initialize a BatchLogging object
        self.batch_logger = loggers.BatchLogger(
            wandb.run.name, self.id2dataset, self.class_distributions
        )

    # * method not called during initialization
    def _set_random_seed(self, random_seed: int):
        """Sets the random number generator seed in several libraries.

        Args:
            random_seed (int): the random seed
        """
        # set the python, numpy and torch random seeds, just to be sure
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def _create_client_train_splits(
        self,
    ) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """Splits the data of the clients into folds.

        Returns:
            Dict[str, List[Tuple[np.ndarray, np.ndarray]]]: a dictionary containing the splits per client
        """
        client_splits = {}

        for client in self.id2dataset.keys():
            client_data = self.processed_train_data[
                self.processed_train_data["dataset"] == client
            ]

            # count the number of example this client has
            num_client_examples = client_data.shape[0]

            # create a StratifiedKFold generator for this dataset/client
            split_generator = StratifiedKFold(
                n_splits=self.config.num_cv_folds,
                random_state=self.random_seed,
                shuffle=True,
            )

            # generate the splits and add them to the dictionary
            for client_train_indices, client_val_indices in split_generator.split(
                np.zeros(num_client_examples), client_data["label"]
            ):
                if client not in client_splits:
                    client_splits[client] = []

                client_splits[client].append((client_train_indices, client_val_indices))

        return client_splits

    def _prepare_run(
        self, run_id: int
    ) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], loggers.RunLogger]:
        """Sets the global random seed, creates a run logger and creates data splits for each client.

        Args:
            run_id (int): the ID of the run we are initializing

        Returns:
            Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], loggers.RunLogger]: the clients' data splits and the run logger, respectively
        """
        # set the random seed
        self._set_random_seed(run_id)

        # initialize a RunLogger
        run_logger = loggers.RunLogger(f"run_{run_id}", self.id2dataset.keys())

        # generate client splits
        client_splits = self._create_client_train_splits()

        return client_splits, run_logger

    @abstractclassmethod
    def _train(self, eval_on_test: bool):
        pass

    def _summarize_results(self):
        """Summarizes the results from all the runs and outputs them if the pipeline is not in tuning mode. Otherwise just uploads the summary to W&B."""
        if not self.tuning:
            (
                client_summary_across_runs,
                summary_across_clients_across_runs,
                client_test_summary_across_runs,
                test_summary_across_clients_across_runs,
            ) = self.batch_logger.get_results_batch(
                output_dir=self.output_dir,
                save_plots=True,
                upload_to_wandb=True,
                eval_on_test=self.config.eval_on_test,
            )
        else:
            (
                client_summary_across_runs,
                summary_across_clients_across_runs,
                client_test_summary_across_runs,
                test_summary_across_clients_across_runs,
            ) = self.batch_logger.get_results_batch(
                output_dir=None,
                save_plots=False,
                upload_to_wandb=True,
                eval_on_test=self.config.eval_on_test,
            )

        if not self.tuning:
            print("Client-level summary:")
            print("######################")
            print(client_summary_across_runs)
            print("######################")
            print("######################")
            print("Summary across clients and runs:")
            print("######################")
            print(summary_across_clients_across_runs)
            print("######################")
            print("######################")

        if not self.tuning:
            if self.config.eval_on_test == True:
                print("Client-level summary on the test set:")
                print("######################")
                print(client_test_summary_across_runs)
                print("######################")
                print("######################")
                print("Test summary across clients and runs:")
                print(test_summary_across_clients_across_runs)
                print("######################")
                print("######################")

    def execute(self, eval_on_test: bool = False):
        """Calls the `train` and `summarize_results` methods.

        Args:
            eval_on_test (bool, optional): Whether to evaluate the models on the test set. Defaults to False.
        """
        self._train(eval_on_test)
        self._summarize_results()


class NonFederatedPipeline(Pipeline):
    """A non-federated implementation of the abstract class Pipeline.

    This implementation includes a:
        - cetralized learning pipeline
        - local learning pipeline
    """

    def _train_eval_mlp(
        self, train: CIDataset, val: CIDataset, class_weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trains a multilayer perceptron on the provided training data and evaluates it on the validation data.

        Args:
            train (CIDataset): the training data
            val (CIDataset): the validation data
            class_weights (np.ndarray): how to reweight the classes during training

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: the groundtruths, predictions and prediction scores of the model on the validation set, respectively
        """

        # create a dataloader for the two data sets
        train_dl = torch.utils.data.DataLoader(
            train, batch_size=self.config.batch_size, shuffle=self.config.shuffle
        )
        # keeping shuffle False in the validation set to add the predictions and label later
        val_dl = torch.utils.data.DataLoader(
            val, batch_size=self.config.batch_size, shuffle=False
        )

        # get model
        model = model_utils.get_model(
            self.config.input_size,
            self.config.num_neurons_per_layer,
            self.config.num_unique_labels,
            self.config.dropout_prob,
        )
        # train model
        _ = model_utils.train_model(
            model, train_dl, class_weights, self.config, self.device
        )

        # make predictions for the validation data set
        val_groundtruths, val_predictions, val_scores = model_utils.predict(
            model, val_dl, self.device
        )

        return val_groundtruths, val_predictions, val_scores

    def _train_eval_xgb(
        self,
        train: CIDataset,
        val: CIDataset,
        class_weights: np.ndarray,
        random_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trains an XGBClassifier on the provided training data and evaluates it on the validation data.

        Args:
            train (CIDataset): the training data
            val (CIDataset): the validation data
            class_weights (np.ndarray): how to reweight the classes during training
            random_seed (int): a random seed for the XGBClassifier

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: the groundtruths, predictions and prediction scores of the model on the validation set, respectively
        """

        # extract only the feature values and the labels
        train_X = train.get_features()
        train_y = train.get_labels()

        # assign sample weights to the training examples
        sample_weights = []
        for label in train_y:
            sample_weights.append(class_weights[label])

        # construct a training dmatrix
        train_dmatrix = xgb.DMatrix(train_X, label=train_y, weight=sample_weights)

        # extract only the feature values and the labels
        val_X = val.get_features()
        val_y = val.get_labels()

        # construct a validation dmatrix
        val_dmatrix = xgb.DMatrix(val_X, label=val_y)

        # construct a booster (for compatibility with FL)
        hyperparameters = {
            "eta": self.config.learning_rate,  # Learning rate
            "max_depth": self.config.max_depth,
            "num_parallel_tree": 1,
            "subsample": self.config.subsample_ratio,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_lambda": self.config.reg_lambda,
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": self.config.tree_method,
            "random_state": random_seed,
        }

        model = xgb.train(
            hyperparameters,
            train_dmatrix,
            num_boost_round=self.config.n_estimators,
            evals=[(train_dmatrix, "train")],
        )

        # predict the validation set
        val_groundtruths, val_predictions, val_scores = (
            val_y.copy(),
            np.argmax(model.predict(val_dmatrix), axis=1),
            np.array(model.predict(val_dmatrix)),
        )

        return val_groundtruths, val_predictions, val_scores

    def _centralized_learning(
        self, fold_idx: int, client_splits: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs centralized learning using the specified fold as the held out validation set and the rest of the client's folds as the training set.

        Args:
            fold_idx (int): index of the fold to be used for validation at each client
            client_splits (Dict): the fold definitions for each client

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: the groundtruths, predictions and prediction scores of the model on the validation set, as well as the client ids of the examples in the validation set, respectively
        """

        # initialize a place where everyone's data will be pooled
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        val_client_ids = pd.Series()

        for client in client_splits.keys():
            # separate the client's training data
            client_data = self.processed_train_data[
                self.processed_train_data["dataset"] == client
            ]
            client_train_indices, client_val_indices = client_splits[client][fold_idx]

            # split the client's training data
            client_train_data = client_data.iloc[client_train_indices]
            client_val_data = client_data.iloc[client_val_indices]

            # add the subsets to the pooling variables
            train_df = pd.concat([train_df, client_train_data], axis=0)
            val_df = pd.concat([val_df, client_val_data], axis=0)
            val_client_ids = pd.concat(
                [val_client_ids, pd.Series([client] * client_val_data.shape[0])]
            )

        # get class weights based on the training set only
        train_class_weights = utils.calc_class_weights(
            num_unique_labels=self.config.num_unique_labels,
            labels=train_df["label"],
            class_weight_constant=self.config.class_weight_constant,
            class_weight_exponent=self.config.class_weight_exponent,
        )

        train = CIDataset(data=train_df, feature_columns=self.feature_columns)
        val = CIDataset(data=val_df, feature_columns=self.feature_columns)

        if self.config.model == "mlp":
            val_groundtruths, val_predictions, val_scores = self._train_eval_mlp(
                train=train, val=val, class_weights=train_class_weights
            )

        elif self.config.model == "xgb":
            val_groundtruths, val_predictions, val_scores = self._train_eval_xgb(
                train=train,
                val=val,
                class_weights=train_class_weights,
                random_seed=self.random_seed,
            )
        else:
            print("Model type not supported, exiting pipeline...")
            exit(-1)

        return val_groundtruths, val_predictions, val_scores, val_client_ids

    def _local_learning(
        self, client: str, fold_idx: int, client_splits: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs local learning for the specificed client using the specified fold as the held out validation set and the rest of the client's folds as the training set.

        Args:
            client (str): the id of the client for whom local learning should be performed
            fold_idx (int): index of the fold to be used for validation
            client_splits (Dict): the fold definitions for each client

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: the groundtruths, predictions and prediction scores of the model on the validation set, respectively
        """

        # separate the client's data
        client_data = self.processed_train_data[
            self.processed_train_data["dataset"] == client
        ].copy()
        client_train_indices, client_val_indices = client_splits[client][fold_idx]

        # split the client's data
        client_train_df = client_data.iloc[client_train_indices]
        client_val_df = client_data.iloc[client_val_indices]

        # get weights
        class_weights = utils.calc_class_weights(
            num_unique_labels=self.config.num_unique_labels,
            labels=client_train_df["label"],
            class_weight_constant=self.config.class_weight_constant,
            class_weight_exponent=self.config.class_weight_exponent,
        )

        train = CIDataset(data=client_train_df, feature_columns=self.feature_columns)
        val = CIDataset(data=client_val_df, feature_columns=self.feature_columns)

        if self.config.model == "mlp":
            (
                val_groundtruths,
                val_predictions,
                val_scores,
            ) = self._train_eval_mlp(train=train, val=val, class_weights=class_weights)
        elif self.config.model == "xgb":
            (
                val_groundtruths,
                val_predictions,
                val_scores,
            ) = self._train_eval_xgb(
                train=train,
                val=val,
                class_weights=class_weights,
                random_seed=self.random_seed,
            )
        else:
            print("Model type not supported, exiting pipeline...")
            exit(-1)

        return val_groundtruths, val_predictions, val_scores

    def _test_eval_centralized(self):
        train_df = self.processed_train_data
        test_df = self.processed_test_data
        test_client_ids = test_df["dataset"]

        # get class weights based on the training set only
        train_class_weights = utils.calc_class_weights(
            num_unique_labels=self.config.num_unique_labels,
            labels=train_df["label"],
            class_weight_constant=self.config.class_weight_constant,
            class_weight_exponent=self.config.class_weight_exponent,
        )

        train = CIDataset(data=train_df, feature_columns=self.feature_columns)
        test = CIDataset(data=test_df, feature_columns=self.feature_columns)

        if self.config.model == "mlp":
            test_groundtruths, test_predictions, test_scores = self._train_eval_mlp(
                train=train, val=test, class_weights=train_class_weights
            )

        elif self.config.model == "xgb":
            test_groundtruths, test_predictions, test_scores = self._train_eval_xgb(
                train=train,
                val=test,
                class_weights=train_class_weights,
                random_seed=self.random_seed,
            )
        return test_groundtruths, test_predictions, test_scores, test_client_ids

    def _test_eval_local(self, client: str):
        train_df = self.processed_train_data[
            self.processed_train_data["dataset"] == client
        ]
        test_df = self.processed_test_data[
            self.processed_test_data["dataset"] == client
        ]
        test_client_ids = test_df["dataset"]

        # get class weights based on the training set only
        train_class_weights = utils.calc_class_weights(
            num_unique_labels=self.config.num_unique_labels,
            labels=train_df["label"],
            class_weight_constant=self.config.class_weight_constant,
            class_weight_exponent=self.config.class_weight_exponent,
        )

        train = CIDataset(data=train_df, feature_columns=self.feature_columns)
        test = CIDataset(data=test_df, feature_columns=self.feature_columns)

        if self.config.model == "mlp":
            test_groundtruths, test_predictions, test_scores = self._train_eval_mlp(
                train=train, val=test, class_weights=train_class_weights
            )

        elif self.config.model == "xgb":
            test_groundtruths, test_predictions, test_scores = self._train_eval_xgb(
                train=train,
                val=test,
                class_weights=train_class_weights,
                random_seed=self.random_seed,
            )

        return test_groundtruths, test_predictions, test_scores, test_client_ids

    def _train(self, eval_on_test: bool):
        for run in range(self.config.num_runs):
            print(f"Starting run {run+1} of {self.config.num_runs}")

            # initialize the run
            client_splits, run_logger = self._prepare_run(run_id=run)

            # iterate through the folds
            for fold_idx in range(self.config.num_cv_folds):
                print(
                    f"*** Working on fold {fold_idx + 1} of {self.config.num_cv_folds}"
                )

                ##############################################
                # centralized learning option
                ##############################################
                if self.config.method == "centralized_learning":
                    (
                        val_groundtruths,
                        val_predictions,
                        val_scores,
                        val_client_ids,
                    ) = self._centralized_learning(fold_idx, client_splits)

                    # separate valdation results per client and log them in the RunLogger
                    for client in client_splits.keys():
                        client_val_groundtruths = val_groundtruths[
                            val_client_ids == client
                        ]
                        client_val_predictions = val_predictions[
                            val_client_ids == client
                        ]
                        client_val_scores = val_scores[val_client_ids == client]
                        run_logger.add_client_fold_results(
                            client_id=client,
                            fold_id=fold_idx,
                            groundtruths=client_val_groundtruths,
                            predictions=client_val_predictions,
                            prediction_scores=client_val_scores,
                        )
                ##############################################
                # local learning option
                ##############################################
                elif self.config.method == "local_learning":
                    for client in client_splits.keys():
                        print(f"****** Working on client {client}")
                        (
                            client_val_groundtruths,
                            client_val_predictions,
                            client_val_scores,
                        ) = self._local_learning(client, fold_idx, client_splits)

                        run_logger.add_client_fold_results(
                            client_id=client,
                            fold_id=fold_idx,
                            groundtruths=client_val_groundtruths,
                            predictions=client_val_predictions,
                            prediction_scores=client_val_scores,
                        )
                ##############################################
                else:
                    print("[ERROR] Method not supported...")
                    exit(-1)

            if eval_on_test == True:
                if self.config.method == "centralized_learning":
                    (
                        test_groundtruths,
                        test_predictions,
                        test_scores,
                        test_client_ids,
                    ) = self._test_eval_centralized()

                    # separate valdation results per client and log them in the RunLogger
                    for client in test_client_ids.unique():
                        client_test_groundtruths = test_groundtruths[
                            test_client_ids == client
                        ]
                        client_test_predictions = test_predictions[
                            test_client_ids == client
                        ]
                        client_test_scores = test_scores[test_client_ids == client]

                        run_logger.add_client_test_results(
                            client_id=client,
                            groundtruths=client_test_groundtruths,
                            predictions=client_test_predictions,
                            prediction_scores=client_test_scores,
                        )
                else:
                    for client in client_splits.keys():
                        (
                            test_groundtruths,
                            test_predictions,
                            test_scores,
                            test_client_ids,
                        ) = self._test_eval_local(client)

                        run_logger.add_client_test_results(
                            client_id=client,
                            groundtruths=test_groundtruths,
                            predictions=test_predictions,
                            prediction_scores=test_scores,
                        )

            self.batch_logger.add_run(run_logger)
        #############################################


class FederatedPipeline(Pipeline):
    """A federated implementation of the abstract class Pipeline.

    This implementation includes a:
        - federated learning pipeline
        - personalized federated learning pipeline
    """

    def _calculate_client_resources(self) -> Dict:
        """Calculates the CPU and GPU resources that can be assiged to each individual client.

        Returns:
            Dict: the resource specification
        """

        gpus_per_client = torch.cuda.device_count() / len(list(self.dataset2id.keys()))
        if "SLURM_CPUS_PER_TASK" in os.environ:
            cpus_per_client = (int(os.environ.get("SLURM_CPUS_PER_TASK")) - 1) / len(
                list(self.dataset2id.keys())
            )
        else:
            cpus_per_client = 3.0 / len(list(self.dataset2id.keys()))

        client_resources = {
            "num_gpus": math.floor(gpus_per_client) if gpus_per_client > 1 else 0,
            "num_cpus": (
                math.floor(cpus_per_client) if cpus_per_client > 1 else cpus_per_client
            ),
        }

        return client_resources

    def _construct_client_datasets(
        self,
        fold_idx: int,
        client_splits: Dict,
        feature_columns: List[str],
    ) -> Tuple[Dict[str, CIDataset], pd.DataFrame, pd.Series]:
        """Constructs CIDatasets for the training and validation data of each client by using the specified fold as the validation partition.

        Args:
            fold_idx (int): index of the fold to be used for validation at each client
            client_splits (Dict): the fold definitions for each client
            feature_columns (List[str]): a list of feature names

        Returns:
            Tuple[Dict[str, CIDataset], pd.DataFrame, pd.Series]: dictionary containing the training CIDatasets of each client, a pooled validation dataframe and the client ids of the instances in the pooled val dataset, respectively
        """

        client_train_datasets = {}
        val_df = pd.DataFrame()
        val_client_ids = pd.Series()

        for client in client_splits.keys():
            # separate the client's data
            client_data = self.processed_train_data[
                self.processed_train_data["dataset"] == client
            ]
            client_train_indices, client_val_indices = client_splits[client][fold_idx]

            # split the client's data
            client_train_data = client_data.iloc[client_train_indices]
            client_val_data = client_data.iloc[client_val_indices]

            client_train_datasets[str(client)] = CIDataset(
                client_train_data, feature_columns
            )
            val_df = pd.concat([val_df, client_val_data], axis=0)
            val_client_ids = pd.concat(
                [val_client_ids, pd.Series([client] * client_val_data.shape[0])]
            )

        return client_train_datasets, val_df, val_client_ids

    def _construct_test_eval_datasets(
        self,
        client_ids: List[str],
        feature_columns: List[str],
    ) -> Tuple[Dict[str, CIDataset], pd.DataFrame, pd.Series]:
        """Constructs CIDatasets for the training and validation data of each client by using the specified fold as the validation partition.

        Args:
            feature_columns (List[str]): a list of feature names

        Returns:
            Tuple[Dict[str, CIDataset], pd.DataFrame, pd.Series]: dictionary containing the training CIDatasets of each client, a pooled validation dataframe and the client ids of the instances in the pooled val dataset, respectively
        """

        client_train_datasets = {}
        test_df = pd.DataFrame()
        test_client_ids = pd.Series()

        for client in client_ids:
            # separate the client's data
            client_train_data = self.processed_train_data[
                self.processed_train_data["dataset"] == client
            ]
            client_test_data = self.processed_test_data[
                self.processed_test_data["dataset"] == client
            ]

            client_train_datasets[str(client)] = CIDataset(
                client_train_data, feature_columns
            )
            test_df = pd.concat([test_df, client_test_data], axis=0)
            test_client_ids = pd.concat(
                [test_client_ids, pd.Series([client] * client_test_data.shape[0])]
            )

        return client_train_datasets, test_df, test_client_ids

    def _config_fn_with_arguments_(self, config: SimpleNamespace) -> Callable:
        """A wrapper function that returns a function suitable for defining the config dictionary in a FLWR strategy

        Args:
            config (SimpleNamespace): contains the parameters for the pipeline's building blocks

        Returns:
            Callable: a config defining function with just the server round as an input parameter
        """

        def config_fn(server_round: int):
            config.random_seed = self.random_seed
            return config

        return config_fn

    def _server_evaluate(
        self, server_round: int, parameters: fl.common.NDArrays, config: SimpleNamespace
    ):
        """A function to be executed at the end of each communication round. Traditionally, evaluation of the server side model is performed here,
        but in our case, it will only be used to save the model parameters into a global variable.

        Args:
            server_round (int): the communication round
            parameters (fl.common.NDArrays): parameters of the aggregated model #! This needs to be updated
            config (Dict[str, fl.common.Scalar]): a configuration dictionary
        """
        global persistent_storage
        persistent_storage["last_model_params"] = parameters

    def _run_simulation(
        self,
        strategy: fl.server.strategy.Strategy,
        client_fn: Callable,
        client_datasets: Dict,
    ) -> fl.server.history.History:
        """Runs a FL simulation.

        Args:
            strategy (fl.server.strategy.Strategy): the strategy to use for model aggregation
            client_fn (Callable): a function for creating client objects
            client_datasets (Dict): a dictionary containing CIDatasets of the training data for each client

        Returns:
            fl.server.history.History: a history object possibly containing metrics over communication rounds
        """

        client_resources = self._calculate_client_resources()

        simulation_history = fl.simulation.start_simulation(
            client_fn=client_fn,
            config=fl.server.ServerConfig(num_rounds=self.config.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
            clients_ids=[str(i) for i in range(len(list(client_datasets.keys())))],
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
            },
            keep_initialised=True,
        )

        return simulation_history

    def _fl_mlp(
        self,
        client_train_datasets: Dict,
        val_df: pd.DataFrame,
        val_client_ids: pd.Series,
        client_fn: Callable,
        persistent_storage: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs a FL simulation using MLP as the model.

        Args:
            client_train_datasets (Dict): a dictionary of CIDatasets for each client
            val_df (pd.DataFrame): the pooled validation data
            val_client_ids (pd.Series): the client ids for the instances in the pooled validationd ata
            client_fn (Callable): a client creating function
            persistent_storage (Dict): a dictionary used for persistent model storage across FL communication rounds

        Returns:
            Tuple [np.ndarray, np.ndarray, np.ndarray]: the groundtruths, predictions and prediction scores of the model on the validation set
        """

        initial_model = model_utils.get_model(
            self.config.input_size,
            self.config.num_neurons_per_layer,
            self.config.num_unique_labels,
            self.config.dropout_prob,
        )

        initial_params = model_utils.get_model_parameters(initial_model)

        # define learning strategy: MLP
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=self.config.C,
            fraction_evaluate=self.config.C,
            on_fit_config_fn=self._config_fn_with_arguments_(self.config),
            on_evaluate_config_fn=self._config_fn_with_arguments_(self.config),
            evaluate_fn=self._server_evaluate,
            initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
            accept_failures=False,
        )
        # start simulation
        _ = self._run_simulation(strategy, client_fn, client_train_datasets)

        # create a CIDataset for the validation data
        val = CIDataset(data=val_df, feature_columns=self.feature_columns)

        if self.config.personalized == "y":
            (val_groundtruths, val_predictions, val_scores) = self._personalised_fl_mlp(
                client_train_datasets,
                val,
                val_client_ids,
                client_fn,
                persistent_storage,
            )
        else:
            eval_model = model_utils.get_model(
                self.config.input_size,
                self.config.num_neurons_per_layer,
                self.config.num_unique_labels,
                self.config.dropout_prob,
            )

            # set the model parameters to the ones from the last round of training
            eval_model = model_utils.set_model_parameters(
                eval_model, persistent_storage["last_model_params"]
            )

            # keeping shuffle False in the validation set to add the predictions and label later
            val_dl = DataLoader(val, batch_size=self.config.batch_size, shuffle=False)

            # make predictions for the validation data set
            val_groundtruths, val_predictions, val_scores = model_utils.predict(
                eval_model, val_dl, self.device
            )

        return val_groundtruths, val_predictions, val_scores

    def _fl_xgb(
        self,
        client_train_datasets: Dict,
        val_df: pd.DataFrame,
        val_client_ids: pd.Series,
        client_fn: Callable,
        persistent_storage: Dict,
    ) -> Tuple:
        """Runs a FL simulation using XGB as the model.

        Args:
            client_train_datasets (Dict): a dictionary of CIDatasets for each client
            val_df (pd.DataFrame): the pooled validation data
            val_client_ids (pd.Series): the client ids for the instances in the pooled validationd ata
            client_fn (Callable): a client creating function
            persistent_storage (Dict): a dictionary used for persistent model storage across FL communication rounds

        Returns:
            Tuple [np.ndarray, np.ndarray, np.ndarray]: the groundtruths, predictions and prediction scores of the model on the validation set
        """
        # define learning strategy: XBG
        strategy = fl.server.strategy.FedXgbBagging(
            fraction_fit=self.config.C,
            fraction_evaluate=self.config.C,
            on_fit_config_fn=self._config_fn_with_arguments_(self.config),
            on_evaluate_config_fn=self._config_fn_with_arguments_(self.config),
            evaluate_function=self._server_evaluate,
            accept_failures=False,
        )

        # start simulation
        _ = self._run_simulation(strategy, client_fn, client_train_datasets)

        # restructure the validation dataframe and make predictions based on the personalized client model
        val = CIDataset(data=val_df, feature_columns=self.feature_columns)

        if self.config.personalized == "y":
            (
                val_groundtruths,
                val_predictions,
                val_scores,
            ) = self._personalised_fl_xgb(
                client_train_datasets,
                val,
                val_client_ids,
                client_fn,
                persistent_storage,
            )
        else:
            val_X = val.get_features()
            val_y = val.get_labels()

            # create an evaluation model and set its "weights" to the last saved during fl simulation
            eval_model = xgb.XGBClassifier()
            global_model = None
            for item in persistent_storage["last_model_params"].tensors:
                global_model = bytearray(item)

            eval_model.load_model(global_model)
            # restructure the validation dataframe and make predictions
            val = CIDataset(data=val_df, feature_columns=self.feature_columns)
            val_X = val.get_features()
            val_y = val.get_labels()
            val_groundtruths, val_predictions, val_scores = (
                val_y.copy(),
                np.array(eval_model.predict(val_X)),
                np.array(eval_model.predict_proba(val_X)),
            )

        return val_groundtruths, val_predictions, val_scores

    def _personalised_fl_mlp(
        self,
        client_train_datasets: Dict,
        val: CIDataset,
        val_client_ids: pd.Series,
        client_fn: Callable,
        persistent_storage: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Personalizes a trained MLP FL model on each client separatelty.
        Args:
            client_train_datasets (Dict): a dictionary of CIDatasets for each client
            val (CIDataset): the pooled validation data
            val_client_ids (pd.Series): the client ids for the instances in the pooled validationd ata
            client_fn (Callable): a client creating function
            persistent_storage (Dict): a dictionary used for persistent model storage across FL communication rounds

        Returns:
            Tuple [np.ndarray, np.ndarray, np.ndarray]: the groundtruths, predictions, and prediction scores of the model on the validation set
        """

        # initialize result pooling variables
        val_groundtruths, val_predictions, val_scores = (
            np.zeros(val.features.shape[0]),
            np.zeros(val.features.shape[0]),
            np.zeros((val.features.shape[0], 3)),
        )

        # iterate over the clients and train the global model for a few epochs on their local data
        for cid in [str(i) for i in range(len(list(client_train_datasets.keys())))]:
            # create a client object
            client = client_fn(cid)

            # create a model
            client_model = model_utils.get_model(
                self.config.input_size,
                self.config.num_neurons_per_layer,
                self.config.num_unique_labels,
                self.config.dropout_prob,
            )

            # set the model parameters to the ones from the last round of training
            client_model = model_utils.set_model_parameters(
                client_model, persistent_storage["last_model_params"]
            )

            # create a dataloader for the client's data set
            client_dataloader = DataLoader(
                client.dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
            )

            # train the personalized model for a few epochs
            # * because of the way this function is set up, we need to briefly overwrite some of the config parameters
            num_epochs = self.config.num_epochs
            learning_rate = self.config.learning_rate

            self.config.num_epochs = self.config.per_num_epochs
            self.config.learning_rate = self.config.per_lr

            _ = model_utils.train_model(
                client_model,
                client_dataloader,
                client.class_weights,
                self.config,
                self.device,
            )

            self.config.num_epochs = num_epochs
            self.config.learning_rate = learning_rate

            # keeping shuffle False in the validation set to add the predictions and label later
            val_dl = DataLoader(val, batch_size=self.config.batch_size, shuffle=False)

            # make predictions for the validation data set
            (
                client_val_groundtruths,
                client_val_predictions,
                client_val_scores,
            ) = model_utils.predict(client_model, val_dl, self.device)

            # separate the predictions and groundtruths that belong to the specific client
            client_val_groundtruths = client_val_groundtruths[
                val_client_ids.values.astype(np.int) == int(cid)
            ]
            client_val_predictions = client_val_predictions[
                val_client_ids.values.astype(np.int) == int(cid)
            ]
            client_val_scores = client_val_scores[
                val_client_ids.values.astype(np.int) == int(cid), :
            ]

            # save them in the pooling variables
            val_groundtruths[val_client_ids.values.astype(np.int) == int(cid)] = (
                client_val_groundtruths
            )
            val_predictions[val_client_ids.values.astype(np.int) == int(cid)] = (
                client_val_predictions
            )
            val_scores[
                val_client_ids.values.astype(np.int) == int(cid),
                : client_val_scores.shape[1],
            ] = client_val_scores

        return val_groundtruths, val_predictions, val_scores

    def _personalised_fl_xgb(
        self,
        client_train_datasets: Dict,
        val: CIDataset,
        val_client_ids: pd.Series,
        client_fn: Callable,
        persistent_storage: Dict,
    ):
        """Personalizes a trained XGB FL model on each client separatelty.
        Args:
            client_train_datasets (Dict): a dictionary of CIDatasets for each client
            val (CIDataset): the pooled validation data
            val_client_ids (pd.Series): the client ids for the instances in the pooled validationd ata
            client_fn (Callable): a client creating function
            persistent_storage (Dict): a dictionary used for persistent model storage across FL communication rounds

        Returns:
            Tuple [np.ndarray, np.ndarray, np.ndarray]: the groundtruths, predictions, and prediction scores of the model on the validation set
        """

        val_X = val.get_features()
        val_y = val.get_labels()

        # initialize result pooling variables
        val_groundtruths, val_predictions, val_scores = (
            np.zeros(val_X.shape[0]),
            np.zeros(val_X.shape[0]),
            np.zeros((val_X.shape[0], 3)),
        )

        # iterate over the clients and extend the global model with a few estimators developed on their local data
        for cid in [str(i) for i in range(len(list(client_train_datasets.keys())))]:
            # create a client
            client = client_fn(cid)

            # becuase of the way things are set up, we need to briefly change the value of the learning rate config parameter
            learning_rate = self.config.learning_rate
            self.config.learning_rate = self.config.per_lr

            # create a model for this newly initialized client
            client.__build_model__(self.config)
            # set the client's model to that saved at the last round of FL training
            client.__set_model_params__(persistent_storage["last_model_params"])

            # update the global model based on the client's data
            client.__local_boost__(self.config.per_n_estimators)
            # get the parameters of the personalized model
            personalized_client_parameters = client.get_parameters().parameters.tensors

            # reset the learning rate value
            self.config.learning_rate = learning_rate

            # create an evaluation model and set its "weights" to the ones from the personalized model
            client_eval_model = xgb.XGBClassifier()
            global_model = None
            for item in personalized_client_parameters:
                global_model = bytearray(item)
            client_eval_model.load_model(global_model)

            # restructure the validation dataframe and make predictions based on the personalized client model
            (
                client_val_groundtruths,
                client_val_predictions,
                client_val_scores,
            ) = (
                val_y.copy(),
                np.array(client_eval_model.predict(val_X)),
                np.array(client_eval_model.predict_proba(val_X)),
            )

            # separate the predictions and groundtruths that belong to the specific client
            client_val_groundtruths = client_val_groundtruths[
                val_client_ids.values.astype(np.int) == int(cid)
            ]
            client_val_predictions = client_val_predictions[
                val_client_ids.values.astype(np.int) == int(cid)
            ]
            client_val_scores = client_val_scores[
                val_client_ids.values.astype(np.int) == int(cid), :
            ]

            # save them in the pooling variables
            val_groundtruths[val_client_ids.values.astype(np.int) == int(cid)] = (
                client_val_groundtruths
            )
            val_predictions[val_client_ids.values.astype(np.int) == int(cid)] = (
                client_val_predictions
            )
            val_scores[
                val_client_ids.values.astype(np.int) == int(cid),
                : client_val_scores.shape[1],
            ] = client_val_scores

        return val_groundtruths, val_predictions, val_scores

    def _train(self, eval_on_test):
        """Train the Federated Learning Pipeline"""

        global persistent_storage
        persistent_storage = {}

        for run in range(self.config.num_runs):
            print(f"Starting run {run+1} of {self.config.num_runs}")

            # initialize the run
            client_splits, run_logger = self._prepare_run(run_id=run)

            # iterate through the folds
            for fold_idx in range(self.config.num_cv_folds):
                print(
                    f"*** Working on fold {fold_idx + 1} of {self.config.num_cv_folds}"
                )

                # construct train and validation client datasets
                client_train_datasets, val_df, val_client_ids = (
                    self._construct_client_datasets(
                        fold_idx, client_splits, self.feature_columns
                    )
                )

                # specify the client creating function
                client_fn = create_client_with_args(
                    client_train_datasets,
                    self.config.class_weight_exponent,
                    self.config.class_weight_constant,
                    self.config.num_unique_labels,
                    self.config.model,
                    self.device,
                )

                ##############################################
                # XGB option
                ##############################################
                if self.config.model == "xgb":
                    (
                        val_groundtruths,
                        val_predictions,
                        val_scores,
                    ) = self._fl_xgb(
                        client_train_datasets,
                        val_df,
                        val_client_ids,
                        client_fn,
                        persistent_storage,
                    )
                ##############################################
                # MLP option
                ##############################################
                elif self.config.model == "mlp":
                    (val_groundtruths, val_predictions, val_scores) = self._fl_mlp(
                        client_train_datasets,
                        val_df,
                        val_client_ids,
                        client_fn,
                        persistent_storage,
                    )

                # separate valdation results per client and log them in the RunLogger
                for client in client_splits.keys():
                    client_val_groundtruths = val_groundtruths[val_client_ids == client]
                    client_val_predictions = val_predictions[val_client_ids == client]
                    client_val_scores = val_scores[val_client_ids == client]
                    run_logger.add_client_fold_results(
                        client_id=client,
                        fold_id=fold_idx,
                        groundtruths=client_val_groundtruths,
                        predictions=client_val_predictions,
                        prediction_scores=client_val_scores,
                    )
                ##############################################

            if eval_on_test == True:
                client_train_datasets, test_df, test_client_ids = (
                    self._construct_test_eval_datasets(
                        list(client_splits.keys()), self.feature_columns
                    )
                )
                # specify the client creating function
                client_fn = create_client_with_args(
                    client_train_datasets,
                    self.config.class_weight_exponent,
                    self.config.class_weight_constant,
                    self.config.num_unique_labels,
                    self.config.model,
                    self.device,
                )

                ##############################################
                # XGB option
                ##############################################
                if self.config.model == "xgb":
                    (
                        test_groundtruths,
                        test_predictions,
                        test_scores,
                    ) = self._fl_xgb(
                        client_train_datasets,
                        test_df,
                        test_client_ids,
                        client_fn,
                        persistent_storage,
                    )
                ##############################################
                # MLP option
                ##############################################
                elif self.config.model == "mlp":
                    (test_groundtruths, test_predictions, test_scores) = self._fl_mlp(
                        client_train_datasets,
                        test_df,
                        test_client_ids,
                        client_fn,
                        persistent_storage,
                    )

                # separate valdation results per client and log them in the RunLogger
                for client in client_splits.keys():
                    client_test_groundtruths = test_groundtruths[
                        test_client_ids == client
                    ]
                    client_test_predictions = test_predictions[
                        test_client_ids == client
                    ]
                    client_test_scores = test_scores[test_client_ids == client]
                    run_logger.add_client_test_results(
                        client_id=client,
                        groundtruths=client_test_groundtruths,
                        predictions=client_test_predictions,
                        prediction_scores=client_test_scores,
                    )

            self.batch_logger.add_run(run_logger)
        return
