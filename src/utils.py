"""
!to-do: Figure out how to best deal with W&B hyperparameter tuning args

Module Name: utils

Description: This module houses all misc utility functions 

Functions:
- calc_class_weights: calculates the weight that should be assigned to examples of each class (more weight to under-represented classes)
- load_data: loads data from a pickle file 
- load_preprocessed_data: loads the preprocessed data from a directory
- save_preprocessed_data: saves the preprocessed data to a directory
- initialize_wandb: initializes the run on W&B
- log_to_wandb: Logs the final metrics to W&B
"""

import os
import json
from typing import List, Tuple
from argparse import ArgumentParser

import pickle
import numpy as np
import pandas as pd

from preprocessing_utils import FeatureExtractor


def flatten_dict(d: dict):
    flattened_dict = {}
    for key1, value1 in d.items():
        if isinstance(value1, dict):
            for key2, value2 in value1.items():
                if isinstance(value2, tuple):
                    flattened_dict[f"{key1}_{key2}_min"] = value2[0]
                    flattened_dict[f"{key1}_{key2}_max"] = value2[1]
                else:
                    flattened_dict[f"{key1}_{key2}"] = value2
    return flattened_dict


def flat_nested_dict(nested_dict):
    """flat dict for each clients results"""
    flattened_dict = {}

    for client, metrics in nested_dict.items():
        for metric_type, metric_values in metrics.items():
            for metric_name, value in metric_values.items():
                if isinstance(value, tuple):
                    min_key = f"{client}_CIs_{metric_name}_min"
                    max_key = f"{client}_CIs_{metric_name}_max"
                    flattened_dict[min_key] = value[0]
                    flattened_dict[max_key] = value[1]
                else:
                    key = f"{client}_{metric_type}_{metric_name}"
                    flattened_dict[key] = value

    return flattened_dict


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


def calc_class_weights(
    num_unique_labels: int,
    labels: np.ndarray,
    class_weight_constant: list,
    class_weight_exponent: float = None,
) -> np.ndarray:
    """Calculates the weight that should be assigned to examples of each class, such that more weights is put on examples from less represented classes.
    The logic for weight calculation is the following:

        1. the inverse of class counts is calculated, representing the initial weights for each class
        2. these initial weights are raised to the power of class_weight_exponent
        3. the resulting weights are normalized by dividing them by the sum of the weights to ensure they add up to 1

    Args:
        num_unique_labels (int): the number of unique labels for which we want a weight
            - in case there are labels in this list that are not encountered in the `labels` pd.Series, a weight of 0.0 is assigned to them
        labels (np.ndarray): the labels based on which we calculate the weights
        class_weight_exponent (float): a modifier for the degree to which weights are adjusted

    Returns:
        np.ndarray: an array of class weights, sorted by label value
    """

    (encountered_labels, encountered_label_counts) = np.unique(
        labels, return_counts=True
    )

    if class_weight_constant is not None:
        class_weights = np.array(class_weight_constant)
        labels_not_encountered = np.setdiff1d(
            np.arange(num_unique_labels), encountered_labels
        )
        class_weights[labels_not_encountered] = 0.0

        class_weights = class_weights / np.sum(class_weights)

    else:
        class_counts = [0] * num_unique_labels
        for encountered_label, label_count in zip(
            encountered_labels, encountered_label_counts
        ):
            class_counts[int(encountered_label)] = label_count

        # if a count of 0.0 is encountered, replace it with np.inf so that the weight in the end is 0.0
        class_counts = np.array([cc if cc != 0.0 else np.inf for cc in class_counts])

        class_weights = 1 / class_counts

        if class_weight_exponent is not None:
            class_weights = (class_weights**class_weight_exponent) / np.sum(
                class_weights**class_weight_exponent
            )
        else:
            class_weights = class_weights / np.sum(class_weights)

    return class_weights


def load_data(
    path_original_data: str,
    path_processed_data: str,
    feature_type: str,
    target: str,
    device: str,
    n_variance: float,
    number_components: int,
    pca: str,
    preprocess: bool = False,
) -> Tuple[pd.DataFrame, int, List[str], int]:
    """Loads the original or preprocessed data from a directory on disk. In case the preprocessing parameter is set, the loaded original data will be preprocessed before
    being returned. When this is not the case, the original pd.DataFrame will be returned, with an input size of zero, an empty list of features, and 0 as the number of unique labels.

    In case preprocess is set, `path_processed _data` will be treated as the path where extracted features should be saved
    Args:
        path_original_data (str): path to the directory where the original data is stored
        path_processed_data (str): path to the directory where the preprocessed data is stored
        feature_type (str): name of the feature set of interest
        target (str): name of the target class
        n_variance(float):number of variance to keep when doing pca
        pca(str): whether to apply pca or not
        preprocess(bool): whether to preprocess the dataset or not


    Returns:
        pd.DataFrame: processed dataframe
        int: number of features
        List[str]: list of feature columns
        int: number of unique of labels
    """
    feature_file_processed = f"{path_processed_data}/features_{feature_type}.pkl"
    columns_file_processed = f"{path_processed_data}/list_{feature_type}.pkl"

    if pca == "apply_pca":
        feature_file_processed = (
            f"{path_processed_data}/features_{feature_type}_pca_{n_variance}_{number_components}.pkl"
        )
        columns_file_processed = (
            f"{path_processed_data}/list_{feature_type}_pca_{n_variance}_{number_components}.pkl"
        )

    if os.path.isfile(feature_file_processed):
        print("Loading preprocessed data...", end="")
        with open(columns_file_processed, "rb") as file:
            feat_columns = pickle.load(file)
        df = load_processed_data(feature_file_processed, feat_columns)
    else:
        if path_original_data:
            print("Loading original data...", end="")

            db = pd.read_pickle(
                f"{path_original_data}/metadata_crossdataset_updated.pkl"
            )
            db.reset_index(inplace=True)

            db["file"] = db["file"].apply(lambda row: os.path.join(f"../data/", row))
            print("done")

            if preprocess:
                print("Starting feature extraction...", end="")
                feature_extractor = FeatureExtractor(device=device)
                df, feat_columns = feature_extractor.load_and_extract_features(
                    db,
                    feature_type,
                    target,
                    path_processed_data,
                    n_variance,
                    number_components,
                    pca,
                )
                ##################################################################
                #! temporary fix for dataloader not working with a DF returned by
                #! load_and_extract_features, but working when load_processed_data is called
                ##################################################################
            #    with open(
            #        columns_file_processed,
            #        "rb",
            #    ) as file:
            #        feat_columns = pickle.load(file)
            #    df = load_processed_data(feature_file_processed, feat_columns)
            #    print("done")
            #else:
            #    return db, 0, [], 0
        else:
            print(
                "If a valid path_processed_data was not provided, a valid path_original_data must be provided"
            )
            exit(-1)

    input_size = len(feat_columns)
    num_labels = len(df["label"].unique())

    return df, input_size, feat_columns, num_labels


def load_processed_data(file_path: str, feature_columns: List[str]) -> pd.DataFrame:
    """Loads the preprocessed data from disk.

    Args:
        file_path (str): absolute path to the file that contains the extracted features
        feature_columns (List[str]): names of the features of interest

    Returns:
        pd.DataFrame: the preprocessed data
    """

    processed_data = pd.read_pickle(file_path)

    # sanity check: asserting labels column has the same length as speaker columns
    assert (
        processed_data.loc[:, "ID"].shape[0] == processed_data.loc[:, "label"].shape[0]
    )

    return processed_data


def build_argument_parser() -> ArgumentParser:
    """Adds all possible CLI parameters to a provided argument parser. The idea is, that this argument parser will be used regardless of the
    pipeline and default values will be given to the FL-only parameters when using the CL pipeline, but they won't matter as they won't be used.

    #! This function allows interfacing with the W&B hyperparameter tuning solution (sweeps)

    Returns:
        ArgumentParser: a fully initialized CLI argument parser
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        default="centralized_learning",
        help="Whether to use CL or FL",
    )
    parser.add_argument(
        "--path_to_original",
        type=str,
        default="../metadata",
        help="Path to the original metadata file",
    )
    parser.add_argument(
        "--feature_output_dir",
        type=str,
        default="../features",
        help="Output directory for feature extraction",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default="diagnosis",
        help="The target class for a classification task",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="wav2vec_multilingual",
        help="Type of features to extract",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="Whether to use a MLP or an XGBoostClassifier",
    )
    parser.add_argument(
        "--device_id", type=int, default=0, help="ID of the GPU we want to use"
    )
    parser.add_argument(
        "--num_cv_folds", type=int, default=3, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Directory to save the results",
    )
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs")
    parser.add_argument(
        "--class_weight_exponent",
        type=float,
        default=None,
        help="Exponent for calculating class weights",
    )
    parser.add_argument(
        "--class_weight_constant",
        type=json.loads,
        default=[],
        help="assign constant weights to each class",
    )

    parser.add_argument(
        "--num_neurons_per_layer",
        type=json.loads,
        default=[512, 64],
        help="Number of neurons per layer (2 layered MLP)",
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--per_num_epochs",
        type=int,
        default=10,
        help="Number of personalisation epochs",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--per_lr",
        type=float,
        default=1e-4,
        help="Learning rate during personalization",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="ExponentialLR",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--scheduler_gamma", type=float, default=1.0, help="Scheduler gamma"
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument("--report_steps", type=int, default=1, help="Report steps")
    parser.add_argument("--shuffle", type=str, default="y", help="Shuffle data")
    parser.add_argument(
        "--eval_on_test",
        type=str,
        default="n",
        help="Whether to evaluate the models on the test set",
    )
    ## FL-specific parameters
    parser.add_argument(
        "--personalized",
        type=str,
        default="n",
        help="Whether to personalize the shared FL model",
    )
    parser.add_argument(
        "--number_variance_pca",
        type=float,
        default=None,
        help="variance to keep for pca",
    )

    parser.add_argument(
        "--number_components",
        type=int,
        default=None,
        help="number of components to keep for pca",
    )

    parser.add_argument(
        "--pca", type=str, default="no_pca", help="whether to apply pca or not"
    )
    parser.add_argument(
        "--unique_labels", type=json.loads, default=[0, 1, 2], help="Unique labels"
    )
    parser.add_argument("--C", type=float, default=0.1, help="C parameter")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of rounds")
    ## XGBoost-specific parameters
    parser.add_argument(
        "--n_estimators", type=int, default=10, help="Number of boosting rounds"
    )
    parser.add_argument(
        "--per_n_estimators",
        type=int,
        default=10,
        help="Number of personalisation boosting rounds",
    )
    parser.add_argument(
        "--max_depth", type=int, default=2, help="Max depth of each boosted tree"
    )
    parser.add_argument(
        "--subsample_ratio",
        type=float,
        default=0.1,
        help="Subsampling ratio for bagging",
    )
    parser.add_argument("--tree_method", type=str, default="hist", help="Tree method")
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=0.1,
        help="The percentage of columns to consider when splitting",
    )
    parser.add_argument(
        "--reg_lambda", type=float, default=1, help="L2 regularization parameter"
    )
    return parser
