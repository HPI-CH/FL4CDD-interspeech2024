"""
Module Name: evaluation_utils
Description: This module contains all the evaluation functions.

Functions:
- calculate_crossentropy_loss: calculates cross-entropy loss
- generate_speaker_level_predictions: generate speaker-level predictions
- calculate_metrics: plot confusion matrix and calcualte the balanced accuracy, the macro f-score and the full classification report
"""

from typing import List, Tuple

import numpy as np
from scipy import stats

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)
import torch
import torch.nn as nn

from confidence_intervals import evaluate_with_conf_int


def calculate_crossentropy_loss(
    groundtruth: np.ndarray, pred_scores: np.ndarray, class_weights: np.ndarray
) -> float:
    """Calculates cross-entropy loss based on the provided groundtruth labels, prediction scores and class weights.

    Args:
        groundtruth (np.ndarray): the groundtruth labels (with shape (-1, len(class_weights)))
        pred_scores (np.ndarray): prediction scores (with shape (-1, len(class_weights)))
        class_weights (np.ndarray): the weight that should be assigned to each class

    Returns:
        float: the weighted cross-entropy loss
    """

    groundtruth = torch.Tensor(groundtruth).type(torch.LongTensor)
    pred_scores = torch.Tensor(pred_scores).type(torch.FloatTensor)
    class_weights = torch.Tensor(class_weights).type(torch.FloatTensor)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    loss = criterion(pred_scores, groundtruth)

    return loss.item()


def flatten_classification_report(report: dict) -> dict:
    """Flattens a classification report dictionary into a one-level dictionary.

    Args:
        report (dict): the classification report dictionary

    Returns:
        dict: the flattened classification report dictionary
    """
    flat_report = {}

    for class_name, class_metrics in report.items():
        if isinstance(class_metrics, dict):
            for metric_name, metric_value in class_metrics.items():
                # allow no spaces in metric names (makes it easier to process results later)
                # also, ignore micro averaged and weighted metrics, as well as accuracy
                if (
                    ("accuracy" not in class_name)
                    and ("micro" not in class_name)
                    and ("weighted" not in class_name)
                ):
                    flat_report[
                        f"{class_name.replace(' ', '_')}_{metric_name.replace(' ', '_')}"
                    ] = float(metric_value)

    return flat_report


def f1_macro_wrapper(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def calculate_metrics_with_cis(
    groundtruths: np.ndarray, predictions: np.ndarray, conditions: np.ndarray = None
):
    metrics = {"means": {}, "CIs": {}}

    (metrics["means"]["acc"], metrics["CIs"]["acc"]) = evaluate_with_conf_int(metric=accuracy_score, samples=predictions, labels=groundtruths, conditions = conditions, num_bootstraps=1000, alpha=5)
    (metrics["means"]["balanced_acc"], metrics["CIs"]["balanced_acc"]) = evaluate_with_conf_int(metric=balanced_accuracy_score, samples=predictions, labels=groundtruths, conditions = conditions, num_bootstraps=1000, alpha=5)
    (metrics["means"]["macro_f_score"], metrics["CIs"]["macro_f_score"]) = evaluate_with_conf_int(metric=f1_macro_wrapper, samples=predictions, labels=groundtruths, conditions = conditions, num_bootstraps=1000, alpha=5)

    return metrics


def calculate_metrics(
    groundtruths: np.ndarray,
    pred_scores: np.ndarray,
    id2label: dict,
) -> dict:
    """Plot confusion matrix and calculate the following metrics:
        - balanced accuracy
        - macro f-score
        - full classification report
        - sensitivity and specificity

    Args:
        groundtruths (np.ndarray): groundtruth labels
        pred_scores (np.ndarray): the scores predicted by a model for each example (shape: (-1, len(id2label)))
        id2label (dict): encoding to readable text mapping

    Returns:
        dct_results(dict): dictionary containing all metrics
    """
    label_ids = [k for k in id2label.keys()]
    label_names = [v for _, v in id2label.items()]

    y_true = groundtruths
    y_pred = np.argmax(pred_scores, axis=1)

    # assert length of ref and predictions is the same
    assert len(y_true) == len(y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0.0,
    )

    # calculate specificity and sensitivity
    ref, sen, spe = [], [], []
    for l in [0, 1, 2]:
        # * It seems like specifying the labels fixes the problem of this metric not reporting anything for labels that don't appear in the
        # * groundtruth?
        _, recall, _, _ = precision_recall_fscore_support(
            y_true == l,
            y_pred == l,
            labels=label_ids,
            pos_label=True,
            average=None,
            zero_division=0.0,
        )

        ref.append(l)

        sen.append(recall[1])
        spe.append(recall[0])

    # calculate basic accuracy
    acc = accuracy_score(y_true, y_pred)
    # calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # calculate f1 score
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)

    # obtain confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    # define dictionary with metrics
    dct_results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "f1_macro": float(macro_f1),
        "confusion_matrix": cm,
    }

    # add the sensitivity and specificity results
    for class_id, sens, spec in zip(ref, sen, spe):
        dct_results[f"sensitivity_{id2label[class_id]}"] = float(sens)
        dct_results[f"specificity_{id2label[class_id]}"] = float(spec)

    # add a deconstructed version of the classification report
    deconstructed_report = flatten_classification_report(report)
    dct_results.update(deconstructed_report)

    return dct_results
