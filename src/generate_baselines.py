import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from confidence_intervals import evaluate_with_conf_int

def f1_macro_wrapper(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def calculate_metrics_with_cis(
    groundtruths: np.ndarray, predictions: np.ndarray, conditions: np.ndarray = None
):
    metrics = {}

    (metrics["meanmacro_f_score"], metrics["CIs_macro_f_score"]) = evaluate_with_conf_int(metric=f1_macro_wrapper, samples=predictions, labels=groundtruths, conditions = conditions, num_bootstraps=1000, alpha=5)

    return metrics

def _generate_predictions(train_y, test_y, random_state):
    
    train_X, test_X = np.zeros((train_y.shape[0], 3)), np.zeros((test_y.shape[0], 3))
    
    prior_clf = DummyClassifier(strategy='prior', random_state=random_state)
    stratified_clf = DummyClassifier(strategy='stratified', random_state=random_state)
    uniform_clf = DummyClassifier(strategy='uniform', random_state=random_state)
    

    predictions_prior = prior_clf.fit(train_X, train_y).predict(test_X)
    predictions_stratified = stratified_clf.fit(train_X, train_y).predict(test_X)
    predictions_uniform = uniform_clf.fit(train_X, train_y).predict(test_X)

    predictions_const_control = ['control']*test_X.shape[0]
    predictions_const_mci = ['MCI']*test_X.shape[0]
    predictions_const_dementia = ['dementia']*test_X.shape[0]

    return predictions_prior, predictions_stratified, predictions_uniform, predictions_const_control, predictions_const_mci, predictions_const_dementia, test_y


def generate_baseline_results(args):
    metadata = pd.read_csv(args.path_to_metadata)[['dataset', 'diagnosis', 'split']]

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # iterate through the datasets/clients and generate baselines
    for dataset in metadata.dataset.unique():
        dataset_metadata = metadata[metadata.dataset==dataset]

        dataset_train, dataset_test = dataset_metadata[dataset_metadata.split=="train"], dataset_metadata[dataset_metadata.split=="test"]

        groundtruths = np.array([])
        priors = np.array([])
        strats = np.array([])
        uniforms = np.array([])
        controls = np.array([])
        mcis = np.array([])
        dems = np.array([])

        for random_state in range(10):
            preds_prior, preds_stratified, preds_uniform, preds_const_control, preds_const_mci, preds_const_dementia, gts = _generate_predictions(dataset_train.diagnosis, dataset_test.diagnosis, random_state)

            groundtruths = np.concatenate([groundtruths, gts])
            priors = np.concatenate([priors, preds_prior])
            strats = np.concatenate([strats, preds_stratified])
            uniforms = np.concatenate([uniforms, preds_uniform])
            controls = np.concatenate([controls, preds_const_control])
            mcis = np.concatenate([mcis, preds_const_mci])
            dems = np.concatenate([dems, preds_const_dementia])
        
        print (f"BASELINE RESULTS FOR {dataset}")
        print ("########################")
        print (f"######### PRIORS RESULTS")
        print (calculate_metrics_with_cis(groundtruths, priors))
        print (f"######### STRATS RESULTS")
        print (calculate_metrics_with_cis(groundtruths, strats))
        print (f"######### UNIFORMS RESULTS")
        print (calculate_metrics_with_cis(groundtruths, uniforms))
        print (f"######### CONTROLS RESULTS")
        print (calculate_metrics_with_cis(groundtruths, controls))
        print (f"######### MCIs RESULTS")
        print (calculate_metrics_with_cis(groundtruths, mcis))
        print (f"######### DEMs RESULTS")
        print (calculate_metrics_with_cis(groundtruths, dems))
        print ("########################")
        print ("########################")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "path_to_metadata",
        type=str,
        help="path to the metadata file that contains the train-test split",
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="path to the results directory where the baselines should be saved",
    )
    args = parser.parse_args()
    
    generate_baseline_results(args)