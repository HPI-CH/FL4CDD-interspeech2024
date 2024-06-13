"""
Module: best_parameters.py
Description: this module retrieves the best performing hyperparameters obtained in a sweep

Functions:
- retrieve_parameters: retrieves and saves the best performing hyperparameters from a sweep
"""

from argparse import ArgumentParser
import json
from tqdm import tqdm

import wandb

def retrieve_parameters():
    """Description: Retrieves the test set results from W&B.
    """
    wandb.login()
    api = wandb.Api()

    summarized_results = {}
    
    project = "tsp2024"
    entity = "hpi-mci-detection"

    runs = api.runs(f"{entity}/{project}")
    
    for run in tqdm(runs):
        tags_set = set(run.tags)
        if "eval_on_test" in tags_set and "setup_v6" in tags_set:
            feature_type = run.config.get("feature_type")
            learning_strat = run.config.get("method")
            model = run.config.get("model")
            personalization = None

            if learning_strat == 'federated_learning':
                personalization = run.config.get("personalized")
                if personalization == 'y':
                    learning_strat = 'personalized_federated_learning'

            if feature_type not in summarized_results:
                summarized_results[feature_type] = {}
            
            if learning_strat not in summarized_results[feature_type]:
                summarized_results[feature_type][learning_strat] = {}

            for dataset in ['pitt_adress', 'delaware', 'lu']:
                summarized_results[feature_type][learning_strat][dataset] = f"{run.summary[f'{dataset}_means_macro_f_score']:.3f} ({run.summary[f'{dataset}_CIs_macro_f_score_min']:.3f}-{run.summary[f'{dataset}_CIs_macro_f_score_max']:.3f})"        

    print(json.dumps(summarized_results, indent=4))

if __name__ == "__main__":
    retrieve_parameters()
