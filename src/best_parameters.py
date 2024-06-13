"""
Module: best_parameters.py
Description: this module retrieves the best performing hyperparameters obtained in a sweep

Functions:
- retrieve_parameters: retrieves and saves the best performing hyperparameters from a sweep
"""

from argparse import ArgumentParser
import json

import wandb

import sys


def retrieve_parameters(config):
    """Description:Retrieves the best performing hyperparameters from a sweep
    Args:
    config: config file to create the sweep name from
    """
    wandb.login()
    api = wandb.Api()

    feature = config["feature_type"]
    model = config["model"]
    method = config["method"]

    version = "setup_v6"
    if method == "federated_learning":
        personalisation = config["personalized"]
        sweep_name = f"{feature}_{model}_{method}_{personalisation}_{version}"
    else:
        sweep_name = f"{feature}_{model}_{method}_{version}"

    project = "tsp2024"
    entity = "hpi-mci-detection"

    runs = api.runs(f"{entity}/{project}")
    sweep_ids = None
    for run in runs:
        tags_set = set(run.tags)
        if "tunning" in tags_set and version in tags_set:
            if (
                run.config.get("feature_type") == feature
                and run.config.get("model") == model
                and run.config.get("method") == method
            ):
                if (
                    method == "federated_learning"
                    and run.config.get("personalized") == personalisation
                ):
                    sweep_ids = run.sweep
                    break
                elif method != "federated_learning":
                    sweep_ids = run.sweep
                    break

    if sweep_ids is not None:
        sweep = sweep_ids
        best_run = sweep.best_run()

        best_hyperparameters = best_run.config

        # save best hyperpameters: json file, name from sweep id
        json_file_name = f"../configs/best_hyperparameters_{sweep_name}.json"
        with open(json_file_name, "w") as json_file:
            json.dump(best_hyperparameters, json_file)
            print(f"Best hyperparameters saved to {json_file_name}")
    else:
        sys.exit("Sweep does not exist yet. Finetuning is needed.")
    return best_hyperparameters


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "config_file_path",
        type=str,
        help="path to the config file that defines the sweep name",
    )
    args = parser.parse_args()
    experiment_config = None
    with open(args.config_file_path, "r") as h:
        experiment_config = json.load(h)

    retrieve_parameters(experiment_config)
