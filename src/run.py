"""
Module Name: run
Description: This module contains driver code/functions which tie together the functionality of all other modules.

Functions:
- run_experiment: Description
"""

import os
import shutil
import json
from argparse import ArgumentParser
from types import SimpleNamespace

import wandb

from pipeline import NonFederatedPipeline, FederatedPipeline

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "config_file_path",
        type=str,
        help="path to the config file that defines the experiment",
    )

    cli_params = parser.parse_args()

    experiment_config = None
    with open(cli_params.config_file_path, "r") as h:
        experiment_config = json.load(h)

    experiment_config = SimpleNamespace(**experiment_config)

    if experiment_config.class_weight_constant != None:
        if not len(experiment_config.class_weight_constant):
            experiment_config.class_weight_constant = None

    if experiment_config.shuffle == "y":
        experiment_config.shuffle = True
    else:
        experiment_config.shuffle = False
        
    if experiment_config.number_variance_pca == "None":
        experiment_config.number_variance_pca = None
    if experiment_config.number_components == "None":
        experiment_config.number_components == None

    tags = ["single_run", "setup_v6"]

    if experiment_config.eval_on_test == "y":
        experiment_config.eval_on_test = True
        tags = ["single_run", "setup_v6", "eval_on_test"]
    else:
        experiment_config.eval_on_test = False

    if experiment_config.method == "centralized_learning":
        tags.append("cl")
    elif experiment_config.method == "local_learning":
        tags.append("ll")
    elif experiment_config.method == "federated_learning":
        tags.append("fl")
        if experiment_config.personalized == "y":
            tags.append("personalized")
    else:
        print("Method unsupported")
        exit(-1)

    wandb.init(
        # set the wandb project where this run will be logged
        project="tsp2024",
        # set the team name
        entity="hpi-mci-detection",
        config=experiment_config,
        tags=tags,
    )

    if (
        experiment_config.method == "centralized_learning"
        or experiment_config.method == "local_learning"
    ):
        pipeline = NonFederatedPipeline(experiment_config, tuning=False)
        pipeline.execute(eval_on_test=experiment_config.eval_on_test)
    elif experiment_config.method == "federated_learning":
        pipeline = FederatedPipeline(experiment_config, tuning=False)
        pipeline.execute(eval_on_test=experiment_config.eval_on_test)

    wandb.finish()
