"""
Module Name: tune
Description: This module contains driver code that runs either the CL or FL pipeline in a hyperparameter tuning setting with W&B
"""

import os
import shutil

import wandb

from utils import build_argument_parser

from pipeline import NonFederatedPipeline, FederatedPipeline

if __name__ == "__main__":

    parser = build_argument_parser()
    experiment_config = parser.parse_args()

    if not len(experiment_config.class_weight_constant):
        experiment_config.class_weight_constant = None

    tags = ["tunning", "setup_v6"]
    if experiment_config.eval_on_test == "y":
        experiment_config.eval_on_test = True
        tags = ["tunning", "setup_v6", "eval_on_test"]

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
        pipeline = NonFederatedPipeline(experiment_config, tuning=True)
        pipeline.execute(eval_on_test=experiment_config.eval_on_test)
    elif experiment_config.method == "federated_learning":
        pipeline = FederatedPipeline(experiment_config, tuning=True)
        pipeline.execute(eval_on_test=experiment_config.eval_on_test)

    wandb.finish()
