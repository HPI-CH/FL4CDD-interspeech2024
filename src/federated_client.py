"""
Module Name: federated_client
Description: This module defines a client class.

Classes & Methods:
Client: Defines a Flower NumPy client that will be used for FL simulation
    - fit: trains a model on the client's data using the provided parameters as a starting point
    - evalute: evaluates a model, whose parameters are set to the ones provided, on the client's data
- create_client_with_args: a wrapper that returns a create_client function with embedded additional parameters (included for Flower compatibility)

#! to-do: update documentation
"""
from typing import Dict, List, Callable, Dict, Tuple
from types import SimpleNamespace

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    GetParametersIns,
    GetParametersRes,
)

import xgboost as xgb
import numpy as np

from torch.utils.data import DataLoader

import evaluation_utils
import model_utils
from dataset import CIDataset
from utils import calc_class_weights


class MLPClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        dataset: CIDataset,
        class_weight_constant:List[float],
        class_weight_exponent: float,
        num_unique_labels: int,
        device
    ):
        """Create a Flower Client with the provided ID and assign it the provided data.

        Args:
            cid (str): the ID of the created client
            dataset (CIDataset): the client's data
            class_weight_exponent (float): a modifier for the degree to which weights are adjusted
            num_unique_labels (int): a list of the labels that can be found in the whole data set (not just this client's subset)
        """

        self.cid = cid
        self.dataset = dataset
        self.device = device

        self.class_weights = calc_class_weights(
            num_unique_labels, self.dataset.get_labels(), class_weight_constant, class_weight_exponent
        )

        self.n_examples = len(self.dataset)

    def fit(
        self, parameters: List[float], config: SimpleNamespace
    ) -> Tuple[List[float], int, dict]:
        """Trains a model on the client's data using the provided parameters as a starting point.

        Args:
            parameters (List[float]): the starting parameters of the model
            config (SimpleNamespace): the training parameters. Defaults to {}.

        Returns:
            Tuple[List[float], int, dict]: the updated model parameters, the lenght of the client's data set, and the metrics of interest, respectively
        """
    
        # create a model from scratch (create it each time so that it doesn't just sit and take up memory)
        model = model_utils.get_model(
            config.input_size,
            config.num_neurons_per_layer,
            len(self.class_weights),
            config.dropout_prob,
        )

        # set the model parameters to those sent over by the server
        model = model_utils.set_model_parameters(model, parameters)

        # create a dataloader for the client's data set
        dataloader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=config.shuffle
        )

        # train a model on the client's data
        print(f"Started training the model of client {self.cid}...", flush=True)
        _ = model_utils.train_model(
            model, dataloader, self.class_weights, config, self.device
        )

        # extract the updated parameters
        updated_params = model_utils.get_model_parameters(model)

        # return the parameters, the number of examples, and metrics
        return updated_params, self.n_examples, {}

    def evaluate(
        self, parameters: List[float], config: SimpleNamespace
    ) -> Tuple[float, int, dict]:
        """Evaluates a model, whose parameters are set to the ones provided, on the client's data.

        Args:
            parameters (List[float]): a 1D representation of the model parameters to be evaluated
            config (dict, optional): the evaluation parameters. Defaults to {}.

        Returns:
            Tuple[float, int, dict]: the loss of the model on the client's data set, the length of the client's dataset, and the metrics of interest, respectively
        """

        # create a model from scratch (create it each time so that it doesn't just sit and take up memory)
        model = model_utils.get_model(
            config.input_size,
            config.num_neurons_per_layer,
            len(self.class_weights),
            config.dropout_prob,
        )
        # set the model parameters to those sent over by the server
        model = model_utils.set_model_parameters(model, parameters)

        # create a dataloader for the client's data set
        dataloader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=False
        )

        # predict the outputs of the model on the client's dataset
        groundtruth, predictions, prediction_scores = model_utils.predict(
            model, dataloader, self.device
        )

        # calculate the loss on this data
        loss = evaluation_utils.calculate_crossentropy_loss(
            groundtruth=groundtruth,
            pred_scores=prediction_scores,
            class_weights=self.class_weights,
        )

        # we don't calculate metrics here as there is a complication with clients observing less than all classes
        return float(loss), self.n_examples, {}


class XGBClient(fl.client.Client):
    def __init__(
        self,
        cid: str,
        dataset: CIDataset,
        class_weight_constant:List[float],
        class_weight_exponent: float,
        num_unique_labels: int,
    ):
        """Create a Flower Client with the provided ID and assign it the provided data.

        Args:
            cid (str): the ID of the created client
            dataset (CIDataset): the client's data
            class_weight_exponent (float): a modifier for the degree to which weights are adjusted
            num_unique_labels (int): a list of the labels that can be found in the whole data set (not just this client's subset)
        """

        # save the client id
        self.cid = cid

        # calculate the weights that should be assigned to each label
        self.class_weights = calc_class_weights(
            num_unique_labels, dataset.get_labels(), class_weight_constant, class_weight_exponent
        )

        # assign the labels weights to each sample in the client's data set
        sample_weights = []
        for label in dataset.get_labels():
            sample_weights.append(self.class_weights[label])

        # construct a dmatrix
        self.train_dmatrix = xgb.DMatrix(
            dataset.get_features(), label=dataset.get_labels(), weight=sample_weights
        )

        # save the number of examples in the client and the number of unique labels the client will encounter
        self.n_examples = dataset.get_features().shape[0]

        self.n_unique_labels = len(np.unique(dataset.get_labels()))

        self.bst = None

    def __build_model__(self, config: SimpleNamespace):
        hyperparameters = {
            "eta": config.learning_rate,  # Learning rate
            "max_depth": config.max_depth,
            "num_parallel_tree": 1,
            "subsample": config.subsample_ratio,
            "colsample_bytree": config.colsample_bytree,
            "reg_lambda": config.reg_lambda,
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": "hist",
            "random_state": config.random_seed
        }

        self.bst = xgb.Booster(params=hyperparameters)

    def __set_model_params__(self, params: Parameters):
        surrogate = None
        for item in params.tensors:
            surrogate = bytearray(item)

        # load the surrogate model into the booster
        self.bst.load_model(surrogate)

    def __local_boost__(self, num_local_rounds):
        # update trees based on local training data.
        for _ in range(num_local_rounds):
            self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())

        # extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_local_rounds : self.bst.num_boosted_rounds()
        ]

        return bst

    def get_parameters(self, ins: GetParametersIns = None) -> GetParametersRes:
        if self.bst is not None:
            local_model = self.bst.save_raw("json")
            local_model_bytes = bytes(local_model)

            return GetParametersRes(
                status=Status(
                    code=Code.OK,
                    message="OK",
                ),
                parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            )
        else:
            return GetParametersRes(
                status=Status(
                    code=Code.OK,
                    message="No model to return",
                ),
                parameters=Parameters(tensor_type="", tensors=[]),
            )

    def fit(self, ins: FitIns) -> FitRes:
        """Trains a model on the client's data using the provided parameters as a starting point.

        Args:
            ins (FitIns): the training instructions containing (global) model parameters received from the server and a dictionary of configuration values


        Returns:
            FitRes: the training result containing updated parameters and other details such as the number of local training examples used for training
        """
        if self.bst is None:
            hyperparameters = {
                "eta": ins.config.learning_rate,  # Learning rate
                "max_depth": ins.config.max_depth,
                "num_parallel_tree": 1,
                "subsample": ins.config.subsample_ratio,
                "colsample_bytree": ins.config.colsample_bytree,
                "reg_lambda": ins.config.reg_lambda,
                "objective": "multi:softprob",
                "num_class": 3,
                "tree_method": "hist",
                "random_state": ins.config.random_seed
            }
            
            bst = xgb.train(
                hyperparameters,
                self.train_dmatrix,
                num_boost_round=ins.config.n_estimators,
                evals=[(self.train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self.__local_boost__(ins.config.n_estimators)

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.n_examples,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.bst is None:
            self.__build_model__(ins.config)
            self.__set_model_params__(ins.parameters)

        eval_results = self.bst.eval_set(
            evals=[(self.train_dmatrix, "train")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.n_examples,
            metrics={"AUC": auc},
        )


def create_client_with_args(
    data: Dict[str, CIDataset],
    class_weight_exponent: float,
    class_weight_constant:List[float],
    unique_labels: List[float],
    type: str,
    device
) -> Callable:
    """A wrapper that returns a create_client function with embedded additional parameters.

    Args:
        data (Dict[str, CIDataset]): a dictionary containing the data of all clients in the complete dataset
        class_weight_exponent (float): a modifier for the degree to which weights are adjusted
        unique_labels (List[float]): a list of the labels that can be found in the whole data set (not just a specific client's subset)
        type (str): the type of client to be created, either an MLP client or an XGBoost client
    Returns:
        Callable[[str], Client]: a function for creating clients with just one parameter
    """

    if type == "mlp":

        def create_client(cid: str) -> MLPClient:
            """Creates and instance of the Client class

            Args:
                cid (str): the ID of the client to be created

            Returns:
                Client: an instance of the Client class containing the data set provided in the wrapper function
            """

            client_data = data[cid]
            return MLPClient(cid, client_data, class_weight_constant, class_weight_exponent, unique_labels, device)

        return create_client

    elif type == "xgb":

        def create_client(cid: str) -> XGBClient:
            """Creates and instance of the Client class

            Args:
                cid (str): the ID of the client to be created

            Returns:
                Client: an instance of the Client class containing the data set provided in the wrapper function
            """

            client_data = data[cid]
            return XGBClient(cid, client_data, class_weight_constant, class_weight_exponent, unique_labels)

        return create_client
