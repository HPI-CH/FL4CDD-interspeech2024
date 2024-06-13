"""
Module Name: model_utils
Description: This module contains all functions related to model training.

Functions:
- get_model: constructs a simple classification model that accepts either embeddings or interpretable features as input
- train_one_epoch: trains a given model on the provided data for one epoch
- train_model: train a given model on the provided data for the specified number of epochs
- predict_batch: calculates the labels and logits produced by the provided model when running on the examples in the provided batch
- predict: calculates the outputs of a model for the provide data
- get_model_parameters: a getter method for the parameters of the model
- set_model_parameters: a setter method for the parameters of a model

#! to-do: fix the situation when not all optimizers and schedulers take the same arguments
"""
from typing import List, Tuple
from types import SimpleNamespace
from collections.abc import Iterable
from collections import OrderedDict
import importlib

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from utils import RunningAvg

def get_model(
    num_features: int,
    num_neurons_per_layer: list,
    num_classes: int,
    dropout_prob: float,
) -> nn.Module:
    """Constructs a simple classification model that accepts either embeddings or interpretable features as input.

    Args:
        num_features (int): the number of features that the classification head should expect
        num_neurons_per_layer(list): the number of layers
        num_classes (int): the number of possible output choices
        dropout_prob (float): the probability of a dropout

    Returns:
        nn.Module: a PyTorch classification model
    """
    print("Creating a PyTorch model...", end="")
    model = nn.Sequential(
        nn.Linear(num_features, num_neurons_per_layer[0]),
        nn.ReLU(),
        nn.Dropout(p=dropout_prob),
        nn.Linear(num_neurons_per_layer[0], num_neurons_per_layer[1]),
        nn.ReLU(),
        nn.Dropout(p=dropout_prob),
        nn.Linear(num_neurons_per_layer[1], num_classes),
    )

    # Apply He initialization to the linear layers
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    print("done")

    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    class_weights: List[float],
    optimizer: Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Trains a given model on the provided data for one epoch.

    Args:
        model (nn.Module): the model to be trained
        dataloader (DataLoader): a dataloader for the training data
        class_weights (List[int]): the weight that should be assiged to each of the classes
        optimizer (Optimizer): the optimizer to use for training
        device (str): the ID of the device where the calculation should be executed

    Returns:
        Tuple[float, float]: the segment- and speaker-level loss of the model
    """
    model.train()
    # send the model to the specified device
    model.to(device)

    # initialze a loss function
    loss_fcn = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)).to(device)

    # initialize instance of a running average for the loss and accuracy
    average_segment_loss = RunningAvg()

    with tqdm(total=len(dataloader.dataset), desc="Single epoch training") as t:
        for i, data in enumerate(dataloader):
            # Every data instance is an input + label + speaker_id tuple
            inputs, labels, _ = data

            # send the data and labels to the same device as the model
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)
            outputs = torch.reshape(outputs, (-1, len(class_weights)))

            # calculate the loss
            loss = loss_fcn(outputs, labels)
            loss.backward()

            optimizer.step()

            # update the running loss average with the loss value of this batch
            average_segment_loss.update(loss.cpu().item())

            t.update(len(labels))
            t.set_postfix({"avg_loss": f"{average_segment_loss():.4f}"})

    return average_segment_loss(), 0.0


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_weights: List[float],
    train_config: SimpleNamespace,
    device: str,
) -> np.ndarray:
    """Train a given model on the provided data for the specified number of epochs.

    Args:
        model (nn.Module): the model to be trained
        dataloader (DataLoader): a dataloader for the training data
        class_weights (List[int]): the weight that should be assiged to each of the classes
        train_config (SimpleNamespace): contains training parameters such as but not limited to:
            - num_epochs
            - batch_size
            - optimizer
            - learning_rate
            - lr_scheduler
            - scheduler_gamma
            - weight_decay
            - report_steps
        device (str): the ID of the device where the calculation should be executed

    Returns:
        np.ndarray: the segment-level loss of the model after each epoch of training
    """
    
    optimizer_class = getattr(importlib.import_module("torch.optim"), train_config.optimizer)

    optimizer = optimizer_class(
       model.parameters(),
       lr=train_config.learning_rate,
       weight_decay=train_config.weight_decay,
    )

    if train_config.lr_scheduler is not None:
        lr_scheduler_class = getattr(importlib.import_module("torch.optim.lr_scheduler"), train_config.lr_scheduler)

        scheduler = lr_scheduler_class(
            optimizer, gamma=train_config.scheduler_gamma
        )
    
    segment_losses = []

    for epoch in range(train_config.num_epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        segment_loss, _ = train_one_epoch(
            model, dataloader, class_weights, optimizer, device
        )

        segment_losses.append(segment_loss)

        if ((epoch) % train_config.report_steps == 0) or (
            epoch == train_config.num_epochs - 1
        ):
            print(
               "Epoch {} - Training (segment) loss: {:.6f} - LR: {:.6f}".format(
                   epoch, segment_loss, optimizer.param_groups[0]["lr"]
               )
            )
        
        if train_config.lr_scheduler is not None:
            scheduler.step()

    return np.array(segment_losses)


def predict_batch(model: nn.Module, batch: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the labels and per-class scores predicted by the model when running on the examples in the batch.

    Args:
        model (nn.Module): the model used to make the predictions
        batch (Iterable): the set of examples to be predicted

    Returns:
        Tuple[np.ndarray, np.ndarray]: the predicted labels, and the per-class scores, respectively
    """
    preds, scores = None, None

    # make predictions for the batch
    with torch.no_grad():
        scores = model(batch)

    # for each example, calculate a single predictions based on highest score
    preds = torch.argmax(scores, dim=-1)

    # detach from compute graph, send to cpu and cast to a numpy array
    preds = preds.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    return preds, scores


def predict(
    model: nn.Module, dataloader: DataLoader, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the outputs of the provided model on the provided data.

    Args:
        model (nn.Module): the model to be evaluated
        dataloader (DataLoader): a dataloader for the evaluation data
        device (str): the ID of the device where the calculation should be executed

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: the groundtruth labels, the predicted labels, and the predicted per-class scores, respectively
        #! Note: The shapes of these arrays are:
        #!          - (-1,)
        #!          - (-1,)
        #!          - (-1, 3)
    """
    
    targets, predictions, scores = np.array([]), np.array([]), np.empty((0, 3), dtype=float)
   
    # set the model to evaluation mode
    model.eval()
    # send the model to the specified device
    model.to(device)

    with tqdm(total=len(dataloader.dataset), desc="Evaluating model:") as t:
        # iterate through the bathces of the dataset and make predictions
        for i, (inputs, labels, _) in enumerate(dataloader):
        
            # send the data the same device as the model
            inputs = inputs.to(device)
            # send to cpu and cast to a numpy array
            labels = labels.cpu().numpy()
            

            # make predictions for this batch
            batch_preds, batch_scores = predict_batch(model, inputs)

            batch_scores = np.reshape(batch_scores, (len(batch_scores), -1))

            predictions = np.append(predictions, batch_preds)
            scores = np.append(scores, batch_scores, axis=0)
            
            # convert one-hot encoded labels to single class labels
            class_labels = np.argmax(labels, axis=1)
            targets = np.append(targets, class_labels)
            
    return targets, predictions, scores


def get_model_parameters(model: nn.Module) -> List[float]:
    """A getter method for the parameters of the model.

    Args:
        model (nn.Model): the model whose parameters we want to extract

    Returns:
        List[float]: a 1D representation of the model weights
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[float]) -> nn.Module:
    """A setter method for the parameters of a model.

    Args:
        model (nn.Model): the model whose parameters we want to set
        parameters (List[float]): a 1D representation of the parameter values

    Returns:
        nn.Model: the model after its weights have been replaced
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    return model
