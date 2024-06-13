"""
Module Name: dataset
Description: This module defines a dataset class.

Classes & Methods:
CIDataset: Extends the PyTorch Dataset class with a specific implementation fitting this project's cognitive impairment data set
"""
from typing import Tuple, List

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class CIDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_columns: List[str]):
        """Generates a PyTorch Dataset instance using the provided data.
        #! Assumes that the data has already been preprocessed!

        Args:
            data (pd.DataFrame): the data to be included
            feature_columns (List[str]): a list of feature names
        """
        # separate the metadata
        self.metadata = data[["ID", "age", "gender", "mmse", "dataset", "split"]]

        # separate the features
        self.features = data[feature_columns].values

        # sepearate the labels
        self.labels = data["label"].values
        # one_hot_encode the labels
        self.labels = self.__one_hot_encode_labels__(self.labels)

    def __one_hot_encode_labels__(self, labels: np.ndarray) -> np.ndarray:
        """OneHot encodes the the labels that are passed to it.
        #! Assumes that the number of classes is three
        Args:
            labels (np.ndarray): the labels to be encoded

        Returns:
            np.ndarray: the OneHot encoded labels
        """

        one_hot_encoded_labels = [
            [(label == 0) * 1.0, (label == 1) * 1.0, (label == 2) * 1.0]
            for label in labels
        ]
        return np.array(one_hot_encoded_labels)

    def __len__(self) -> int:
        """Returns the number of instances in the data set.

        Returns:
            int: the number of instances in the data set
        """
        return self.features.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Returns the instance of the data set at the specified index, as a torch.Tensor.

        Args:
            index (int): the index of the instances to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]: the instance at the appropriate index,
                                                    the label of the instance,
                                                    the ID of the speaker to whom it belongs,
                                                    respectively
"""

        features = torch.tensor(self.features[index, :], dtype=torch.float)
        labels = torch.tensor(self.labels[index], dtype=torch.float)
        speaker_id = self.metadata["ID"].iloc[index]
      
        return (
            features,
            labels,
            speaker_id,
        )

    def get_features(self) -> np.ndarray:
        """Returns a simple view of the client's data

        Returns:
            np.ndarray: the client's data
        """
        return self.features

    def get_labels(self) -> np.ndarray:
        """Returns a simple view of the client's labels.
        Example: [1, 2, 1, 0, 0]

        Returns:
            np.ndarray: the client's labels
        """
        return np.argmax(self.labels, axis=1)

    def get_metadata(self) -> pd.DataFrame:
        """Returns the metadata about the instances in this dataset.

        Returns:
            pd.DataFrame: a pd.DataFrame containing the ID, age, gener, mmse of the speaker of each instance in the dataset, as well as the dataset which the instance is comming from.
        """

        return self.metadata
