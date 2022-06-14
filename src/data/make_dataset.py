from importlib.resources import path
from pathlib import Path
import tensorflow as tf
from typing import Tuple
from scipy.io import arff
import torch
import numpy as np


def get_eeg() -> path:
    """
    This function downloads the eeg dataset from:
    https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
    and stores it in the given data dir. 
    """
    data_dir = "../../data/raw"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
    datapath = tf.keras.utils.get_file(
        "eeg", origin=url, untar=False, cache_dir=data_dir
    )

    return datapath


class BaseDataset:
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.dataset: List = []
        self.max_length: int = 0
        self.min_length: int = 0
        self.process_data()

    def process_data(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def get_max_dist(self) -> int:
        return(self.max_length)

    def __getitem__(self, idx: int) -> Tuple:
        return self.dataset[idx]

class EegDataset(BaseDataset):
    # this is called inheritance.
    # we get all the methods from the BaseDataset for free
    # Only thing we need to do is implement the process_data method
    def process_data(self) -> None:
        data = arff.loadarff(str(self.path))
        observations = []
        current_label = int(data[0][0][14])

        for x in data[0]:
            # compares old label with new label
            if ( int(x[14]) != current_label):
                joined_tensor = torch.stack(observations)
                if (len(observations) > self.max_length):
                    self.max_length = len(observations)

                self.dataset.append((joined_tensor, current_label))
                observations = []
                current_label = int(x[14])

            X_ = np.array((tuple(x))).astype(np.float)
            x_data = torch.tensor(X_[:13])
            observations.append(x_data)
        # last one also needs to be stored, is that happening?