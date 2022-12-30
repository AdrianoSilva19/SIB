
import sys
import os
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset:Dataset,test_size: float = 0.2, random_state: int = 40)-> tuple[Dataset, Dataset]:
    """
    Random splits a dataset into a train and a test set
    :param dataset: Dataset object
    :type dataset: Dataset
    :param test_size: size of the test dataset. Defaults to 0.2
    :type test_size: float
    :param random_state: Seed to feet the random permutations. Defaults to 40
    :type random_state: int
    """
    np.random.seed(random_state)

    len_samples = dataset.get_shape()[0]
    len_test = int(test_size * len_samples)
    permutations = np.random.permutation(len_samples)
    test_split = permutations[:len_test]
    train_split = permutations[len_test:]
    train = Dataset(dataset.X[train_split], dataset.y[train_split], features=dataset.features,
                    label=dataset.labels)

    test = Dataset(dataset.X[test_split], dataset.y[test_split], features=dataset.features,
                   label=dataset.labels)
    return train,test