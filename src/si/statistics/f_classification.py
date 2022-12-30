import pandas as pd
import numpy as np
from scipy import stats

from si.data.dataset import Dataset
from typing import Tuple, Union


def f_classification(dataset:Dataset):
    """
    Scoring function for classification. Group samples by class and computes ANOVA's for the samples
    returning F and p values.
    :param dataset: A dataset object
    :type dataset: Dataset
    :return: Tuple of np arrays with F score and p-values
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F,p = stats.f_oneway(*groups)
    return F,p
