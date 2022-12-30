from typing import Callable
import pandas as pd
import numpy as np

from si.statistics.f_classification import f_classification
from si.data.dataset import Dataset


class SelectKBest:
    def __init__(self,score_function: Callable = f_classification, k: int= 10):
        """
        Select features according to the k highest scores.
        The scores are computed using ANOVA F-values between label/feature
        :param score_function: Function that takes the dataset and returns scores and p-values
        :type score_function: Callable
        :param k: Number of top features to select. Default to 10
        :type k: int
        """
        self.score_function=score_function
        self.k=k
        self.F=None
        self.p=None 

    def fit(self,dataset):
        """
        Fits SelectKBest to compute the F scores and p_values.
        :param dataset: dataset object
        :type dataset: Dataset
        """
        self.F,self.p=self.score_function(dataset)
        return self


    def transformer (self,dataset):
        """
        It transforms the dataset by selecting k highest score features
        :param dataset: Dataset object
        :type dataset: Dataset
        :return: Dataset object
        :rtype: Dataset
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.labels)

    def fit_transform(self, dataset):
        """
        It fits SelectKBest and then transforms the dataset by selecting the k highest score features.
        :param dataset: Dataset object
        :type dataset: Dataset
        :return: A dataset object with the k highest score features
        :rtype: Dataset
        """
        self.fit(dataset)
        return self.transformer(dataset)


if __name__ == "__main__":
    a = SelectKBest(k = 3)
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a.fit(dataset)
    b = a.transformer(dataset)
    print(b.features) 