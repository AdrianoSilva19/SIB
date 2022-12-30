from typing import Callable
import numpy as np


from si.statistics.f_classification import f_classification
from si.data.dataset import Dataset


class SelectPercentile:
    def __init__(self,score_function: Callable = f_classification, percentile: float = 0.25) -> None:
        """
        Select the higest scoring features according to the percentile given.
        Scores are computed using ANOVA F-values between label/feature.
        :param score_function: Function that takes a dataset and returns the score and p-values. Default to f_classification.
        :type score_function: Callable
        :param percentile: Percentile of the features to be selected. Default to 0.25
        :type percentile: float
        """
        self.score_function=score_function
        self.percentile= percentile
        self.F=None
        self.p=None

    def fit (self,dataset):
        """
        Method that fits the dataset
        :param dataset: Dataset to be fited
        """
        self.F,self.p=self.score_function(dataset)
        return self

    def transformer (self,dataset):
        """
        Method to transform the dataset, selects the highest scores according to the percentile
        :param dataset: Dataset to be transformed
        :return: Dataset transformed
        :rtype: Dataset
        """
        len_features = len(dataset.features)
        percentile = int(len_features * self.percentile)
        idxs = np.argsort(self.F)[-percentile:] # queremos as mehores, com o limite do percentil
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.labels)


    def fit_transform(self,dataset):
        """
        Runs fit and transform methods in the given dataset
        :param dataset: A dataset object
        :return: A dataset object with the features with the highest percentile scores
        :rtype: Dataset
        """
        self.fit(dataset)
        return self.transformer(dataset)


if __name__ == "__main__":
    a = SelectPercentile(percentile = 0.6)
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a.fit(dataset)
    b = a.transformer(dataset)
    print(b.features) 