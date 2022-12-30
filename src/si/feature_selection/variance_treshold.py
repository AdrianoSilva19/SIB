import numpy as np
from si.data.dataset import Dataset



class VarianceThreshold:
    def __init__ (self, threshold = 0.0):
        """
        The variance threshold represents a baseline approach for feature selection.
        Removes all features wich variance doesn't meet a threshhold given by the user.
        :param threshold: Non negative threshold. Defaults to 0.
        :type threshold: int
        """
        self.threshold=threshold
        self.variance=None

    def transform(self, dataset):
        """
        Selects all Features with variance higher than the threshold and returns a new dataset with the selected features
        :param dataset: Dataset to be transformed
        :type dataset: Dataset
        :return: Dataset object with best Features
        :rtype: Dataset
        """
        mask= self.variance>self.threshold
        newX=dataset.X[:,mask]
        features= np.array(dataset.features)[mask]
        return Dataset(X=newX,y=dataset.y,features=list(features),label=None)


    def fit (self, dataset):
        """
        Calculate the variance of each feature in a dataset
        :param dataset: Dataset object
        :type dataset: Dataset
        """
        variance=dataset.get_variance()
        self.variance=variance
        return self

    def fit_transform(self, dataset):
        """
        Runs fit and transform method automatically
        :param dataset: Dataset object
        :type dataset: Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    dataset = Dataset(X=np.array([[0, 2, 0, 5],
                                  [0, 1, 4, 1],
                                  [0, 1, 1, 1]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    
    b = VarianceThreshold(3)
    b = b.fit_transform(dataset)
    print(b.features)