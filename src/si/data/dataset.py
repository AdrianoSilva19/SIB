import pandas as pd
import numpy as np
from typing import Tuple, List


class Dataset:
    
    def __init__(self, X:np.ndarray = None, y:np.ndarray = None, features:List = None, label:str = None):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            features (_type_): _description_
            labels (_type_): _description_
        """
        self.X=X  # numpy array
        self.y=y   # array de uma dimensão
        self.features=features  # lista de strings
        self.labels=label # string 


    def get_shape(self):
        """
        Returns: indicates the dimension of the dataset 
        """
        
        return self.X.shape
    
    def has_label(self):
        """
        Returns: indicates if exists label or not 
        """
        
        if self.y is not None:
            return True
        else:
            return False
    
    def get_classes(self):
        """
        Returns: label class as list 
        """
        
        if self.y is None:
            raise Exception("You've an unsupervised dataset")
        else:
            return np.unique(self.y)
        
    def get_mean(self):
        """
        Returns: mean of each feature
        """
        return np.mean(self.X, axis=0)
    
    def get_variance(self):
        """
        Returns: variance of each feature
        """
        return np.var(self.X, axis=0)
    
    def get_median(self):
        """
        Returns: median of each feature
        """
        return np.median(self.X, axis=0)
    
    def get_min(self):
        """
        Returns: minimum of each column(feature)
        """
        return np.min(self.X, axis=0)
    
    def get_max(self):
        """
        Returns: maximun of each column
        """
        return np.max(self.X, axis=0)
    
    def summary(self):
        """
        Returns: dictionary with the stats of the dataset
            
        """
        return pd.DataFrame(
            {"mean": self.get_mean(),
             "median": self.get_median(),
             "variance": self.get_variance(),
             "min": self.get_min(),
             "max": self.get_max()}
        )
    
    def dropna (self):
        """Class method that removes samples with atleast one null (NaN) value."""
        
        return pd.DataFrame(self.X).dropna(axis=0).reset_index(drop=True)
    
    def fillna(self, value: int):
        """Class method that fills all NaN values with the given value
        Args:
            value (int): Given value to replace null values with
        """
        
        return pd.DataFrame(self.X).fillna(value)
    
    def print_dataframe(self):
        """Prints dataframe in pandas DataFrame format
        """
        return pd.DataFrame(self.X, columns=self.features, index=self.y)
        



if __name__ == "__main__":
    x= np.array([[1,2,3],[1,2,3]])
    y=np.array([1,2])
    features=["A","B","C"]
    label="y"
    dataset=Dataset(X=x,Y=y,features=features,labels=None)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.summary())