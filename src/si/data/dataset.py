import pandas as pd
import numpy as np



class Dataset:
    def __init__(self,X,Y,features,labels) -> None:
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            features (_type_): _description_
            labels (_type_): _description_
        """
        self.X=X  # numpy array
        self.Y=Y   # array de uma dimensão
        self.features=features  # lista de strings
        self.labels=labels # string 


    def shape(self):
        """dimensões do dataset
        """
        return self.X.shape

    def has_label(self):
        """ 
        """
        if self.labels is not None:
            return True
        else:
            return False


    def get_class(self):
        pass

    def get_stats(self):
        line = '\n'
        return f"Media:{np.mean(self.X,axis=0)}{line}Variancia:{np.var(self.X,axis=0)}"

    def summary(self,x,y):
        pass




if __name__ == "__main__":
    x= np.array([[1,2,3],[1,2,3]])
    y=np.array([1,2])
    features=["A","B","C"]
    label="y"
    dataset=Dataset(X=x,Y=y,features=features,labels=None)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_stats())