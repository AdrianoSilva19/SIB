from typing import Optional, Union
import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ",", features: Optional[bool] = True, label: Optional[bool] = False) -> Dataset:
    """Function that reads csv file and returns a Dataset object of that file.
    Args:
        filename (str): name/path of file
        sep (str): separator between values. Defaults to , .
        features (Optional[bool], optional): If the csv file has feature names. Defaults to True.
        label (int): If the dataset has defined labels. Defaults to False
    Returns:
        Dataset: The dataset object
    """
    dataframe = pd.read_csv(filename, sep=sep)

    if features:
        features_dataframe = dataframe.iloc[:, :-1].to_numpy()
        features_names = dataframe.columns[:-1].tolist()
    else:
        features_dataframe = None
        features_names = None

    if label:
        y = dataframe.iloc[:, -1].to_numpy()
        label_name = dataframe.columns[-1]
    else:
        y = None
        label_name = None

    return Dataset(features_dataframe, y, features_names, label_name)


def write_csv(dataset: Dataset, filename: str, sep: str = ",", features: Optional[bool] = True,
              label: Optional[bool] = True) -> None:
    """Writes a csv file from a dataset object
    Args:
        dataset (_type_): Dataset to save on csv format
        filename (str): Name of the csv file that will be saved
        sep (str, optional): Separator of values. Defaults to ",".
        features (Optional[bool], optional): Boolean value that tells if the dataset object has feature names. Defaults to True.
        label (Optional[bool], optional): Boolean value that tells if the dataset object has label names Defaults to True.
    """
    csv = pd.DataFrame(data=dataset.X)

    if features:
        csv.columns = dataset.features

    if label:
        csv.insert(loc=0, column=dataset.label, value=dataset.y)
        # csv[dataset.label] = dataset.y

    csv.to_csv(filename, sep=sep, index=False)


if __name__ == "__main__":
    file = r"C:\Users\35193\Desktop\sib\SIB\datasets\iris.csv"
    a = read_csv(filename=file,label=True, features=True)
    print(a.print_dataframe())
