from typing import Optional, Union
import pandas as pd
import numpy as np

from si.data.dataset import Dataset

def read_csv(filename:str, sep:str = ",", features: Optional[bool] = True, label: Union[None, int]= None):
    imported_data = pd.read_csv(filepath_or_buffer=filename, sep=sep)
    data = imported_data.values.tolist()
    headers = list(imported_data.columns)
    header_label = headers[label]
    
    if features:
        if label is not None:
            del headers[label]
    else:
        headers = None
    
    if label is not None:
        y = list(imported_data.iloc[:, label])
        imported_data = imported_data.drop(imported_data.columns[label], axis=1)
        data = imported_data.values.tolist()
    else: y = None
    
    return Dataset(X=data, y=y, features = headers, label = header_label)
        
    
def write_csv(dataset, filename: str, sep: str = ",", features: Optional[bool] = True, label: Optional[bool] = True):
    csv = pd.DataFrame(data=dataset.X)
    
    if features:
        csv.columns = dataset.features
    
    if label:
        csv.insert(loc=0, column=dataset.label, value=dataset.y)
        
    csv.to_csv(filename, sep = sep, index=False)
    
    
if __name__ == "__main__":
    file = r"C:\Users\ampsi\OneDrive\Ambiente de Trabalho\segundo_ano\SIB\datasets\iris_missing_data.csv"
    a = read_csv(filename=file, sep = ",", features=True, label=4)
    # print(a.dropna())
    print(a.fillna(100))