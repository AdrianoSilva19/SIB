import numpy as np
from math import sqrt

def rmse(y_true:int,y_pred:int):
    """
    This function calculates the error between real and predicted y using the RMSE formula

    Args:
        y_true (int): real values from y
        y_pred (int): estimated values from y

    Returns:
        _type_: RMSE between real and predicted values
    """
    rmse = sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))

    return rmse