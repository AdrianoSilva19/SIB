import numpy as np


def rmse(y_true:int,y_pred:int):
    """
    This function calculates the error between real and predicted y using the RMSE formula

    Args:
        y_true (int): real values from y
        y_pred (int): estimated values from y

    Returns:
        _type_: RMSE between real and predicted values
    """
    
    return np.sqrt(((y_true - y_pred) ** 2).mean())