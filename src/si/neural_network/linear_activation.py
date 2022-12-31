
import numpy as np

class LinearActivation:
    def __init__(self):
        pass

    @staticmethod
    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Computes the linear relationship.
        :param input_data: input data
        :return: Returns the linear relationship.
        """

        return input_data