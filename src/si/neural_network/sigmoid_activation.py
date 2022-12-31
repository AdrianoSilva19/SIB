import numpy as np
from si.statistics.sigmoid_function import sigmoid_function

class SigmoidActivation:
    def __init__(self):
        # attribute
        self.x = None

    @staticmethod
    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the layer.
        :param input_data: input data
        :return: Returns the input data multiplied by the weights.
        """

        return sigmoid_function(input_data)

    @staticmethod
    def backward(input_data: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the layer.
        :return: Returns the error of the previous layer.
        """
        # multiplication of each element by the derivative and not by the entire matrix

        sigmoid_derivative = sigmoid_function(input_data) * (1 - sigmoid_function(input_data))

        error_to_propagate = error * sigmoid_derivative

        return error_to_propagate