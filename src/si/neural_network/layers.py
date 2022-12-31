import numpy as np



class Dense:
    def __init__(self, input_size: int, output_size: int):
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        # weight matriz initialization
        shape = (input_size, output_size)

        self.x = None
        self.weights = np.random.randn(*shape) * 0.01  # 0.01 is a hyperparameter to avoid exploding
        # gradients
        # each layer receives a weight that multiplies by the input that are then summed
        self.bias = np.zeros((1, output_size))  # bias initialization, receives a bias to avoid overfitting

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the layer.
        :param x: input data value
        :return: Returns the input data multiplied by the weights.
        """
        self.x = x
        # the input_data needs to be a matrix with the same number of columns as the number of features
        # the number os columns of the input_data must be equal to the number of rows of the weights
        return np.dot(x, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Computes the backward pass of the layer
        :param error: error value of the loss function
        :param learning_rate: learning rate
        :return: Returns the error of the previous layer.
        """

        error_to_propagate = np.dot(error, self.weights.T)

        # updates the weights and bias
        self.weights = self.weights - learning_rate * np.dot(self.x.T, error)  # x.T is used to multiply the error by
        # the input data due to matrix multiplication rules

        self.bias = self.bias - learning_rate * np.sum(error, axis=0)  # sum because the bias has the dimension of
        # nodes and the error has the dimension of samples and nodes (batch size, nodes)

        return error_to_propagate