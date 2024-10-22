import numpy as np


def sigmoid(z):
    """
    calculate sigmoid function
    :param z: vector or Numpy array
    :return:
    """
    return 1 / (1 + np.exp(-z))
class EVOLUNET():
    def __init__(self, size):
        self.size = size
        self.num_layers = len(self.size)
        self.bias = [np.random.randn(y,1) for y in size[1:]]
        self.weights = [np.random.randn(x,y) for (x,y) in zip(size[:-1],size[1:])

    def feedforward(self, a):
        """
        return output of the network if inputs
        :param a: input vector
        :return:
        """
        for b, w in zip(self.bias, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a