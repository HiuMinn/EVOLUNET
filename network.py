import numpy as np
import random


class EVOLUNET():
    def __init__(self, size,bias = None,weights = None):
        self.size = size
        self.num_layers = len(self.size)
        if bias is None:
            self.bias = []
        else:
            self.bias =bias
        if weights is None:
            self.weights = []

    def new_network(self):
        """
        initiate a new random parameters of the network
        :param self:
        :return:
        """
        self.bias = [np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [np.random.randn(x, y) for (x, y) in zip(self.size[:-1], self.size[1:])]
        return None

    def mutate_param (self,mutation_rate):
        """
        regarding a neural network as a individual in the population,
        mutate the bias and weights
        :return:
        """
        for i in range(len(self.bias)):
            if random.random()< mutation_rate:
                self.bias[i] += random.uniform(-1,1)

        for i in range(len(self.weights)):
            if random.random()< mutation_rate:
                self.weights[i] += random.uniform(-0.1,0.1)
        return None

    def mutate(self):
        sign_weights = np.random.choice([-1,1], size=self.weights)
        mutate_weights =
        sign_bias = np.random.choice([-1,1], size=self.bias)
        mutate_bias = 0.1
        self.bias = [self.bias[i] + sign_bias[i]*mutate_bias for i in range(len(self.bias))]
        self.weights = [self.weights[i] + sign_weights[i]*mutate_weights for i in range(len(self.weights))]

    def feedforward(self, a):
        """
        return output of the network if inputs
        :param a: input vector
        :return:
        """
        for b, w in zip(self.bias, self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
        return a

    def sigmoid(self,z):
        """
        calculate sigmoid function
        :param z: vector or Numpy array
        :return:
        """
        return 1 / (1 + np.exp(-z))

    def evaluate(self, test_data):
        """
        Return the number of correct predictions. Seen as fitness for the GA
        :param test_data:
        :return:
        """
        test_output = [(self.feedforward(test_data),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_output)


    # def evaluate_by_matrix(self,test_data):
    #     test_output = [(self.feedforward(test_data), y) for (x, y) in test_data]
    #     tp=tn=fn=fp =0
    #     for (x,y) in test_output:
    #         tp += int(x==y)
    #         fp += int(x!=y)
    #     return 0