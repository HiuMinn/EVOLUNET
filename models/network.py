import numpy as np
import random
import matplotlib.pyplot as plt



class EVOLUNET():
    def __init__(self, size,bias = None,weights = None):
        """

        :param size:
        :param bias: If no bias given, generate randomly by default
        :param weights: If no weights given, generate randomly by default
        """
        self.size = size
        self.num_layers = len(self.size)
        self.scale_factor = 1
        self.std_weights = 1
        self.std_bias = 1
        if bias is None or weights is None:
            self.new_param()
        else:
            self.bias = bias
            self.weights = weights
        self.fitness = 0
    def new_param(self):
        """
        initiate a new random parameters of the network
        :param self:
        :return:
        """
        self.bias = [self.scale_factor * np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [self.scale_factor * np.random.randn(y, x) for x, y in zip(self.size[:-1], self.size[1:])]
        return None

    def mutate_param (self, reset_prob = 0.09):
        """
        regarding a neural network as a individual in the population, varie by norminal distribution
        mutate the bias and weights
        :return:
        """

        mutated_weights = []
        mutated_bias = []

        for w,b in zip(self.weights, self.bias):
            w_mutated = w + np.random.normal(0, self.std_weights, w.shape)
            b_mutated = b + np.random.normal(0, self.std_bias, b.shape)

            reset_mask_w = np.random.rand(*w.shape) < reset_prob
            reset_mask_b = np.random.rand(*b.shape) < reset_prob

            w_mutated[reset_mask_w] = np.random.randn(*w[reset_mask_w].shape)*self.scale_factor
            b_mutated[reset_mask_b] = np.random.randn(*b[reset_mask_b].shape)*self.scale_factor

            mutated_weights.append(w_mutated)
            mutated_bias.append(b_mutated)
        self.bias = mutated_bias
        self.weights = mutated_weights

    def get_params(self):
        print(f"Weights:{self.weights}")
        print(f"bias:{self.bias}")
        return

    def feedforward(self, a):
        """
        return output of the network if inputs
        :param a: input vector
        :return:
        """

        for b, w in zip(self.bias, self.weights):
            # print(f"a {a}")
            # print(f"w {w}")
            a = self.sigmoid(np.dot(w,a)+b)
        return a

    def sigmoid(self,z):
        """
        calculate sigmoid function
        :param z: vector or Numpy array
        :return:
        """
        z = np.clip(z,-500,500)
        return 1 / (1 + np.exp(-z))

    def evaluate(self, training_data):
        """
        Return the number of correct predictions. Seen as fitness for the GA
        NB: for classification problems only
        :param test_data:
        :return:
        """
        self.fitness = 0
        X_train,y_train = training_data
        y_pred = self.feedforward(X_train.T)
        y_pred = np.array(y_pred>=0.5,dtype= 'int')[0]
        tp = fp =fn =0
        for pred,truth in zip(y_pred,y_train):
            tp += (pred ==1 ) and (truth ==1)
            fp += (pred ==1) and (truth ==0)
            fn += (pred ==0) and (truth ==1)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        self.fitness =2*precision*recall/(precision+recall) if precision+recall > 0 else 0
        return 2*precision*recall/(precision+recall) if precision+recall > 0 else 0

    def plot_decision_boundary(self, training_data, resolution=0.01):
        """
        Plots the decision boundaries of a neural network classifier.

        :param neural_network: The neural network object with `feedforward` method.
        :param training_data: Tuple (X_train, y_train) with input data and labels.
        :param resolution: The resolution of the grid for plotting.
        """
        X_train, y_train = training_data

        # Extract input feature ranges
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

        # Create a grid of points
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, resolution),
            np.arange(y_min, y_max, resolution)
        )

        # Flatten the grid and evaluate the neural network
        grid_points = np.c_[xx.ravel(), yy.ravel()].T  # Shape (2, n_points)
        predictions = self.feedforward(
            grid_points)  # Neural network expects input of shape (n_features, n_samples)
        predictions = np.array(predictions >= 0.5, dtype='int')  # Convert to binary predictions

        # Reshape predictions to match the grid shape
        Z = predictions.reshape(xx.shape)

        # Plot decision boundaries
        plt.contourf(xx, yy, Z, alpha=0.6, cmap='coolwarm')
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap='coolwarm')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary")
        plt.show()
