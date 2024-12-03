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
        if bias is None or weights is None:
            self.new_param()
        else:
            self.bias = bias
            self.weights = weights

    def new_param(self):
        """
        initiate a new random parameters of the network
        :param self:
        :return:
        """
        self.bias = [np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.size[:-1], self.size[1:])]
        # self.bias = [np.zeros([y,1]) for y in self.size[1:]]
        # self.weights = [np.zeros([y, x]) for x, y in zip(self.size[:-1], self.size[1:])]
        return None

    def mutate_param (self,std_weights = 1, std_bias =1):
        """
        regarding a neural network as a individual in the population, varie by norminal distribution
        mutate the bias and weights
        :return:
        """
        for i in range(len(self.weights)):
            self.weights[i] += np.random.normal(0,std_weights,size = self.weights[i].shape)
        for i in range(len(self.bias)):
            self.bias[i] += np.random.normal(0,std_bias,size = self.bias[i].shape)


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
        return 1 / (1 + np.exp(-z))

    def evaluate(self, training_data):
        """
        Return the number of correct predictions. Seen as fitness for the GA
        NB: for classification problems only
        :param test_data:
        :return:
        """
        # fitness = 0
        X_train,y_train = training_data
        y_pred = self.feedforward(X_train.T)
        # print(np.array(y_pred))
        # print(y_train)
        cost = 1/len(X_train)*np.sum(y_train*np.log(y_pred)+(1-y_train)*np.log(1-y_pred))
        return cost

    def plot_decision_boundary(self,train_X,train_y):
        x_min, x_max = train_X[:, 0].min(), train_X[:, 0].max()
        y_min, y_max = train_X[:, 1].min(), train_X[:, 1].max()

        xx ,yy = np.meshgrid(np.linspace(x_min, x_max, 500)
                             ,np.linspace(y_min, y_max, 500))

        grid = np.c_[xx.ravel(), yy.ravel()]

        prediction = []
        for point in grid:
            point = point.reshape(-1,1)
            output = np.argmax(self.feedforward(point))
            prediction.append(output)
        print(prediction)
        prediction = np.array(prediction).reshape(xx.shape)
        plt.contourf(xx, yy, prediction, cmap = plt.cm.coolwarm, alpha = 0.6)
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=plt.cm.coolwarm, edgecolor='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary of the Neural Network')
        plt.show()


if __name__ == '__main__':
    network  = EVOLUNET(size=[2,3,2,1])
    network.new_param()

    print("old weights:",network.weights)
    print("old bias",network.bias)

    network.mutate_param()
    print("new weights: ",network.weights)
    print("new bias: ", network.bias)