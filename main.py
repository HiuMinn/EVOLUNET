import numpy as np
import pandas as pd
from Bio.Pathway import Network
from network import EVOLUNET
from evolution import Evolution
import matplotlib.pyplot as plt
import time
from sklearn import datasets
import random

def plot_dataset(data):
    X = data[0]
    y = data[1]
    # Plot linear classification
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')

    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    plt.show()


if __name__ == '__main__':
    # data = datasets.make_gaussian_quantiles(n_samples=100,n_classes=2)
    data = datasets.make_blobs(n_samples=100, centers=2, cluster_std=.5)
    network_sizes = [2,3,2,1]
    plot_dataset(data)
    new_population = Evolution(network_sizes,50,1000)
    new_population.init_population()
    new_population.generation_evaluation(data)


