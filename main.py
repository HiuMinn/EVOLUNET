import numpy as np
from models.evolution import Evolution
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv

def plot_dataset(data):
    X = data[0]
    y = data[1]
    # Plot linear classification
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')

    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    plt.show()

def read_dataset_csv(filename,header = None):
    X = []
    y = []
    with open(filename, 'r') as f:
        datasets = csv.reader(f, delimiter=',')
        if header:
            next(datasets,None)
        for row in datasets:
            X.append((float(row[0]),float(row[1])))
            y.append(int(float(row[2])))
    return np.array(X), np.array(y)

if __name__ == '__main__':

    # X,y = datasets.make_circles(n_samples=400,noise = 0.05, factor = 0.5)
    X,y = read_dataset_csv('spiral_data.csv',header = True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train,X_test,y_train, y_test = train_test_split(X_scaled,y,test_size=0.75)
    train_data = (X_train, y_train)
    test_data  =(X_test,y_test)
    plot_dataset(train_data)

    network_sizes = [2,7,10,4,1]
    #Initialize the population
    new_population = Evolution(network_sizes,100,1000)

    #Run the evolution
    best_network = new_population.generation_evolution(train_data,crossover=True)

    print(best_network.evaluate(test_data))
    new_population.plot_fitness()
    best_network.plot_decision_boundary(test_data)