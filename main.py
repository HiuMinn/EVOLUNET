import numpy as np
import pandas as pd
from network import EVOLUNET


def prepare_data(data,training = 0.8):
    training_index = int(len(data)*training)
    training_x = data[:training_index]
    training_y = data[training_index:]
    test_x = data[training_index:]
    test_y = data[training_index:]
    return training_x,training_y,test_x,test_y

def generate_data_gausien():

    # Parameters for the first class (Class 0)
    mean_class_0 = [2, 2]  # Mean of the Gaussian distribution for class 0
    cov_class_0 = [[1, 0.5], [0.5, 1]]  # Covariance matrix for class 0

    # Parameters for the second class (Class 1)
    mean_class_1 = [7, 7]  # Mean of the Gaussian distribution for class 1
    cov_class_1 = [[1, 0.2], [0.2, 1]]  # Covariance matrix for class 1

    # Number of samples per class
    n_samples = 1000

    # Generate data for Class 0
    class_0_data = np.random.multivariate_normal(mean_class_0, cov_class_0, n_samples)
    class_0_labels = np.zeros((n_samples, 1))  # Class label 0

    # Generate data for Class 1
    class_1_data = np.random.multivariate_normal(mean_class_1, cov_class_1, n_samples)
    class_1_labels = np.ones((n_samples, 1))  # Class label 1

    # Combine the data and labels
    data = np.vstack((class_0_data, class_1_data))
    labels = np.vstack((class_0_labels, class_1_labels))

    # Create a DataFrame for easier handling
    df = pd.DataFrame(np.hstack((data, labels)), columns=['Feature_1', 'Feature_2', 'Label'])

    # Save the dataset to a CSV file
    df.to_csv('./bin/gaussian_classification_dataset.csv', index=False)

    print("Classification dataset generated and saved as 'gaussian_classification_dataset.csv'")
