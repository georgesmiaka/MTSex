
import math
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random

class LIME():
    '''
    The LIME Explainer class.
    
    Parameters:
    model: model or prediction function of model takes 2D input (np.ndarray) and return 2D output.
    features_list_names: List of name of features.
    labels_name: List of the labels being predicted.
    '''
    def __init__(self):
        self.model = None
        self.feature_list_names = None
        self.labels_name = None

    def fit_exp(self, model, features_list_names, labels_name):
        self.model = model
        self.feature_names = features_list_names
        self.class_names = labels_name
    
    def _is_3d(self, data):
        y = False
        if data.ndim == 3:
            y = True
        return y

    def _is_2d(self, data):
        y = False
        if data.ndim == 2:
            y = True
        return y

    def _transform_to_3d(self, data):
        y = data.reshape((1, data.shape[0], data.shape[1])) #(num_samples, sample_size, nu_features)
        return y

    def _transform_to_2d(self, data):
        y = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
        return y
    
    def _perturb_time_series_lime(self, matrix, n_samples=500):
        """
        Generates multiple perturbed samples around  the original sequence.

        Parameters:
            matrix (np.ndarray): The original sequence of lime (timesteps, features).
        
        Returns:
            np.ndarray: Multiple perturbed samples of default shape (500, timesteps, features).
        """
        original_trajectory = np.array(matrix)
        timesteps, features = original_trajectory.shape
        # Initialize perturbation matrix
        perturbed_samples = np.zeros((n_samples, timesteps, features))

        for sample in range(n_samples):
            random_choice = np.random.choice([-2.0, 2.0])
            random_integer = random.randint(1, 100)
            # Iterate over each feature
            for i in range(original_trajectory.shape[1]):
                perturbed_samples[sample, :, i] = original_trajectory[:, i] + ((random_integer*random_choice))
        
        return perturbed_samples
    
    def _euclidean_distance(original, perturbed):
        """
        Calculates the Euclidean distance between each corresponding point in the original and perturbed trajectories.

        Parameters:
        - original (np.ndarray): Original trajectory data of shape (timesteps, features).
        - perturbed (np.ndarray): Perturbed trajectory data of shape (n_samples, timesteps, features).

        Returns:
        - distances (np.ndarray): Array of distances for each sample, shape (n_samples,)
        """
        # Subtract the original trajectory from each perturbed sample, square the results, sum across features, and take the square root
        squared_diff = np.sum((perturbed - original)**2, axis=2)
        distances = np.sqrt(np.mean(squared_diff, axis=1))
        return distances
    
    ### Complete the implementation