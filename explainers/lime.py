
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from statsmodels.tsa.api import VAR

class LIME():
    '''
    The LIME Explainer class.
    
    Parameters:
    model: model or prediction function of model takes 2D input (np.ndarray) and return 2D output.
    features_list_names: List of name of features.
    labels_name: List of the labels being predicted.
    loss: loss function for measuring prediction accurancy
    '''
    def __init__(self):
        self.model = None
        self.feature_list_names = None
        self.labels_name = None
        self.loss = None

    def fit_exp(self, model, loss, features_list_names, labels_name):
        self.model = model
        self.loss = loss
        self.feature_list_names = features_list_names
        self.labels_name = labels_name
    
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
    
    def _is_1d(self, data):
        y = False
        if data.ndim == 1:
            y = True
        return y

    def _transform_to_3d(self, data):
        y = data.reshape((1, data.shape[0], data.shape[1])) 
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
    
    def _euclidean_distance(self, original, perturbed):
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
    
    def _get_best_perturbed_samples(self, original, samples, limit):
        # Compute the euclidean distance (original, perturbed)
        distances = self._euclidean_distance(original, samples)

        # Get indices of the "limit" samples with the smallest distances
        indices = np.argsort(distances)[:limit]

        # Select the "limit" best samples and their distances
        best_samples = samples[indices]
        best_distances = distances[indices]

        # return best_samples and their distances
        return best_samples, best_distances
    
    def _pred_perturbed_sample_with_blackbox(self, original, samples):

        # compute original prediction
        original_pred = self.model(original)
        
        # Compute best samples predictions
        best_samples, best_distances = self._get_best_perturbed_samples(original, samples, 10)
        best_samples_pred = self.model(best_samples)
        
        mse_scores = []
        # Evaluations original predictions vs best_samples_pred
        for sample in best_samples_pred:
            mse = self.loss(original_pred, sample)
            mse_scores.append(mse)

        return best_samples, best_samples_pred, best_distances, mse_scores
    
    def blackbox_evaluation(self, y):
        # create perturbed samples around y
        samples = self._perturb_time_series_lime(y)
        # Get best perturbed samples, their predictions, their proximity score, and their mse scores
        best_samples, best_samples_pred, distance_scores, mse_scores = self._pred_perturbed_sample_with_blackbox(y, samples)
        return best_samples, best_samples_pred, distance_scores, mse_scores 
    
    def plot_blackbox_evaluation(self, distance_scores, mse_scores):
        # Create a figure with two subplots, 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot on the first subplot
        axs[0].plot(distance_scores, 'b', label='Loss Scores')
        axs[0].set_title('Black-box Sensitivity Analysis')
        axs[0].set_xlabel('Perturbed Samples')
        #axs[0].set_ylabel('Distance-Scores Evaluatio')
        axs[0].legend()

        # Plot on the second subplot
        axs[1].plot(mse_scores, 'r', label='Distance Scores')
        axs[0].set_xlabel('Perturbed Samples ')
        #axs[0].set_ylabel('MSE-Scores Evaluation')
        axs[1].legend()

        # Adjust layout for better fit and visibility
        plt.tight_layout()
        plt.show()


    
    def _nparray_to_dataframe(self, y):
        x = pd.DataFrame(y, columns=self.feature_list_names)
        return x
    
    def _surogate_model_var(self, sample, y_true):
        """
        Fits permuted samples in the Vector Auto Regression (VAR) model

        Parameters:
            sample (np.array): single perturbed sample of shape (timesteps, features)
            y_true (np.array): predictions for the perturbed sample using the black-box
        
        Returns:
            - Effect matrix of the sample
            - np.ndarray: loss score L(y_true, sample)
        """
        # Get the length of the forecasting horizon
        lag_nr = len(y_true) 
        # Transform sample (np.array) into dataframe
        df = self._nparray_to_dataframe(sample)
        # Init the VAR model
        model = VAR(df)
        # Fit the model
        results = model.fit(maxlags=14, trend='n')
        lag_order = results.k_ar
        fcst = results.forecast(df.values[-lag_order:], lag_nr)
        model_accuracy = self.loss(y_true, fcst[:, :len(self.labels_name)])
        return results.params, model_accuracy
    
    def _extract_lagged_effects(self, dataframe, labels):
        """
        Extracts matrices of lagged variable effects for specified labels from a VAR model's coefficient DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing lagged coefficients.
        - labels (list of str): List of variable labels to extract lagged effects for.

        Returns:
        - dict of pd.DataFrame: A dictionary where each key is a label and the value is a DataFrame containing the coefficients
                                for all lags of that label.
        """
        results = {}
        # Loop through each label and collect the corresponding lagged effects
        for label in labels:
            # Filter rows for each label across all specified lags
            # This assumes the DataFrame's index is properly named with L1.*, L2.*, ..., L12.*
            lag_pattern = f'L[0-9]+.{label}'
            filtered_df = dataframe.filter(regex=lag_pattern, axis=0)
            results[label] = filtered_df

        return results
    
    def _calculate_average_effects(self, lagged_df):
        """
        Calculates the average effect of each variable on a specific label across all specified lags.

        Parameters:
        - lagged_df (pd.DataFrame): DataFrame containing the lagged coefficients for a specific label.

        Returns:
        - np.ndarray: An array containing the average effects of each variable on the label.
        """
        # Calculate the mean of each column in the DataFrame
        mean_effects = lagged_df.mean()
        
        return mean_effects.values  # Convert the Pandas Series to a NumPy array
    
    def _average_across_labels(self, effect_arrays):
        # Assuming all arrays are aligned and have the same length
        mean_effects = np.mean(effect_arrays, axis=0)
        return mean_effects
    
    def _plot_average_feature_effect(self, mean_scores):
        values = mean_scores
        feature_names = self.feature_list_names

        # Determine bar colors based on the sign of the scaled Shapley values
        bar_colors = ['red' if x < 0 else 'blue' for x in values]

        # Plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        plt.barh(y_pos, values, align='center', color=bar_colors)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Magnitude of the coefficients effect')
        plt.title('variables significance')
        plt.show()

    def _evaluate_surogate_model(self, loss_score):
        # Plotting for visual analysis
        plt.figure(figsize=(8, 6))
        plt.plot(loss_score, alpha=0.7)
        plt.xlabel('Pertubated Samples')
        plt.ylabel('Loss scores Evaluation')
        plt.title('Surogate Model Sensitivity Analysis')
        plt.grid(True)
        plt.show()

    def average_feature_effect(self, best_samples, best_samples_pred, one_sample_only=True):
        
        # Average effects array
        avg_effects = []
        # loss scores
        loss_s = []
        # if one_sample_only = True show result for the 1st sample only
        for item in range(len(best_samples)):
            # get result from surogate mode
            results, loss_scores = self._surogate_model_var(best_samples[item], best_samples_pred[item])
            # Extracts matrices of lagged variable effects for specified labels from a VAR model's coefficient DataFrame.
            lagged_effects = self._extract_lagged_effects(results, self.labels_name)

            for label in self.labels_name:
                label_effect = self._calculate_average_effects(lagged_effects[label])
                avg_effects.append(label_effect)
            # Calculate the average acrros labels
            mean_effects = self._average_across_labels(avg_effects)
            self._plot_average_feature_effect(mean_effects)
        # plot surogate model adaptability
        for item in range(len(best_samples)):
            # get result from surogate mode
            results, loss_scores = self._surogate_model_var(best_samples[item], best_samples_pred[item])
            loss_s.append(loss_scores)
        self._evaluate_surogate_model(loss_s)




