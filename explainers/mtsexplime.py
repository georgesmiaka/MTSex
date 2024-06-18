
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from statsmodels.tsa.api import VAR


class MTSexpLIME:
    '''
    The LIME Explainer class for Multivariate Time Series.
    
    Parameters:
    model: model or prediction function of model takes 2D input (np.ndarray) and return 2D output.
    features_list_names: List of name of features.
    labels_name: List of the labels being predicted.
    loss: loss function for measuring prediction accuracy.
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
    
    def _transform_to_3d(self, data):
        return data.reshape((1, data.shape[0], data.shape[1]))

    def _transform_to_2d(self, data):
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    
    def _perturb_time_series(self, time_series, noise_sd=0.1, num_samples=500):
        """
        Generate multiple perturbed samples with random permutation and noise for a multivariate time series dataset.
        
        Parameters:
        - time_series: numpy array of shape (timesteps, features)
        - noise_sd: standard deviation of the Gaussian noise to be added
        - num_samples: number of perturbed samples to generate
        
        Returns:
        - noisy_time_series_samples: numpy array of shape (num_samples, timesteps, features)
        """
        timesteps, features = time_series.shape
        noisy_time_series_samples = np.zeros((num_samples, timesteps, features))
        
        for i in range(num_samples):
            for feature in range(features):
                permuted_timesteps = np.random.permutation(timesteps)
                permuted_data = time_series[permuted_timesteps, feature]
                noise = np.random.normal(0, noise_sd, timesteps)
                noisy_time_series_samples[i, :, feature] = permuted_data + noise
        
        return noisy_time_series_samples

    
    def _euclidean_distance(self, original, perturbed):
        """
        Calculates the Euclidean distance between each corresponding point in the original and perturbed trajectories.

        Parameters:
        - original (np.ndarray): Original trajectory data of shape (timesteps, features).
        - perturbed (np.ndarray): Perturbed trajectory data of shape (n_samples, timesteps, features).

        Returns:
        - distances (np.ndarray): Array of distances for each sample, shape (n_samples,)
        """
        squared_diff = np.sum((perturbed - original) ** 2, axis=2)
        distances = np.sqrt(np.mean(squared_diff, axis=1))
        return distances
    
    def _get_best_perturbed_samples(self, original, samples, limit):
        distances = self._euclidean_distance(original, samples)
        indices = np.argsort(distances)[:limit]
        return samples[indices], distances[indices]
    
    def _pred_perturbed_sample_with_blackbox(self, original, samples):
        original_pred = self.model(original)
        best_samples, best_distances = self._get_best_perturbed_samples(original, samples, 10)
        best_samples_pred = self.model(best_samples)
        
        mse_scores = [self.loss(original_pred, sample) for sample in best_samples_pred]
        return best_samples, best_samples_pred, best_distances, mse_scores
    
    def blackbox_evaluation(self, y):
        samples = self._perturb_time_series(y)
        best_samples, best_samples_pred, distance_scores, mse_scores = self._pred_perturbed_sample_with_blackbox(y, samples)
        return best_samples, best_samples_pred, distance_scores, mse_scores 
    
    def plot_blackbox_evaluation(self, distance_scores, mse_scores):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].plot(distance_scores, 'b', label='Distance Scores')
        axs[0].set_title('Black-box Sensitivity Analysis')
        axs[0].set_xlabel('Perturbed Samples')
        axs[0].legend()

        axs[1].plot(mse_scores, 'r', label='Loss Scores')
        axs[1].set_xlabel('Perturbed Samples')
        axs[1].legend()

        plt.tight_layout()
        plt.close(fig)
        return fig

    def _nparray_to_dataframe(self, y):
        return pd.DataFrame(y, columns=self.feature_list_names)
    
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
        results = model.fit(maxlags=4, trend='n')
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
    
    def _plot_average_feature_effect(self, mean_effects):
        """
        Plot the average feature effect and return the plot.
        
        Parameters:
        - mean_effects: The mean effects for each feature.
        
        Returns:
        - fig: The matplotlib Figure object.
        """
        values = mean_effects
        feature_names = self.feature_list_names

        # Determine bar colors based on the sign of the scaled Shapley values
        bar_colors = ['red' if x < 0 else 'blue' for x in values]

        # Create Figure and Axes objects
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, values, align='center', color=bar_colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Magnitude of the coefficients effect')
        ax.set_title('Variables Significance')
        
        plt.close(fig)  # Close the figure to prevent it from displaying immediately in some environments
        return fig


    def _evaluate_surogate_model(self, loss_score):
        """
        Plot the surrogate model evaluation and return the plot.
        
        Parameters:
        - loss_score: The loss scores for each perturbed sample.
        
        Returns:
        - fig: The matplotlib Figure object.
        """
        # Create Figure and Axes objects
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(loss_score, alpha=0.7)
        ax.set_xlabel('Perturbed Samples')
        ax.set_ylabel('Loss scores Evaluation')
        ax.set_title('Surrogate Model Sensitivity Analysis')
        ax.grid(True)
        
        plt.close(fig)  # Close the figure to prevent it from displaying immediately in some environments
        return fig

    def calculate_mean_effects_and_losses(self, best_samples, best_samples_pred):
        mean_effects_list = []
        loss_scores_list = []
        
        for sample, sample_pred in zip(best_samples, best_samples_pred):
            results, loss_score = self._surogate_model_var(sample, sample_pred)
            lagged_effects = self._extract_lagged_effects(results, self.labels_name)

            avg_effects = []
            for label in self.labels_name:
                label_effect = self._calculate_average_effects(lagged_effects[label])
                avg_effects.append(label_effect)
            
            mean_effects = self._average_across_labels(avg_effects)
            mean_effects_list.append(mean_effects)
            loss_scores_list.append(loss_score)
        
        return mean_effects_list, loss_scores_list
    
    def compute_cross_mean_effects(self, best_samples, best_samples_pred):
        mean_effects_list, loss_scores_list = self.calculate_mean_effects_and_losses(best_samples, best_samples_pred)
        
        # Compute the average effects across all samples
        cross_mean_effects = np.mean(mean_effects_list, axis=0)
        
        return cross_mean_effects, loss_scores_list
    
    def average_neighborhood_feature_effect(self, best_samples, best_samples_pred):
        cross_mean_effects, loss_scores_list = self.compute_cross_mean_effects(best_samples, best_samples_pred)
        
        # Plot the cross mean effect
        fig1 = self._plot_average_feature_effect(cross_mean_effects)
        
        # Evaluate and plot the surrogate model adaptability
        fig2 = self._evaluate_surogate_model(loss_scores_list)
        return fig1, fig2







