import math
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt


class MTSexpSHAP():
    """
    The MTSexp SHAP Explainer class.
    
    Parameters:
    model: Model or prediction function that takes 2D input (np.ndarray) and returns 2D output.
    features_list_names: List of feature names.
    labels_name: List of the labels being predicted.
    loss: Loss function for measuring prediction accuracy.
    """
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.labels_name = None
        self.loss = None

    def fit_exp(self, model, loss, feature_name_list, label_names):
        self.model = model
        self.loss = loss
        self.feature_names = feature_name_list
        self.labels_name = label_names
    
    @staticmethod
    def _is_3d(data: np.ndarray) -> bool:
        return data.ndim == 3

    @staticmethod
    def _is_2d(data: np.ndarray) -> bool:
        return data.ndim == 2

    @staticmethod
    def _transform_to_3d(data: np.ndarray) -> np.ndarray:
        return data.reshape((1, data.shape[0], data.shape[1]))  # (num_samples, sample_size, nu_features)

    @staticmethod
    def _transform_to_2d(data: np.ndarray) -> np.ndarray:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    @staticmethod
    def _perturb_time_series(time_series: np.ndarray, noise_sd: float = 0.1) -> np.ndarray:
        """
        Apply random permutation and add noise to a multivariate time series dataset.
        
        Parameters:
        - time_series: Numpy array of shape (timesteps, features).
        - noise_sd: Standard deviation of the Gaussian noise to be added.
        
        Returns:
        - Noisy time series with added noise.
        """
        timesteps, features = time_series.shape
        noisy_time_series = np.zeros_like(time_series)
        
        for feature in range(features):
            # Generate a random permutation of timesteps
            permuted_timesteps = np.random.permutation(timesteps)
            
            # Apply the permutation
            permuted_data = time_series[permuted_timesteps, feature]
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_sd, timesteps)
            noisy_time_series[:, feature] = permuted_data + noise
        
        return noisy_time_series
    
    def _baseline(self, y):
        """
        Generate baseline perturbed sample from the input data.

        Parameters:
        y (np.ndarray): Input data being explained (2D shape). 

        Returns:
        - perturbed sample.
        """
        pertubed_samples = self._perturb_time_series(y)

        return pertubed_samples
    
    @staticmethod
    def _compute_weight(total_features: int, subset_size: int) -> float:
        """
        Compute the weight for subsets of a given size in the context of Shapley values.
        
        Parameters:
        - total_features: Total number of features (|N|).
        - subset_size: Size of the subset including the feature (|S|).
        
        Returns:
        - The weight for the subset.
        """
        return (math.factorial(subset_size - 1) * math.factorial(total_features - subset_size)) / math.factorial(total_features)
    
    def _generate_coalition_matrix(self, feature_num: int) -> np.ndarray:
        """
        Generate the full coalition matrix.
        
        Parameters:
        - feature_num: Number of features.
        
        Returns:
        - Coalition matrix.
        """
        return np.array(list(product(range(2), repeat=feature_num)))
    
    def _filter_coalitions(self, mask: np.ndarray, exclude_feature: int = None) -> np.ndarray:
        """
        Filter the coalition matrix to remove all-zero, all-one, and specified feature coalitions.
        
        Parameters:
        - mask: Coalition matrix.
        - exclude_feature: Feature to exclude from the coalitions.
        
        Returns:
        - Filtered coalition matrix.
        """
        mask = mask[~np.all(mask == 0, axis=1)]
        mask = mask[~np.all(mask == 1, axis=1)]
        
        if exclude_feature is not None and 0 <= exclude_feature < mask.shape[1]:
            mask = mask[mask[:, exclude_feature] == 0]
        
        return mask
    
    def _sample_coalitions(self, mask: np.ndarray, sample_size: int) -> np.ndarray:
        """
        Sample a subset of coalitions if the total number exceeds the sample size.
        
        Parameters:
        - mask: Coalition matrix.
        - sample_size: Maximum number of coalitions to sample.
        
        Returns:
        - Sampled coalition matrix.
        """
        if mask.shape[0] > sample_size:
            return mask[np.random.choice(mask.shape[0], sample_size, replace=False)]
        return mask

    def _create_mask(self, feature_num: int, exclude_feature: int = None) -> np.ndarray:
        """
        Create the coalition mask matrix.
        
        Parameters:
        - feature_num: Number of features.
        - exclude_feature: Feature to exclude from the coalitions.
        
        Returns:
        - Mask matrix.
        """
        max_feature_num = 10
        sample_size = 2 ** max_feature_num

        mask = self._generate_coalition_matrix(feature_num)
        mask = self._filter_coalitions(mask, exclude_feature)
        if feature_num > max_feature_num:
            mask = self._sample_coalitions(mask, sample_size)
        
        return mask
    
    def _calculate_rmse(self, original_pred: np.ndarray, modified_pred: np.ndarray) -> float:
        """
        Calculate the RMSE between original and modified predictions.
        
        Parameters:
        - original_pred: Original predictions.
        - modified_pred: Modified predictions.
        
        Returns:
        - RMSE value.
        """
        return self.loss(original_pred, modified_pred)
    
    def _prepare_feature_masks(self, baseline_tab: np.ndarray, data: np.ndarray, item: np.ndarray, feature: int) -> tuple:
        """
        Prepare feature masks for the given coalition item.
        
        Parameters:
        - baseline_tab: Baseline perturbed data.
        - data: Original input data.
        - item: Coalition item.
        - feature: Feature to include in the masks.
        
        Returns:
        - Tuple of (without_feature, with_feature, number_of_feature_masked).
        """
        without_feature = baseline_tab.copy()
        with_feature = baseline_tab.copy()
        num_masked = 0
        
        for i, use_data in enumerate(item):
            if use_data:
                without_feature[:, i] = data[:, i]
                with_feature[:, i] = data[:, i]
                num_masked += 1
        
        with_feature[:, feature] = data[:, feature]
        num_masked += 1
        
        return without_feature, with_feature, num_masked
    
    def _compute_marginal_contribution(self, feature: int, feature_num: int, y: np.ndarray, perturbed_data: np.ndarray, mask: np.ndarray) -> list:
        """
        Compute the marginal contributions of a feature across all coalitions.
        
        Parameters:
        - feature: The feature being analyzed.
        - feature_num: The number of features in the input data.
        - y: Input data being explained.
        - perturbed_data: Perturbed data derived from y.
        - mask: Coalition mask matrix.
        
        Returns:
        - List of marginal contributions.
        """
        marginal_contributions = []
        data = y
        baseline_tab = perturbed_data
        
        # Compute baseline prediction and original prediction
        baseline_pred = self.model(baseline_tab)
        original_pred = self.model(data)
        
        # Compute marginal contribution for baseline {feature} - {}
        rmse = self._calculate_rmse(original_pred, baseline_pred)
        weight = self._compute_weight(feature_num, 1)
        marginal_contributions.append(weight * rmse)
        
        for item in mask:
            without_feature, with_feature, num_masked = self._prepare_feature_masks(baseline_tab, data, item, feature)
            weight = self._compute_weight(feature_num, num_masked)
            
            pred_without_feature = self.model(without_feature)
            pred_with_feature = self.model(with_feature)
            
            rmse_without_feature = self._calculate_rmse(original_pred, pred_without_feature)
            rmse_with_feature = self._calculate_rmse(original_pred, pred_with_feature)
            marginal_contributions.append(weight * (rmse_with_feature - rmse_without_feature))
        
        return marginal_contributions

    def _value_function(self, feature: int, feature_num: int, y: np.ndarray, perturbed_data: np.ndarray) -> float:
        """
        Compute and return the average contribution of a feature.
    
        Parameters:
        - feature: The feature being analyzed.
        - feature_num: The number of features in the input data.
        - y: Input data being explained.
        - perturbed_data: Perturbed data derived from y.

        Returns:
        - The average contribution of the feature.
        """
        mask = self._create_mask(feature_num, feature)
        marginal_contributions = self._compute_marginal_contribution(feature, feature_num, y, perturbed_data, mask)
        return np.mean(marginal_contributions)

    def shap_values(self, y: np.ndarray) -> tuple:
        """
        Compute and return the average contribution of all features.
    
        Parameters:
        - y: Input data being explained (timesteps, features) (2D shape).
        
        Returns:
        - Tuple of (average contribution of each feature, baseline perturbed sample).
        """
        perturbed_data = self._perturb_time_series(y)
        n_features = y.shape[1]
        shapley_values = [self._value_function(feature, n_features, y, perturbed_data) for feature in range(n_features)]
        return np.array(shapley_values), perturbed_data
    
    def plot_shap_values(self, s_values: np.ndarray):
        """
        Plot the features' average contribution with SHAP values and return the plot.
        """
        feature_names = self.feature_names
        bar_colors = ['red' if x < 0 else 'blue' for x in s_values]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, s_values, align='center', color=bar_colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Mean SHAP Value')
        ax.set_title('Feature Average Contribution with SHAP Values')
        
        plt.close(fig)  # Close the figure to prevent it from displaying immediately in some environments
        return fig