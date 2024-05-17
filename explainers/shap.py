import math
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt


class SHAP():
    '''
    The SHAP Explainer class.
    
    Parameters:
    model: model or prediction function of model takes 2D input (np.ndarray) and return 2D output.
    features_list_names: List of name of features.
    labels_name: List of the labels being predicted.
    loss: loss function for measuring the prediction accurancy
    '''
    def __init__(self):
        self.model = None
        self.feature_list_names = None
        self.labels_name = None
        self.loss = None

    def fit_exp(self, model, loss, features_list_names, labels_name):
        self.model = model
        self.loss = loss
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

    def _perturb_time_series(self, matrix):
        """
        Generates a signle perturbed sample that is significantly different from the original sequence.
        The algorithm starts by computing the correlation matrix of the input data matrix using Pearson 
        correlation coefficient. Then, it iterates over each feature in the matrix and identifies 
        highly correlated features (correlation coefficient between 0.7 and 1.0). For each highly 
        correlated feature, it computes a perturbation value based on the correlation grade between 
        the features. However, for features that are not highly correlated with any other feature, 
        noise perturbation is added to introduce variation in their values.
        
        Parameters:
            matrix (np.ndarray): The original sequence of shape (timesteps, features).
        
        Returns:
            np.ndarray: A single perturbed sample of the same shape as the original sequence.
        """
        # Convert the 2D array to a DataFrame
        df = pd.DataFrame(matrix)
        #compute the correlation DataFrame
        corr_DataFrame = df.corr()
        # Convert DataFrame to NumPy array
        corr_matrix = corr_DataFrame.values
        # Replace NaN values with 0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0)
        
        # Initialize perturbation matrix
        perturbed_matrix = np.copy(matrix)
        
        # Iterate over each feature
        for i in range(matrix.shape[1]):
            # Find indices of features with high correlation with feature i
            correlated_features = np.where(np.abs(corr_matrix[i]) > 0.7)[0]
            
            # Compute perturbation values based on correlation grade
            for j in correlated_features:
                if i != j:
                    perturbation_factor = corr_matrix[i, j]
                    perturbed_matrix[:, j] += perturbation_factor * perturbed_matrix[:, i]
        
        # Add noise perturbation for features with low correlation
        for i in range(matrix.shape[1]):
            if np.abs(corr_matrix[i]).max() <= 0.7:
                perturbed_matrix[:, i] *= (100 + (np.mean(perturbed_matrix[:, i]) * ((matrix.shape[1])/2) * (np.std(perturbed_matrix[:, i]))))
        
        return perturbed_matrix
    
    def _baseline(self, y):
        """
        Generate baseline pertubated sample from the input data

        Parameters:
        y (np.ndarray): Input data being explained (2D shape). Data used to create pertubated sample. 

        Returns:
        - perturbed sample.
        """
        pertubed_samples = self._perturb_time_series(y)

        return pertubed_samples
    
    def _compute_weight(self, total_features, subset_size):
        """
        Computes the weight for subsets of a given size in the context of Shapley values.
        
        Parameters:
        - total_features: Total number of features (|N|).
        - subset_size: Size of the subset including the feature (|S|).
        
        Returns:
        - The weight for the subset.
        """
        return (math.factorial(subset_size - 1) * math.factorial(total_features - subset_size)) / math.factorial(total_features)

    def _create_mask(self, feature_num):
        """
        Generate and return the coalition matrix of size 2^feature_num
        """
        if(feature_num>12):
            feature_num = 11
        mask = np.array(list(product(range(2), repeat=feature_num)))
        mask = mask[~np.all(mask == 0, axis=1)]
        mask = mask[~np.all(mask == 1, axis=1)]
    
        return mask

    def _value_function(self, feature, feature_num, y, pertubed_data, model):
        """
        The function that computes and returns the average contribution of a feature.
    
        Parameters:
        feature: The feature being analyzed.
        feature_num: The number of features in the input data
        y (np.ndarray): Input data being explained (2D shape).
        pertubed_data: Perturbed data derived from y
        model: model or prediction function of model takes 2D input and return 2D output.

        Returns:
        - The average contribution of "feature".
        """
        # create coalition matrix for masking
        mask=self._create_mask(feature_num)

        # configuration
        marginal_contribution = []
        average_contribution = []
        data = y

        # baseline or pertubed data
        baseline_tab = pertubed_data
        
        # compute baseline prediction
        #baseline_pred = model.predict_2Dto3D(baseline_tab)
        baseline_pred = model(baseline_tab)
        # compute y_true
        original_pred = model(data)

        # compute RMSEvalue original_pred vs baseline_pres
        rmse = self.loss(original_pred, baseline_pred)

        # compute marginal contribution for baseline {feature} - {}
        weight = self._compute_weight(feature_num, 1)

        pred = weight*(rmse)
        marginal_contribution.append(pred)

        # compute marginal contribution for the rest of the combinations
        for item in mask:
            # Initialize arrays as copies of data and baseline_tab
            without_feature = baseline_tab.copy()
            with_feature = baseline_tab.copy()
            number_of_feature_masked = 0
            
            # Iterate over each element in the mask
            for i, use_data in enumerate(item):
                if use_data:
                    without_feature[:, i] = data[:, i]
                    with_feature[:, i] = data[:, i]
                    number_of_feature_masked = number_of_feature_masked + 1

            # Include the feature
            with_feature[:, feature] = data[:, feature]
            number_of_feature_masked = number_of_feature_masked + 1
            weight = self._compute_weight(feature_num, number_of_feature_masked)
        

            # compute marginal contribution
            pred_without_feature = model(without_feature)
            pred_with_feature = model(with_feature)

            # compute rmse without_feature vs original_pred, rmse with_feature vs original_pred
            rmse_without_feature = self.loss(original_pred, pred_without_feature)
            rmse_with_feature = self.loss(original_pred, pred_with_feature)
            # compute the marginal contribution of the feature in the combination
            pred = weight*(rmse_with_feature - rmse_without_feature)
            marginal_contribution.append(pred)
        
        # compute the average contribution
        average_contribution = np.mean(marginal_contribution)
        return average_contribution

    def shap_values(self, y):
        """
        The function that computes and returns the average contribution of all features.
    
        Parameters:
        y (np.ndarray): Input data being explained (timesteps, features) (2D shape).
        
        Returns:
        - The average contribution of each features (np.arrays)".
        - The baseline perturbed sample
        """
        # step 1: configuration
        pertubed_data = self._perturb_time_series(y)
        n_features = y.shape[1]
        shapley_values = []
        
        # step 2: Compute the shapeley values for each feature
        for feature in range(n_features):
            s_value = self._value_function(feature, n_features, y, pertubed_data, self.model)
            # Load values
            shapley_values.append(s_value)
        return shapley_values, pertubed_data
    
    def plot_shap_values(self, s_values):
        """
        Plot the features average contribution with SHAP value
        """
        shap_values = s_values
        feature_names = self.feature_names

        # Determine bar colors based on the sign of the mean Shapley values
        bar_colors = ['red' if x < 0 else 'blue' for x in shap_values]

        # Plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        plt.barh(y_pos, shap_values, align='center', color=bar_colors)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Mean SHAP Value')
        plt.title('Feature Average Contribution with SHAP Values')
        plt.show()