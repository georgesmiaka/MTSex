import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from explainers.mtsexpshap import MTSexpSHAP
from explainers.mtsexplime import MTSexpLIME

# Function to load the prediction function (model)
def load_prediction_function(model_path):
    # Assume the model is saved as a .npy file
    return np.load(model_path, allow_pickle=True).item()

# Function to load the input data
def load_input_data(data_path):
    return np.load(data_path)

# Function to create output directory
def create_output_dir(base_dir="explainers/out"):
    # Check if the base directory exists, create if it doesn't
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create the subdirectory with the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"ex_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

# Function to save the plot
def save_plot(fig, output_dir, plot_name):
    plot_path = os.path.join(output_dir, f"{plot_name}.png")
    fig.savefig(plot_path)

############# ------->
# Remove this! It is Only for test!
# Simulate the time series input data
def simulate_time_series(timesteps=10, features=4):
    return np.random.rand(timesteps, features)

# Define a simple prediction model that averages the features over the past 4 timesteps
def f_pred(time_series):
    if len(time_series) < 4:
        raise ValueError("Time series data should have at least 4 timesteps.")
    
    # Take the average of the last 4 timesteps for each feature
    return np.mean(time_series[-4:], axis=0)

# Define the mean squared error (MSE) loss function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate simulated time series data
y = simulate_time_series(timesteps=10, features=4)
############## ----> Remove

# Main function
def main():
    # Define the paths to the model and data
    #model_path = 'path/to/your/model.npy'  # Update this with your model path
    #data_path = 'path/to/your/data.npy'    # Update this with your data path
    
    # Load the prediction function and input data
    #f_pred = load_prediction_function(model_path)
    #y = load_input_data(data_path)
    
    # Define the loss function (assuming MSE for this example)
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    # Define the list of features and labels
    feature_name_list = ['feature1', 'feature2', 'feature3', 'feature4']
    label_names = ['feature2']
    
    # Create the output directory
    output_dir = create_output_dir()
    
    # Initialize the MTSexpSHAP explainer
    shap_exp = MTSexpSHAP()
    shap_exp.fit_exp(model=f_pred, loss=mse, feature_name_list=feature_name_list, label_names=label_names)
    
    # Compute MTSexpSHAP values
    shap_values, perturbed_data = shap_exp.shap_values(y)
    
    # Plot MTSexpSHAP values and save the plot
    fig_shap = shap_exp.plot_shap_values(shap_values)
    save_plot(fig_shap, output_dir, "MTSexpSHAP")
    
    # Initialize the MTSexpLIME explainer
    lime_exp = MTSexpLIME()
    lime_exp.fit_exp(model=f_pred, loss=mse, features_list_names=feature_name_list, labels_name=label_names)
    
    # Perform MTSexpLIME evaluation
    best_samples, best_samples_pred, distance_scores, mse_scores = lime_exp.blackbox_evaluation(y)
    
    # Plot MTSexpLIME evaluations and save the plots
    fig_lime_ev = lime_exp.plot_blackbox_evaluation(distance_scores, mse_scores)
    save_plot(fig_lime_ev, output_dir, "MTSexpLIME_blackbox_evaluation")
    

    fig_lime_ev_surogate, fig_lime_effects = lime_exp.average_neighborhood_feature_effect(best_samples, best_samples_pred)
    save_plot(fig_lime_effects, output_dir, "MTSexpLIME_evaluate_surogate_model")
    save_plot(fig_lime_ev_surogate, output_dir, "MTSexpLIME_average_neighborhood_feature_effect")

if __name__ == "__main__":
    main()
