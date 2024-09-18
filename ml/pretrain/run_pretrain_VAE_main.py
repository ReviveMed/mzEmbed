'''
this is the main function to run the pretrain VAE model

Parameters:
    X_data_train: The training data.
    y_data_train: The training labels.
    X_data_val: The validation data.
    y_data_val: The validation labels.
    kwargs: The keyword arguments to pass to the model. 

Returns:    
    The trained model.  

'''

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import importlib
import random
import argparse
import pickle

import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import optuna

import optuna.visualization as vis
import plotly.io as pio

from optuna.visualization import plot_optimization_history, plot_param_importances
from jinja2 import Template

# importing my own functions

from models.models_VAE import VAE
from pretrain.train_pretrain_VAE import PretrainVAE, pretrain_vae, optimize_hyperparameters


import os
import pandas as pd
import plotly.io as pio
import optuna.visualization as vis
from jinja2 import Template

def save_optuna_study_results_as_html(study, combined_params, save_dir, file_name="optuna_report.html"):
    """
    Save the results of the Optuna study as an HTML file including history plot, hyperparameter importance,
    best trial details, and a DataTable with all trials and parameter ranges.

    Parameters:
        study: The Optuna study object.
        combined_params: Dictionary containing the parameter ranges and default kwargs.
        save_dir: Directory where the HTML file will be saved.
        file_name: The name of the HTML file (default is 'optuna_report.html').
    """
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Plot optimization history and importance
    history_plot = vis.plot_optimization_history(study)
    importance_plot = vis.plot_param_importances(study)

    # Save the plots as HTML components
    history_html = history_plot.to_html(include_plotlyjs='cdn')
    importance_html = importance_plot.to_html(include_plotlyjs='cdn')

    # Get the best trial
    best_trial = study.best_trial
    best_trial_value = best_trial.value
    best_trial_params = best_trial.params

    # Create a DataFrame of all trials and their parameters
    trials_data = []
    for trial in study.trials:
        trial_dict = trial.params.copy()  # Copy trial parameters
        trial_dict['Objective Value'] = trial.value  # Add the objective value
        trials_data.append(trial_dict)  # Append to list

    # Convert to a DataFrame
    trials_df = pd.DataFrame(trials_data)

    # Convert combined_params to a string for display at the beginning of the table
    param_ranges_str = "<ul>"
    for param, value in combined_params.items():
        param_ranges_str += f"<li><strong>{param}:</strong> {value}</li>"
    param_ranges_str += "</ul>"

    # Create an HTML template using Jinja2
    template = Template('''
    <html>
    <head>
        <title>Optuna Study Report</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.3/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.3/js/jquery.dataTables.js"></script>
        <style>
            body { font-family: Arial, sans-serif; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Optuna Study Report</h1>
        
        <h2>Best Trial</h2>
        <p><strong>Best Objective Value:</strong> {{ best_trial_value }}</p>
        <p><strong>Best Trial Parameters:</strong></p>
        <ul>
            {% for param, value in best_trial_params.items() %}
            <li><strong>{{ param }}:</strong> {{ value }}</li>
            {% endfor %}
        </ul>

        <h2>Tested Parameter Ranges</h2>
        {{ param_ranges_str | safe }}

        <h2>Optimization History</h2>
        {{ history_html | safe }}

        <h2>Hyperparameter Importance</h2>
        {{ importance_html | safe }}

        <h2>All Trials</h2>
        <table id="trials_table" class="display">
            <thead>
                <tr>
                    {% for col in trials_df.columns %}
                    <th>{{ col }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for _, row in trials_df.iterrows() %}
                <tr>
                    {% for val in row %}
                    <td>{{ val }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <script>
            $(document).ready(function() {
                $('#trials_table').DataTable({
                    "scrollX": true,  // Enable horizontal scrolling
                    "paging": true,   // Enable pagination
                    "searching": true, // Enable search
                    "ordering": true,  // Enable column sorting
                    "info": true      // Show table information
                });
            });
        </script>

    </body>
    </html>
    ''')

    # Render the template
    html_content = template.render(
        best_trial_value=best_trial_value,
        best_trial_params=best_trial_params,
        param_ranges_str=param_ranges_str,
        history_html=history_html,
        importance_html=importance_html,
        trials_df=trials_df
    )

    # Save the rendered HTML to a file
    html_file_path = os.path.join(save_dir, file_name)
    with open(html_file_path, "w") as f:
        f.write(html_content)

    print(f"Optuna study results saved as HTML at {html_file_path}")



def get_pretrain_input_data(data_location):

    # #defining the input datasets
    # pretrain_X_all=f'{data_location}/X_Pretrain_All.csv'
    # pretrain_y_all=f'{data_location}/y_Pretrain_All.csv'

    pretrain_X_train=f'{data_location}/X_Pretrain_Discovery_Train.csv'
    pretrain_y_train=f'{data_location}/y_Pretrain_Discovery_Train.csv'

    pretrain_X_val=f'{data_location}/X_Pretrain_Discovery_Val.csv'
    pretrain_y_val=f'{data_location}/y_Pretrain_Discovery_Val.csv'

    pretrain_X_test=f'{data_location}/X_Pretrain_Test.csv'
    pretrain_y_test=f'{data_location}/y_Pretrain_Test.csv'

    #loading the data
    # X_data_all = pd.read_csv(pretrain_X_all, index_col=0)
    # y_data_all = pd.read_csv(pretrain_y_all, index_col=0)

    X_data_train = pd.read_csv(pretrain_X_train, index_col=0)
    y_data_train = pd.read_csv(pretrain_y_train, index_col=0)

    X_data_val = pd.read_csv(pretrain_X_val, index_col=0)
    y_data_val = pd.read_csv(pretrain_y_val, index_col=0)

    X_data_test = pd.read_csv(pretrain_X_test, index_col=0)
    y_data_test = pd.read_csv(pretrain_y_test, index_col=0)

    #returning the data
    return(X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)


def parse_range_or_single(values, is_int=False):
    """
    Custom argparse type to parse either a single value or a range from a list.
    If `is_int` is True, parses integers; otherwise, parses floats.
    
    Example inputs: 
        ["256"] -> Outputs: 256 (int)
        ["256", "512", "32"] -> Outputs: (256, 512, 32) (int)
    """
    # If it's a list but contains only one element, treat it as a single value
    if len(values) == 1:
        if is_int:
            return int(values[0])  # Single integer value
        else:
            return float(values[0])  # Single float value
    
    # Otherwise, treat it as a range
    if is_int:
        return tuple(map(int, values))  # Convert to tuple of integers
    else:
        return tuple(map(float, values))  # Convert to tuple of floats





def main():
    
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Run VAE pretraining with hyperparameter optimization.')

    # Add arguments for the input directory and save directory
    parser.add_argument('--input_data_location', type=str, required=True, 
                        help='Location of the input data directory')
    parser.add_argument('--pretrain_save_dir', type=str, required=True, 
                        help='Directory to save the pre-trained models')

    # Add arguments for the hyperparameters
    parser.add_argument('--latent_size', nargs='*', default=[256],
                        help='Latent size: Either a single value or "min max step" for range (integer)')
    parser.add_argument('--num_hidden_layers', nargs='*', default=[2],
                        help='Number of hidden layers (either fixed or a range, integer)')
    parser.add_argument('--dropout_rate', nargs='*', default=[0.1], 
                        help='Dropout rate: Either a single value or "min max step" for range (float)')
    parser.add_argument('--learning_rate', nargs='*', default=["1e-5", "1e-2"], 
                        help='Learning rate range: Either a single value or "min max" for range (float)')
    parser.add_argument('--weight_decay', nargs='*', default=["1e-6", "1e-4"],
                        help='Weight decay range: Either a single value or "min max" for range (float)')
    parser.add_argument('--batch_size', nargs='*', default=[64],
                        help='Batch size (either fixed or a range, integer)')
    parser.add_argument('--patience', nargs='*', default=[10, 30, 5], 
                        help='Patience: Either a single value or "min max step" for range (integer), 0 means no early stopping')
    parser.add_argument('--num_epochs', nargs='*', default=[100], 
                        help='Number of epochs (either fixed or a range, integer)')
    parser.add_argument('--trial_name', type=str, default='vae_hyperparam_optimization_test', 
                        help='The name of this trial for Optuna study')
    parser.add_argument('--n_trials', type=int, default=20, 
                        help='Number of trials for hyperparameter optimization')

    print('Arguments are parsed')

    # Parse the arguments
    args = parser.parse_args()

    # Use parsed arguments and convert them using parse_range_or_single
    combined_params = {
        'latent_size': parse_range_or_single(args.latent_size, is_int=True),
        'num_hidden_layers': parse_range_or_single(args.num_hidden_layers, is_int=True),
        'dropout_rate': parse_range_or_single(args.dropout_rate, is_int=False),
        'learning_rate': parse_range_or_single(args.learning_rate, is_int=False),
        'weight_decay': parse_range_or_single(args.weight_decay, is_int=False),
        'batch_size': parse_range_or_single(args.batch_size, is_int=True),
        'patience': parse_range_or_single(args.patience, is_int=True),
        'num_epochs': parse_range_or_single(args.num_epochs, is_int=True)
    }

    # Use parsed arguments for other params
    input_data_location = args.input_data_location
    pretrain_save_dir = args.pretrain_save_dir
    trial_name = args.trial_name
    n_trials = args.n_trials

    # Print to check that arguments are parsed correctly
    print(f"Parsed params: {combined_params}")

    # Making sure the directory exists
    os.makedirs(pretrain_save_dir, exist_ok=True)


    print ('Getting the input data')
    # Getting the input data
    (X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test) = get_pretrain_input_data(input_data_location)
    
    print (f'Training data shape: {X_data_train.shape}')
    print (f'Validation data shape: {X_data_val.shape}')
    print (f'Test data shape: {X_data_test.shape}')

    
    # Run the hyperparameter optimization
    print ('Optimizing the hyperparameters')
    best_trial, study = optimize_hyperparameters(X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, combined_params, trial_name, pretrain_save_dir, n_trials=n_trials)

    print(f"Best hyperparameters: {best_trial.params}")

    # Assuming 'study' is the Optuna study object and 'combined_params' contains the parameter ranges
    save_optuna_study_results_as_html(study, combined_params, save_dir=f'{pretrain_save_dir}/{trial_name}', file_name=f'{trial_name}_optuna_report.html')

    with open(f"{pretrain_save_dir}/{trial_name}/{trial_name}_optuna_report.pkl", "wb") as f:
            pickle.dump(study, f)
            
    #save the results
    print ('All done and results are saved')



if __name__ == "__main__":
    main()