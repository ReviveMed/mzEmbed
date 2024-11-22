'''
retraining fine-tune VAE models

'''
import os
import shutil
import argparse
import copy
import pandas as pd
import importlib
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import optuna

from collections.abc import Iterable
from itertools import product
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

import optuna.visualization as vis
import plotly.io as pio

from optuna.visualization import plot_optimization_history, plot_param_importances
from jinja2 import Template


#importing fundtion to get encoder info and perfrom tasks 
from models.models_VAE import VAE

from finetune.get_finetune_encoder import  get_finetune_input_data

from finetune.freez_encoder_latent_avg_COX import fine_tune_cox_model, FineTuneCoxModel
from finetune.best_finetune_model_test_eval_cox import best_finetune_model_test_eval_cox
from finetune.freez_encoder_latent_avg import fine_tune_model, FineTuneModel
from finetune.best_finetune_model_test_eval import evaluate_model_main




def get_finetune_VAE_TL_noTL(finetune_save_dir, pretrain_name, pretrain_trial_id):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #path to pre-train and fine-tune models
    models_path=f'{finetune_save_dir}/{pretrain_name}/trial_{pretrain_trial_id}'

    #finetune models files
    finetune_VAE_TL_file= f'{models_path}/finetune_VAE_TL_True_best_model_state.pt'
    finetune_VAE_TL=torch.load(finetune_VAE_TL_file, map_location=device)

    finetune_VAE_noTL_file= f'{models_path}/finetune_VAE_TL_False_best_model_state.pt'
    finetune_VAE_noTL=torch.load(finetune_VAE_noTL_file, map_location=device)


    return finetune_VAE_TL, finetune_VAE_noTL



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




def objective(trial, finetune_VAE_TL, finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, task, task_type, num_classes, task_event, seed, model_path, combined_params, patience=0, latent_passes= 20, **kwargs):
    

    global best_model, best_auc

    best_model_TL = None  # Variable to store the best TL model
    best_model_noTL = None  # Variable to store the best noTL model
    best_val_metric = -float('inf')  # Initialize to a very low value to keep track of the best AUC
    
    # Hyperparameters to optimize specific to retraining the model
    add_post_latent_layers = trial.suggest_categorical('add_post_latent_layers', combined_params['add_post_latent_layers'])
    
    if add_post_latent_layers:
        post_latent_layer_size = trial.suggest_categorical('post_latent_layer_size', combined_params['post_latent_layer_size'])
    else:
        post_latent_layer_size = 1  # Default value when add_post_latent_layers is False

    num_layers_to_retrain = trial.suggest_categorical('num_layers_to_retrain', combined_params['num_layers_to_retrain'])
    
    # Hyperparameters to optimize common to VAEs
    # Dropout rate: float range or fixed
    if isinstance(combined_params['dropout'], tuple):
        min_val, max_val, step = combined_params['dropout']
        dropout = trial.suggest_float('dropout', min_val, max_val, step=step)
    else:
        dropout = combined_params['dropout']
        
    # Learning rate: loguniform or fixed
    if isinstance(combined_params['learning_rate'], tuple):
        learning_rate = trial.suggest_loguniform('learning_rate', combined_params['learning_rate'][0], combined_params['learning_rate'][1])
    else:
        learning_rate = combined_params['learning_rate']
        
    # L1 regularization: loguniform or fixed
    if isinstance(combined_params['l1_reg_weight'], tuple):
        l1_reg_weight = trial.suggest_loguniform('l1_reg_weight', combined_params['l1_reg_weight'][0], combined_params['l1_reg_weight'][1])
    else:
        l1_reg_weight = combined_params['l1_reg_weight']

    # Weight decay: loguniform or fixed
    if isinstance(combined_params['l2_reg_weight'], tuple):
        l2_reg_weight = trial.suggest_loguniform('l2_reg_weight', combined_params['l2_reg_weight'][0],combined_params['l2_reg_weight'][1] )
    else:
        l2_reg_weight = combined_params['l2_reg_weight']

    # Batch size: categorical (fixed list of values) or fixed
    if isinstance(combined_params['batch_size'], tuple):
        min_val, max_val, step = combined_params['batch_size']
        batch_size = trial.suggest_int('batch_size', min_val, max_val, step=step)
    else:
        batch_size = combined_params['batch_size']

    # Patience: integer range or fixed
    if isinstance(combined_params['patience'], tuple):
        min_val, max_val, step = combined_params['patience']
        patience = trial.suggest_int('patience', min_val, max_val, step=step)
    else:
        patience = combined_params['patience']

    # Number of epochs: integer range or fixed
    if isinstance(combined_params['num_epochs'], tuple):
        min_val, max_val, step = combined_params['num_epochs']
        num_epochs = trial.suggest_int('num_epochs', min_val, max_val, step=step)
    else:
        num_epochs = combined_params['num_epochs']     
        

    # Default configuration for fine-tuning, which can be overridden by kwargs
    config = {
        'X_train': X_data_train,
        'y_data_train': y_data_train[task],
        'X_val': X_data_val,
        'y_data_val': y_data_val[task],
        'num_layers_to_retrain': num_layers_to_retrain,
        'add_post_latent_layers': add_post_latent_layers,
        'num_post_latent_layers': 1,
        'post_latent_layer_size': post_latent_layer_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'l1_reg_weight': l1_reg_weight,
        'l2_reg_weight': l2_reg_weight,
        'latent_passes': latent_passes,
        'seed': seed
    }

    if task_type=='classification':
        # Fine-tune TL model
        config['num_classes'] = num_classes
        model_TL, val_metrics_TL = fine_tune_model(
            VAE_model=finetune_VAE_TL, 
            log_path=f"{model_path}/temp_log",
            **config  # Unpack the rest of the arguments from the config
        )

        # Fine-tune no TL model
        model_noTL, val_metrics_noTL = fine_tune_model(
            VAE_model=finetune_VAE_noTL, 
            log_path=f"{model_path}/temp_log",
            **config  # Unpack the rest of the arguments from the config
        )        
    
    elif task_type=='cox':
        # Fine-tune TL model
        config['y_event_train']= y_data_train[task_event]
        config['y_event_val']= y_data_val[task_event]
        model_TL, val_metrics_TL = fine_tune_cox_model(
            VAE_model=finetune_VAE_TL, 
            log_path=f"{model_path}/temp_log",
            patience=patience,  # Early stopping patience
            **config  # Unpack the rest of the arguments from the config
        )

        # Fine-tune no TL model
        model_noTL, val_metrics_noTL = fine_tune_cox_model(
            VAE_model=finetune_VAE_noTL, 
            log_path=f"{model_path}/temp_log",
            patience=patience,  # Early stopping patience
            **config  # Unpack the rest of the arguments from the config
        )
        
    else: 
        raise ValueError('Task type must be either "classification" or "cox"')

    best_log_dir = f"{model_path}/TL_{task.replace(' ','_')}_best_logs_optuna"
    log_dir=f"{model_path}/temp_log"

    if task_type=='classification':
        val_auc_TL = val_metrics_TL['AUC'].iloc[-1]
        #val_auc_noTL = val_metrics_noTL['AUC'].iloc[-1]
        
        if val_auc_TL > best_val_metric:
            best_val_metric = val_auc_TL
            best_model_TL = model_TL
            best_model_noTL = model_noTL
            
            # Copy logs to a "best logs" directory
            if os.path.exists(best_log_dir):
                shutil.rmtree(best_log_dir)  # Remove previous best logs
            shutil.copytree(log_dir, best_log_dir)  # Copy current trial's logs as the best logs
            
        
    elif task_type=='cox':
        
        # optimizng for C-index of TL model
        val_c_index_TL = val_metrics_TL['C-index'].iloc[-1]
        #val_auc_noTL = val_metrics_noTL['AUC'].iloc[-1]

        if val_c_index_TL > best_val_metric:
            best_val_metric = val_c_index_TL
            best_model_TL = model_TL
            best_model_noTL = model_noTL
            
            # Copy logs to a "best logs" directory
            if os.path.exists(best_log_dir):
                shutil.rmtree(best_log_dir)  # Remove previous best logs
            shutil.copytree(log_dir, best_log_dir)  # Copy current trial's logs as the best logs
    else:
        raise ValueError('Task type must be either "classification" or "cox"')
    
    # Save the best models
    torch.save(best_model_TL, f"{model_path}/finetune_TL_{task.replace(' ','_')}_best_model_optuna.pth")
    torch.save(best_model_noTL, f"{model_path}/finetune_noTL_{task.replace(' ','_')}_best_model_optuna.pth")

    
    # return auc_avg
    return best_val_metric




def retrain_finetune_VAE_TL_noTL_Optuna_optimization(finetune_VAE_TL, finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test,  task, task_type, num_classes, task_event, seed, model_path, combined_params, n_trials=50, patience=0, latent_passes= 20, **kwargs):

    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    
    study.optimize(lambda trial: objective(trial, finetune_VAE_TL, finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, task, task_type, num_classes, task_event, seed, model_path, combined_params, patience=0, latent_passes= 20, **kwargs), n_trials=n_trials)


    # Evaluate both models on the test set
    best_model_TL=torch.load(f"{model_path}/finetune_TL_{task.replace(' ','_')}_best_model_optuna.pth")
    best_model_noTL=torch.load(f"{model_path}/finetune_noTL_{task.replace(' ','_')}_best_model_optuna.pth")

    if task_type=='classification':
        #TL
        # Evaluate both models on the test set
        result_metrics_TL,latent_rep_train, latent_rep_val, latent_rep_test = evaluate_model_main(
            best_model_TL, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task, num_classes, seed)
        #NO TL
        result_metrics_noTL, latent_rep_train, latent_rep_val, latent_rep_test = evaluate_model_main(
            best_model_noTL, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task, num_classes, seed
        )
    
    elif task_type=='cox':
        #TL
        result_metrics_TL,latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
            best_model_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed)
        #NO TL
        result_metrics_noTL, latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
            best_model_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed)
    else:
        raise ValueError('Task type must be either "classification" or "cox')


    # Add model labels to differentiate between Transfer Learning and No Transfer Learning results
    result_metrics_TL['Model'] = 'Transfer Learning'
    result_metrics_noTL['Model'] = 'No Transfer Learning'

    # Combine both results into a single DataFrame
    results_combined_df = pd.concat([result_metrics_TL, result_metrics_noTL], ignore_index=True)

    # Move the 'Model' column to the front for clarity
    results_combined_df = results_combined_df[['Model']+ [col for col in results_combined_df.columns if col != 'Model']]

    return study, results_combined_df



import os
import pandas as pd

def save_grid_search_results(results_list, model_path, task):
    
    file_path = f"{model_path}/{task.replace(' ', '_')}_all_grid_search_results.csv"
    
    # Convert results_list to a DataFrame
    new_results_df = pd.DataFrame(results_list)
    
    # Check if the CSV file exists
    if os.path.exists(file_path):
        try:
            # If the file exists, read the existing results into a DataFrame
            existing_df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            # If the file is empty, create an empty DataFrame
            existing_df = pd.DataFrame()
        
        # Append the new results
        combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        
        # Remove duplicates based on all columns
        combined_df = combined_df.drop_duplicates()
    else:
        # If the file doesn't exist, the new results are the only data
        combined_df = new_results_df
    
    # Save the updated DataFrame back to the CSV file
    combined_df.to_csv(file_path, index=False)



def retrain_finetune_VAE_TL_noTL_grid_search_optimization(
    finetune_VAE_TL_OG, finetune_VAE_noTL_OG,
    X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test,
    task, task_type, num_classes, task_event, seed, model_path, combined_params):
    
    # Default configuration for fine-tuning, which can be overridden by kwargs
    config = {
        'X_train': X_data_train,
        'y_data_train': y_data_train[task],
        'X_val': X_data_val,
        'y_data_val': y_data_val[task],
        'seed': seed
    }

    for key, value in combined_params.items():
        # Check if the value is an iterable but not a string or bytes
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            # Wrap the value in a list if it's not iterable
            combined_params[key] = [value]
    print ('combined_params:', combined_params)
    
    param_names = list(combined_params.keys())
    param_values = list(combined_params.values())
    all_param_combinations = list(product(*param_values))
    
    print ('Nummber of grid search paramteres:', len(all_param_combinations))

    best_val_metric = -float('inf')  # Initialize to a very low value to keep track of the best metric
    best_params = None  # Variable to store the best parameters
    
    results_list = []  # List to store results for summary table
    # creating directory to save the results
    temp_path_TL = f"{model_path}/temp_path_TL"
    temp_path_noTL = f"{model_path}/temp_path_noTL"
    best_model_dir_noTL = f"{model_path}/noTL_{task.replace(' ', '_')}_best_model_grid_search"
    best_model_dir_TL = f"{model_path}/TL_{task.replace(' ', '_')}_best_model_grid_search"
    

    for idx, param_values in enumerate(all_param_combinations):
        
        param_combination = dict(zip(param_names, param_values))
        # Create a config for this combination
        config_combination = config.copy()
        config_combination.update(param_combination)

        finetune_VAE_TL=copy.deepcopy(finetune_VAE_TL_OG)
        finetune_VAE_noTL=copy.deepcopy(finetune_VAE_noTL_OG)

        if task_type == 'classification':
            # Fine-tune TL model
            config_combination['num_classes'] = num_classes

            val_metrics_TL = fine_tune_model(
                VAE_model=finetune_VAE_TL,
                model_path=temp_path_TL,
                **config_combination  # Unpack the rest of the arguments from the config
            )

            # Fine-tune no TL model
            val_metrics_noTL = fine_tune_model(
                VAE_model=finetune_VAE_noTL,
                model_path=temp_path_noTL,
                **config_combination  # Unpack the rest of the arguments from the config
            )

            # Evaluate models
            if num_classes == 2:
                val_metric_TL = val_metrics_TL['AUC'].iloc[-1]
                val_metric_noTL = val_metrics_noTL['AUC'].iloc[-1]
            else:
                val_metric_TL = val_metrics_TL['F1 Score'].iloc[-1]
                val_metric_noTL = val_metrics_noTL['F1 Score'].iloc[-1]
            
            # Collect results
            result_entry = param_combination.copy()
            result_entry['val_metric_TL'] = val_metric_TL
            result_entry['val_metric_noTL'] = val_metric_noTL
            results_list.append(result_entry)

            if val_metric_TL > best_val_metric:
                best_val_metric = val_metric_TL
                best_params = param_combination.copy()

                # Copy temp to a "best model" directory - TL
                if os.path.exists(best_model_dir_TL):
                    shutil.rmtree(best_model_dir_TL)  # Remove previous best logs
                shutil.copytree(temp_path_TL, best_model_dir_TL)  # Copy current trial's logs
                
                # Copy logs to a "best logs" directory - noTL
                if os.path.exists(best_model_dir_noTL):
                    shutil.rmtree(best_model_dir_noTL)  # Remove previous best logs
                shutil.copytree(temp_path_noTL, best_model_dir_noTL)  # Copy current trial's logs 

        elif task_type == 'cox':
            # Fine-tune TL model
            config_combination['y_event_train'] = y_data_train[task_event]
            config_combination['y_event_val'] = y_data_val[task_event]

            val_metrics_TL = fine_tune_cox_model(
                VAE_model=finetune_VAE_TL,
                model_path=temp_path_TL,
                **config_combination  # Unpack the rest of the arguments from the config
            )

            # Fine-tune no TL model
            val_metrics_noTL = fine_tune_cox_model(
                VAE_model=finetune_VAE_noTL,
                model_path=temp_path_noTL,
                **config_combination  # Unpack the rest of the arguments from the config
            )

            # Evaluate models
            val_c_index_TL = val_metrics_TL['C-index'].iloc[-1]
            val_c_index_noTL = val_metrics_noTL['C-index'].iloc[-1]

            # Collect results
            result_entry = param_combination.copy()
            result_entry['val_c_index_TL'] = val_c_index_TL
            result_entry['val_c_index_noTL'] = val_c_index_noTL
            results_list.append(result_entry)

            if val_c_index_TL > best_val_metric:
                
                best_val_metric = val_c_index_TL
                best_params = param_combination.copy()

                # Copy the model to a "best model" directory - TL
                if os.path.exists(best_model_dir_TL):
                    shutil.rmtree(best_model_dir_TL)  # Remove previous best logs
                shutil.copytree(temp_path_TL, best_model_dir_TL)  # Copy current trial's logs
                
                # Copy the model to a "best model" directory - no TL
                if os.path.exists(best_model_dir_noTL):
                    shutil.rmtree(best_model_dir_noTL)  # Remove previous best logs
                shutil.copytree(temp_path_noTL, best_model_dir_noTL)  # Copy current trial's logs 

        else:
            raise ValueError('Task type must be either "classification" or "cox"')


    #remove temp_path
    shutil.rmtree(temp_path_TL)  
    shutil.rmtree(temp_path_noTL) 
    
    # Optionally, save the best parameters
    with open(f"{model_path}/{task.replace(' ', '_')}_best_params_grid_search.txt", 'w') as f:
        f.write(str(best_params))
        

    # Save the results to a CSV file
    save_grid_search_results(results_list, model_path, task)

    # load the best models
    best_model_TL= torch.load(f"{best_model_dir_TL}/best_model.pth")
    best_model_noTL= torch.load( f"{best_model_dir_noTL}/best_model.pth")

    # Evaluate both models on the test set
    if task_type=='classification':
        #TL
        # Evaluate both models on the test set
        result_metrics_TL,latent_rep_train, latent_rep_val, latent_rep_test = evaluate_model_main(
            best_model_TL, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task, num_classes, seed)
        #NO TL
        result_metrics_noTL, latent_rep_train, latent_rep_val, latent_rep_test = evaluate_model_main(
            best_model_noTL, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task, num_classes, seed
        )
    
    elif task_type=='cox':
        #TL
        result_metrics_TL,latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
            best_model_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed)
        #NO TL
        result_metrics_noTL, latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
            best_model_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed)
    else:
        raise ValueError('Task type must be either "classification" or "cox')

    # Add model labels to differentiate between Transfer Learning and No Transfer Learning results
    result_metrics_TL['Model'] = 'Transfer Learning'
    result_metrics_noTL['Model'] = 'No Transfer Learning'

    # Combine both results into a single DataFrame
    results_combined_df = pd.concat([result_metrics_TL, result_metrics_noTL], ignore_index=True)

    # Move the 'Model' column to the front for clarity
    results_combined_df = results_combined_df[['Model']+ [col for col in results_combined_df.columns if col != 'Model']]

    return results_combined_df



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
    #Set up the argument parser
    parser = argparse.ArgumentParser(description='It get a list of pre-trained models, and use them to develop finetune VAE models with trasnfer learning of the pretrained model. It also develop a fine-tune model with NO tansfer learning (i,e, random init.). It uses optuna to find the best hyper-paramteres to minize the Recon loss of finetune models. it also visulaize the latent space of the pretrained model used for transfer learning and save all the results.')

    # Define the arguments with defaults and help messages
    parser.add_argument('--input_data_location', type=str,
                        default='/home/leilapirhaji/PROCESSED_DATA',
                        help='Path to the input data location.')

    parser.add_argument('--finetune_save_dir', type=str,
                        default='/home/leilapirhaji/finetune_VAE_models',
                        help='Directory to save finetuned models.')
    
    parser.add_argument('--pretrain_model_name', type=str,
                        default='pretrain_VAE_L_410_490_e_400_p_25_S_8.1.1',
                        help='The name of pretrained model was used for finetuning.')
    
    parser.add_argument('--pretrain_trial_ID', type=str,
                        default='106',
                        help='The name of pretrained model trail was used for finetuning.')
    
    parser.add_argument('--task', type=str,
                        default='IMDC BINARY',
                        help='The task to tune the VAE model for')
    
    parser.add_argument('--task_type', type=str,
                        default='classification',
                        help='either classification or cox')
    
    parser.add_argument('--num_classes', type=str,
                        default='2',
                        help='The number of classes for the classification task')
    
    parser.add_argument('--task_event', type=str,
                        default='OS_event', 
                        help='The censoring event for the cox analysis task')
    
    parser.add_argument('--optimization_type', type=str,
                    default='grid_search', 
                    help='The censoring event for the cox analysis task')
    
    parser.add_argument('--add_post_latent_layers', type=str,
                        default='False,True', 
                        help='If addign a layer post latetn')
    
    parser.add_argument('--post_latent_layer_size', type=str,
                        default='16,32', 
                        help='categorical list of post latent layer size')
    
    parser.add_argument('--num_layers_to_retrain', type=str,
                        default='1,2', 
                        help='the number of layers to retrain')
        
    parser.add_argument('--dropout_rate', nargs='*', default=[0.1], 
                        help='Dropout rate: Either a single value or "min max step" for range (float)')
    parser.add_argument('--learning_rate', nargs='*', default=["1e-5", "1e-2"], 
                        help='Learning rate range: Either a single value or "min max" for range (float)')
    parser.add_argument('--l1_reg', nargs='*', default=["1e-6", "1e-2"],
                        help='Weight decay range: Either a single value or "min max" for range (float)')
    parser.add_argument('--weight_decay', nargs='*', default=["1e-6", "1e-2"],
                        help='Weight decay range: Either a single value or "min max" for range (float)')
    parser.add_argument('--batch_size', nargs='*', default=[32],
                        help='Batch size (either fixed or a range, integer)')
    parser.add_argument('--patience', nargs='*', default=[0], 
                        help='Patience: Either a single value or "min max step" for range (integer), 0 means no early stopping')
    parser.add_argument('--num_epochs', nargs='*', default=[30], 
                        help='Number of epochs (either fixed or a range, integer)')
    parser.add_argument('--n_trials', type=int, default=50, 
                        help='Number of trials for hyperparameter optimization')
    
    # Parse the arguments
    args = parser.parse_args()
    
     # Access the arguments
    input_data_location = args.input_data_location
    finetune_save_dir = args.finetune_save_dir
    pretrain_model_name = args.pretrain_model_name
    pretrain_trial_ID = args.pretrain_trial_ID
    optimization_type=args.optimization_type
    
    task=args.task
    task_type=args.task_type
    num_classes=int(args.num_classes)
    
    add_post_latent_layers=args.add_post_latent_layers.split(',')
    post_latent_layer_size=args.post_latent_layer_size.split(',')
    num_layers_to_retrain=args.num_layers_to_retrain.split(',')
    
    task_event=args.task_event
    n_trials = args.n_trials
    
    
    # Use parsed arguments and convert them using parse_range_or_single
    combined_params = {
        'add_post_latent_layers': [s == 'True' for s in add_post_latent_layers] , #boolean list: [False, True]
        'post_latent_layer_size': [int(x) for x in post_latent_layer_size if x.isdigit()],  # intiger list [16,32]
        'num_layers_to_retrain': [int(x) for x in num_layers_to_retrain if x.isdigit()],  # '1
        'dropout': parse_range_or_single(args.dropout_rate, is_int=False),
        'learning_rate': parse_range_or_single(args.learning_rate, is_int=False),
        'l1_reg_weight': parse_range_or_single(args.l1_reg, is_int=False),
        'l2_reg_weight': parse_range_or_single(args.weight_decay, is_int=False),
        'batch_size': parse_range_or_single(args.batch_size, is_int=True),
        'patience': parse_range_or_single(args.patience, is_int=True),
        'num_epochs': parse_range_or_single(args.num_epochs, is_int=True)
    }

    print ('combined_params:', combined_params)
    
   
    #get fine-tuning input data 
    print ('getting fine-tuning input data')
    (X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_finetune_input_data(input_data_location)


    #get the pre-trained models
    print ('getting fine-tune models')
    (finetune_VAE_TL_OG, finetune_VAE_noTL_OG)=get_finetune_VAE_TL_noTL(finetune_save_dir, pretrain_model_name, pretrain_trial_ID)

    #loading the seed file
    seed=42
    
    # Create a deep copy of the model before retraining
    finetune_VAE_TL = copy.deepcopy(finetune_VAE_TL_OG) 
    finetune_VAE_noTL = copy.deepcopy(finetune_VAE_noTL_OG)

    #path to pre-train and fine-tune models
    model_path=f"{finetune_save_dir}/{pretrain_model_name}/trial_{pretrain_trial_ID}/{task.replace(' ','_')}"
    os.makedirs(model_path, exist_ok=True)
    
    #Running optuna optimization
    print ('running optuna optimization')
    print ('task:', task)
    
    if optimization_type=='optuna':
        study, results_combined_df= retrain_finetune_VAE_TL_noTL_Optuna_optimization(finetune_VAE_TL, finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_type, num_classes, task_event, seed, model_path, combined_params, n_trials=n_trials)
        
    elif optimization_type=='grid_search':
        results_combined_df= retrain_finetune_VAE_TL_noTL_grid_search_optimization(finetune_VAE_TL, finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_type, num_classes, task_event, seed, model_path, combined_params)
    else:
        raise ValueError('optimization_type must be either "optuna" or "grid_search"')
        
    
    #saving the results
    print ('saving the results')
    results_combined_df.to_csv(f"{model_path}/{task.replace(' ','_')}_best_model_results_{optimization_type}.csv")
    #print (results_combined_df)
    
    if n_trials>1 and optimization_type=='optuna':
        #saving the optuna results
        save_optuna_study_results_as_html(study, combined_params, model_path, file_name=f"{task.replace(' ','_')}_optuna_report.html")
        
        
    print ('done')
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()  # run the main function