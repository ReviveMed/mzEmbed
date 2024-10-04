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

from pretrain.get_pretrain_encoder import get_pretrain_encoder_from_local

from pretrain.freez_pretrain_encoder_latent_avg_num import retrain_pretrain_num_task
from pretrain.eval_pretrained_VAE import evalute_pretrain_latent_extra_task



def get_pretrain_input_data(data_location):

    #defining the input datasets
    pretrain_X_all=f'{data_location}/X_Pretrain_All.csv'
    pretrain_y_all=f'{data_location}/y_Pretrain_All.csv'

    pretrain_X_train=f'{data_location}/X_Pretrain_Discovery_Train.csv'
    pretrain_y_train=f'{data_location}/y_Pretrain_Discovery_Train.csv'

    pretrain_X_val=f'{data_location}/X_Pretrain_Discovery_Val.csv'
    pretrain_y_val=f'{data_location}/y_Pretrain_Discovery_Val.csv'

    pretrain_X_test=f'{data_location}/X_Pretrain_Test.csv'
    pretrain_y_test=f'{data_location}/y_Pretrain_Test.csv'

    #loading the data
    X_data_all = pd.read_csv(pretrain_X_all, index_col=0)
    y_data_all = pd.read_csv(pretrain_y_all, index_col=0)

    X_data_train = pd.read_csv(pretrain_X_train, index_col=0)
    y_data_train = pd.read_csv(pretrain_y_train, index_col=0)

    X_data_val = pd.read_csv(pretrain_X_val, index_col=0)
    y_data_val = pd.read_csv(pretrain_y_val, index_col=0)

    X_data_test = pd.read_csv(pretrain_X_test, index_col=0)
    y_data_test = pd.read_csv(pretrain_y_test, index_col=0)

    #returning the data
    return(X_data_all, y_data_all, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)




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




def retrain_pretrain_VAE_numerical_grid_search_optimization(
    pretrain_VAE_OG,
    X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test,
    task, seed, model_path, combined_params):
    
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
    best_val_mae= float('inf')
    best_params = None  # Variable to store the best parameters
    
    results_list = []  # List to store results for summary table
    # creating directory to save the results
    temp_path = f"{model_path}/temp_path_TL"
    best_model_dir = f"{model_path}/TL_{task.replace(' ', '_')}_best_model_grid_search"
    

    for idx, param_values in enumerate(all_param_combinations):
        
        param_combination = dict(zip(param_names, param_values))
        # Create a config for this combination
        config_combination = config.copy()
        config_combination.update(param_combination)


        pretrain_VAE=copy.deepcopy(pretrain_VAE_OG)

        val_metrics = retrain_pretrain_num_task(
            VAE_model=pretrain_VAE,
            model_path=temp_path,
            **config_combination  # Unpack the rest of the arguments from the config
        )

        # Evaluate models
        val_mae = val_metrics ['Mean avg. error'].iloc[-1]
        val_mse = val_metrics ['Mean sq. error'].iloc[-1]
        val_r2 = val_metrics ['R2'].iloc[-1]
        val_loss = val_metrics ['Validation Loss'].iloc[-1]


        # Collect results
        result_entry = param_combination.copy()
        result_entry['Mean avg. error'] = val_mae
        result_entry['Mean sq. error'] = val_mse
        result_entry['R2'] = val_r2
        result_entry['val_loss'] = val_loss
        results_list.append(result_entry)
        
        if val_mae < best_val_mae:
            
            best_val_mae = val_mae
            best_params = param_combination.copy()

            # Copy the model to a "best model" directory - TL
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)  # Remove previous best logs
            shutil.copytree(temp_path, best_model_dir)  # Copy current trial's logs
            
    
    #remove temp_path
    shutil.rmtree(temp_path)  
    
    # Optionally, save the best parameters
    with open(f"{model_path}/{task.replace(' ', '_')}_best_params_grid_search.txt", 'w') as f:
        f.write(str(best_params))
        

    # Save the results to a CSV file
    save_grid_search_results(results_list, model_path, task)

    # load the best models
    best_model= torch.load(f"{best_model_dir}/best_model.pth")

    # Evaluate both models on the test set 
    result_metrics_TL,latent_rep_train, latent_rep_val, latent_rep_test = evalute_pretrain_latent_extra_task(
        best_model, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, best_model_dir)
    
    
    # Add model labels to differentiate between Transfer Learning and No Transfer Learning results
    result_metrics_TL['Model'] = 'Pretrain VAE'


    # Combine both results into a single DataFrame
    results_combined_df = result_metrics_TL

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
                        default='/home/leilapirhaji/PROCESSED_DATA_S_8.1.1',
                        help='Path to the input data location.')

    parser.add_argument('--pretrain_save_dir', type=str,
                        default='/home/leilapirhaji/pretrained_models',
                        help='Directory to save finetuned models.')
    
    parser.add_argument('--pretrain_model_name', type=str,
                        default='pretrain_VAE_L_410_490_e_400_p_25_S_8.1.1',
                        help='The name of pretrained model was used for finetuning.')
    
    parser.add_argument('--pretrain_trial_ID', type=str,
                        default='106',
                        help='The name of pretrained model trail was used for finetuning.')
    
    parser.add_argument('--task', type=str,
                        default='BMI',
                        help='The task to tune the VAE model for')
        
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
    
    # Parse the arguments
    args = parser.parse_args()
    
     # Access the arguments
    input_data_location = args.input_data_location
    pretrain_save_dir = args.pretrain_save_dir
    pretrain_model_name = args.pretrain_model_name
    pretrain_trial_ID = args.pretrain_trial_ID
    
    task=args.task
    
    add_post_latent_layers=args.add_post_latent_layers.split(',')
    post_latent_layer_size=args.post_latent_layer_size.split(',')
    num_layers_to_retrain=args.num_layers_to_retrain.split(',')
   
    
    
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
    print ('getting pretrain input data')
    (X_data_all, y_data_all, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=  get_pretrain_input_data(input_data_location)


    #get the pre-trained models
    print ('getting pretrain VAE models')
    (pretrain_vae_OG, Z_train, Z_val, Z_test)= get_pretrain_encoder_from_local(pretrain_model_name, pretrain_trial_ID, pretrain_save_dir)

    #loading the seed file
    seed=42
    
    # Create a deep copy of the model before retraining
    pretrain_vae = copy.deepcopy(pretrain_vae_OG) 

    #path to pre-train and fine-tune models
    model_path=f"{pretrain_save_dir}/{pretrain_model_name}/trial_{pretrain_trial_ID}/{task.replace(' ','_')}"
    os.makedirs(model_path, exist_ok=True)
    
    #Running optuna optimization
    print ('running grid search of hyper-par')
    print ('task:', task)
    
    results_combined_df= retrain_pretrain_VAE_numerical_grid_search_optimization(pretrain_vae, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, seed, model_path, combined_params)
    
    
    #saving the results
    print ('saving the results')
    results_combined_df.to_csv(f"{model_path}/{task.replace(' ','_')}_best_model_results_adverserial.csv")
    #print (results_combined_df)
    
        
        
    print ('done')
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()  # run the main function