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

from finetune.freez_encoder_latent_avg_syn_COX import fine_tune_adv_cox_model, FineTuneCoxModel
from finetune.best_finetune_model_test_eval_cox import best_finetune_model_test_eval_cox




def get_finetune_VAE_TL_noTL(finetune_save_dir, pretrain_name, pretrain_trial_id):


    #path to pre-train and fine-tune models
    models_path=f'{finetune_save_dir}/{pretrain_name}/trial_{pretrain_trial_id}'
    
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #finetune models files
    finetune_VAE_TL_file= f'{models_path}/finetune_VAE_TL_True_best_model_state.pt'
    finetune_VAE_TL=torch.load(finetune_VAE_TL_file, map_location=device)

    finetune_VAE_noTL_file= f'{models_path}/finetune_VAE_TL_False_best_model_state.pt'
    finetune_VAE_noTL=torch.load(finetune_VAE_noTL_file, map_location=device)


    return finetune_VAE_TL, finetune_VAE_noTL



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



def retrain_finetune_VAE_TL_adverserial_cox_grid_search_optimization(
    finetune_VAE_TL_OG,
    X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test,
    task, adv_task, task_event, lambda_adv, seed, model_path, combined_params):
    
    # Default configuration for fine-tuning, which can be overridden by kwargs
    config = {
        'X_train': X_data_train,
        'y_data_train': y_data_train[task],
        'y_data_train_adv': y_data_train[adv_task],
        'y_event_train': y_data_train[task_event],
        'X_val': X_data_val,
        'y_data_val': y_data_val[task],
        'y_data_val_adv': y_data_val[adv_task],
        'y_event_val': y_data_val[task_event],
        'lambda_adv': lambda_adv,
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
    best_val_loss= float('inf')
    best_params = None  # Variable to store the best parameters
    
    results_list = []  # List to store results for summary table
    # creating directory to save the results
    temp_path_TL = f"{model_path}/temp_path_TL"
    best_model_dir_TL = f"{model_path}/TL_{task.replace(' ', '_')}_best_model_grid_search"
    

    for idx, param_values in enumerate(all_param_combinations):
        
        param_combination = dict(zip(param_names, param_values))
        # Create a config for this combination
        config_combination = config.copy()
        config_combination.update(param_combination)


        finetune_VAE_TL=copy.deepcopy(finetune_VAE_TL_OG)

        val_metrics_TL = fine_tune_adv_cox_model(
            VAE_model=finetune_VAE_TL,
            model_path=temp_path_TL,
            **config_combination  # Unpack the rest of the arguments from the config
        )

        # Evaluate models
        val_c_index_TL = val_metrics_TL['C-index'].iloc[-1]
        val_loss_index_TL = val_metrics_TL['Validation Loss'].iloc[-1]
        val_adv_c_index_TL = val_metrics_TL['Syn. C-index'].iloc[-1]
        avg_c_index= (val_c_index_TL + val_adv_c_index_TL)/2

        # Collect results
        result_entry = param_combination.copy()
        result_entry['lambda_syn'] = lambda_adv
        result_entry['val_c_index_TL'] = val_c_index_TL
        result_entry['Syn. C-index'] = val_adv_c_index_TL
        result_entry['val_loss'] = val_loss_index_TL
        result_entry['Avg_c_index'] = avg_c_index
        results_list.append(result_entry)
        
        # if lambda_adv<0 and val_loss_index_TL < best_val_loss:
            
        #     best_val_metric = val_loss_index_TL
        #     best_params = param_combination.copy()

        #     # Copy the model to a "best model" directory - TL
        #     if os.path.exists(best_model_dir_TL):
        #         shutil.rmtree(best_model_dir_TL)  # Remove previous best logs
        #     shutil.copytree(temp_path_TL, best_model_dir_TL)  # Copy current trial's logs
            

        if avg_c_index > best_val_metric:
            
            best_val_metric = avg_c_index
            best_params = param_combination.copy()

            # Copy the model to a "best model" directory - TL
            if os.path.exists(best_model_dir_TL):
                shutil.rmtree(best_model_dir_TL)  # Remove previous best logs
            shutil.copytree(temp_path_TL, best_model_dir_TL)  # Copy current trial's logs
            
    
    #remove temp_path
    shutil.rmtree(temp_path_TL)  
    
    # Optionally, save the best parameters
    with open(f"{model_path}/{task.replace(' ', '_')}_best_params_grid_search.txt", 'w') as f:
        f.write(str(best_params))
        

    # Save the results to a CSV file
    save_grid_search_results(results_list, model_path, task)

    # load the best models
    best_model_TL= torch.load(f"{best_model_dir_TL}/best_model.pth")

    # Evaluate both models on the test set     #TL
    result_metrics_TL,latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
        best_model_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed)
    
    
    # Add model labels to differentiate between Transfer Learning and No Transfer Learning results
    result_metrics_TL['Model'] = 'Transfer Learning'


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
                        default='NIVO OS',
                        help='The task to tune the VAE model for')
    
    parser.add_argument('--syn_task', type=str,
                        default='EVER OS',
                        help='The task for adverserial learning')
    
    parser.add_argument('--task_event', type=str,
                        default='OS_event', 
                        help='The censoring event for the cox analysis task')
    
    parser.add_argument('--lambda_syn', type=float,
                        default='1.0', 
                        help='the weight of adverserial loss')
    
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
    finetune_save_dir = args.finetune_save_dir
    pretrain_model_name = args.pretrain_model_name
    pretrain_trial_ID = args.pretrain_trial_ID
    
    task=args.task
    syn_task=args.syn_task
    task_event=args.task_event
    lambda_syn=float(args.lambda_syn)
    
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
    print ('getting fine-tuning input data')
    (X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_finetune_input_data(input_data_location)


    #get the pre-trained models
    print ('getting fine-tune models')
    (finetune_VAE_TL_OG, finetune_VAE_noTL_OG)=get_finetune_VAE_TL_noTL(finetune_save_dir, pretrain_model_name, pretrain_trial_ID)

    #loading the seed file
    seed=42
    
    # Create a deep copy of the model before retraining
    finetune_VAE_TL = copy.deepcopy(finetune_VAE_TL_OG) 

    #path to pre-train and fine-tune models
    model_path=f"{finetune_save_dir}/{pretrain_model_name}/trial_{pretrain_trial_ID}/{task.replace(' ','_')}_synergistic_{syn_task.replace(' ','_')}"
    os.makedirs(model_path, exist_ok=True)
    
    #Running optuna optimization
    print ('running optuna optimization')
    print ('task:', task)
    
    results_combined_df= retrain_finetune_VAE_TL_adverserial_cox_grid_search_optimization(finetune_VAE_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, syn_task, task_event, lambda_syn, seed, model_path, combined_params)
    
    
    #saving the results
    print ('saving the results')
    results_combined_df.to_csv(f"{model_path}/{task.replace(' ','_')}_best_model_results_synergistic.csv")
    #print (results_combined_df)
    
        
        
    print ('done')
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()  # run the main function