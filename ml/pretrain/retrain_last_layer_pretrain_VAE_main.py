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
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import optuna.visualization as vis
import plotly.io as pio

from optuna.visualization import plot_optimization_history, plot_param_importances
from jinja2 import Template


#importing fundtion to get encoder info and perfrom tasks 
from models.models_VAE import VAE

from pretrain.get_pretrain_encoder import get_pretrain_encoder_from_local

from pretrain.freez_pretrain_encoder_latent_avg_num import retrain_pretrain_num_task
from pretrain.freez_pretrain_encoder_latent_avg import retrain_pretrain_classification_task
from pretrain.eval_pretrained_VAE import get_avg_latent_space
from pretrain.latent_task_predict import ridge_regression_predict



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
        combined_df = combined_df.drop_duplicates(subset=combined_df.columns[1:10], keep='first', ignore_index=True)
    else:
        # If the file doesn't exist, the new results are the only data
        combined_df = new_results_df
    
    # Save the updated DataFrame back to the CSV file
    combined_df.to_csv(file_path, index=False)




def evalute_pretrain_latent_task_classification( best_model, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task, num_classes, batch_size=32, latent_passes=20):
    
    best_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    
    def eval_dataset (model, X_data, y_data, num_classes, batch_size=32, latent_passes=20):
        # Convert pandas DataFrames to PyTorch tensors
        X_tensor = torch.tensor(X_data.values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_data.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32).to(device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        metrics_df = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Validation Loss'])
        
        all_labels = []
        all_preds = []
        correct = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                # Latent averaging for validation
                latent_reps = []
                for _ in range(latent_passes):  # Multiple passes through the encoder
                    latent_rep = model.encoder(inputs.to(device))  # Ensure inputs are on the same device
                    mu = latent_rep[:, :model.latent_size]  # Slice out mu (first half of the output)
                    latent_reps.append(mu)
                latent_rep = torch.mean(torch.stack(latent_reps), dim=0)

                #print (latent_rep_val.shape)
                model.latent_mode = True  # Use precomputed latent representations for validation
                    
                # Forward pass using averaged latent representations
                outputs = model(latent_rep)
                
                # Switch back to not latent mode
                model.latent_mode = False

                if num_classes == 2:
                    predicted_probs = outputs.squeeze()  # Binary classification probabilities
                    predicted = (predicted_probs >= 0.5).float()  # Binary classification threshold
                    valid_mask = labels >= 0  # Mask for valid labels
                    all_preds.extend(predicted_probs[valid_mask.to(device)].cpu().numpy())  # Use only valid predictions for AUC
                    all_labels.extend(labels[valid_mask.to(device)].cpu().numpy())  # Use only valid labels for AUC
                    # Ensure labels and predicted are on the same device
                    correct += ((predicted == labels.to(device)) & valid_mask.to(device)).sum().item()
                else:
                    outputs_np = outputs.cpu().numpy()  # Convert outputs to numpy
                    valid_mask = (labels >= 0) & (labels < num_classes)
                    
                    # Append predicted probabilities for all samples in the batch
                    all_preds.extend(outputs_np[valid_mask.cpu().numpy()])  # Append predictions for all batches
                    all_labels.extend(labels[valid_mask.to(device)].cpu().numpy())  # Use only valid labels
            
                
            # Calculate metrics
            if num_classes == 2:
                accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
                precision = precision_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
                recall = recall_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
                f1 = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
                auc = roc_auc_score(all_labels, all_preds) * 100
            else:
                # Convert predicted probabilities to predicted class labels for multi-class classification
                all_preds_np = np.vstack(all_preds)  # Stack the list of predictions
                all_preds_class = np.argmax(all_preds_np, axis=1)  # Get the class with the highest probability
                
                accuracy = accuracy_score(all_labels, all_preds_class) * 100
                precision = precision_score(all_labels, all_preds_class, average='weighted') * 100
                recall = recall_score(all_labels, all_preds_class, average='weighted') * 100
                f1 = f1_score(all_labels, all_preds_class, average='weighted') * 100
            
                # Calculate AUC for multi-class classification
                auc = roc_auc_score(all_labels, all_preds_np, multi_class='ovr') * 100  # AUC for multi-class

        # Create a DataFrame with the metrics for the current epoch
        metrics_df = pd.DataFrame({
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1],
            'AUC': [auc],
        })
        
        return metrics_df
    
    metrics_df_train= eval_dataset(best_model, X_data_train, y_data_train[task], num_classes)
    metrics_df_train['Data'] = 'Train'
    metrics_df_val= eval_dataset(best_model, X_data_val, y_data_val[task], num_classes)
    metrics_df_val['Data'] = 'Validation'
    metric_df_test= eval_dataset(best_model, X_data_test, y_data_test[task], num_classes)
    metric_df_test['Data'] = 'Test'
    
    results_all= pd.concat([metrics_df_train, metrics_df_val, metric_df_test], axis=0)
    
    return results_all




def retrain_pretrain_VAE_classification_grid_search_optimization(
    pretrain_VAE_OG,
    X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test,
    task, seed, model_path, combined_params):
    
    num_classes=len(y_data_train[task].dropna().unique())
    print ('task:', task)
    print ('num_classes:', num_classes)
    
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

    best_val_loss = float('inf')  # Initialize to a very low value to keep track of the best metric
    best_val_metric= -float('inf')
    best_params = None  # Variable to store the best parameters
    
    results_list = []  # List to store results for summary table
    # creating directory to save the results
    temp_path = f"{model_path}/temp_path"
    best_model_dir = f"{model_path}/{task.replace(' ', '_')}_best_model_grid_search"
    

    for idx, param_values in enumerate(all_param_combinations):
        
        param_combination = dict(zip(param_names, param_values))
        # Create a config for this combination
        config_combination = config.copy()
        config_combination.update(param_combination)
        config_combination['num_classes'] = num_classes


        pretrain_VAE=copy.deepcopy(pretrain_VAE_OG)

        val_metrics = retrain_pretrain_classification_task(
            VAE_model=pretrain_VAE,
            model_path=temp_path,
            **config_combination  # Unpack the rest of the arguments from the config
        )

        # Evaluate models
        if num_classes == 2:
            val_metric = val_metrics['AUC'].iloc[-1]
        else:
            val_metric  = val_metrics ['F1 Score'].iloc[-1]
        
        # Collect results
        result_entry = param_combination.copy()
        result_entry['val_metric'] = val_metric
        results_list.append(result_entry)
        
        
        if val_metric > best_val_metric:
            
            best_val_metric = val_metric
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
    model_results_df = evalute_pretrain_latent_task_classification(
        best_model, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task, num_classes)
    
    
    # Add model labels to differentiate between Transfer Learning and No Transfer Learning results
    #model_results_df['Model'] = 'Pretrain VAE'

    # Combine both results into a single DataFrame
    results_combined_df = model_results_df

    # Move the 'Model' column to the front for clarity
    results_combined_df = results_combined_df[['Data']+ [col for col in results_combined_df.columns if col != 'Data']]

    return results_combined_df






def evalute_pretrain_latent_task_regression( vae_model, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task, latent_passes=20):
    
    def eval_dataset_reg(model, X_data, y_data, batch_size=32, latent_passes=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert pandas DataFrames to PyTorch tensors
        X_tensor = torch.tensor(X_data.values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_data.fillna(-1).values, dtype=torch.float32).to(device)
    
        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        metrics_df = pd.DataFrame(columns=['Epoch', 'Mean sq. error', 'Mean avg. error', 'R2', 'Validation Loss'])
        
        # Validation
        model.eval()
        model.to(device)
        #model.latent_mode = True  # Use precomputed latent representations for validation
        
        all_labels = []
        all_preds = []
        all_preds_np=[]
        all_labels_np=[]
        
        # Validation loop (corrected)
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Latent averaging for validation
                latent_reps = []
                for _ in range(latent_passes):  # Multiple passes through the encoder
                    latent_rep = model.encoder(inputs.to(device))  # Ensure inputs are on the same device
                    mu = latent_rep[:, :model.latent_size]  # Slice out mu (first half of the output)
                    latent_reps.append(mu)
                latent_rep_val = torch.mean(torch.stack(latent_reps), dim=0)

                #print (latent_rep_val.shape)
                model.latent_mode = True  # Use precomputed latent representations for validation
                # Forward pass using averaged latent representations
                outputs = model(latent_rep_val)

                # Switch back to not latent mode
                model.latent_mode = False

                # Move predictions and labels back to CPU
                outputs_np = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # Ensure valid_mask is applied to both predictions and labels
                valid_mask = (labels_np >= 0)  # Assuming you want to ignore invalid labels

                # Append predicted probabilities for all samples in the batch
                all_preds.extend(outputs_np[valid_mask])  # Append predictions for all batches
                all_labels.extend(labels_np[valid_mask])  # Use only valid labels
        
        # Convert lists to NumPy arrays
        all_preds_np = np.array(all_preds)  
        all_labels_np = np.array(all_labels)
        
        # Calculate regression metrics
        mse = mean_squared_error(all_labels_np, all_preds_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_labels_np, all_preds_np)
        r2 = r2_score(all_labels_np, all_preds_np)  # R-squared metric
        
        # Create a DataFrame with the metrics for the current epoch
        metrics_df = pd.DataFrame({
            'Mean sq. error': [mse],
            'Mean avg. error': [mae],
            'R2': [r2]
        })

        return metrics_df
            
        
    metric_df_train= eval_dataset_reg(vae_model, X_data_train, y_data_train[task])
    metric_df_train['Data'] = 'Train'
    metric_df_val= eval_dataset_reg(vae_model, X_data_val, y_data_val[task])
    metric_df_val['Data'] = 'Validation'
    metric_df_test= eval_dataset_reg(vae_model, X_data_test, y_data_test[task])
    metric_df_test['Data'] = 'Test'
    
    model_results_df= pd.concat([metric_df_train, metric_df_val, metric_df_test], axis=0)
    
    return model_results_df


def retrain_pretrain_VAE_regression_grid_search_optimization(
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
    
    print ('Number of grid search paramteres:', len(all_param_combinations))

    best_val_loss = float('inf')  # Initialize to a very low value to keep track of the best metric
    best_val_mae= float('inf')
    best_params = None  # Variable to store the best parameters
    
    results_list = []  # List to store results for summary table
    # creating directory to save the results
    temp_path = f"{model_path}/temp_path"
    best_model_dir = f"{model_path}/{task.replace(' ', '_')}_best_model_grid_search"
    

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
        
        
        if val_loss < best_val_loss:
            
            best_val_loss = val_mae
            best_params = param_combination.copy()

            # Copy the model to a "best model" directory - TL
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)  # Remove previous best logs
            shutil.copytree(temp_path, best_model_dir)  # Copy current trial's logs
            
        # if val_mae < best_val_mae:
            
        #     best_val_mae = val_mae
        #     best_params = param_combination.copy()

        #     # Copy the model to a "best model" directory - TL
        #     if os.path.exists(best_model_dir):
        #         shutil.rmtree(best_model_dir)  # Remove previous best logs
        #     shutil.copytree(temp_path, best_model_dir)  # Copy current trial's logs
            
    
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
    model_results_df = evalute_pretrain_latent_task_regression(
        best_model, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task)
    
    # Combine both results into a single DataFrame
    results_combined_df = model_results_df

    # Move the 'Model' column to the front for clarity
    results_combined_df = results_combined_df[['Data']+ [col for col in results_combined_df.columns if col != 'Data']]

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
    
    parser.add_argument('--task_type', type=str,
                        default='regression',
                        help='regression or classification')
        
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
    task_type=args.task_type
    
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
    
    if task_type=='classification':
        results_combined_df= retrain_pretrain_VAE_classification_grid_search_optimization(pretrain_vae, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, seed, model_path, combined_params)
        
    elif task_type=='regression':
        results_combined_df= retrain_pretrain_VAE_regression_grid_search_optimization(pretrain_vae, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, seed, model_path, combined_params)
    else: 
        print ('task type is not correct')
    
    
    #saving the results
    print ('saving the results')
    results_combined_df.to_csv(f"{model_path}/{task.replace(' ','_')}_best_model_results.csv")
    #print (results_combined_df)
    
        
        
    print ('done')
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()  # run the main function