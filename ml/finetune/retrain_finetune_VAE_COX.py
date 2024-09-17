'''
retraining fine-tune VAE models

'''


import pandas as pd
import importlib
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import optuna


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


#importing fundtion to get encoder info and perfrom tasks 
from models.models_VAE import VAE

from finetune.get_finetune_encoder import  get_finetune_input_data

from finetune.freez_encoder_latent_avg_COX import fine_tune_cox_model, FineTuneCoxModel
from finetune.best_finetune_model_test_eval_cox import best_finetune_model_test_eval_cox





def get_finetune_VAE_TL_noTL(pretrain_model_ID, finetune_save_dir ):


    #path to pre-train and fine-tune models
    models_path=f'{finetune_save_dir}/{pretrain_model_ID}'

    #finetune models files
    finetune_VAE_TL_file= f'{models_path}/Finetune_VAE_pretain_{pretrain_model_ID}_TL_True_model.pth'
    finetune_VAE_TL=torch.load(finetune_VAE_TL_file)

    finetune_VAE_noTL_file= f'{models_path}/Finetune_VAE_pretain_{pretrain_model_ID}_TL_False_model.pth'
    finetune_VAE_noTL=torch.load(finetune_VAE_noTL_file)


    return finetune_VAE_TL, finetune_VAE_noTL






def retrain_finetune_VAE_TL_noTL_fixed_hyper_par(finetune_VAE_TL, finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed, **kwargs):

    # Default configuration for fine-tuning, which can be overridden by kwargs
    config = {
        'X_train': X_data_train,
        'y_data_train': y_data_train[task],
        'y_event_train': y_data_train[task_event],
        'X_val': X_data_val,
        'y_data_val': y_data_val[task],
        'y_event_val': y_data_val[task_event],
        'num_layers_to_retrain': 1,
        'add_post_latent_layers': False,
        'num_post_latent_layers': 1,
        'post_latent_layer_size': 32,
        'num_epochs': 30,
        'batch_size': 32,
        'learning_rate': 1e-5,
        'dropout': 0.25,
        'l1_reg_weight': 1e-7,
        'l2_reg_weight': 1e-7,
        'latent_passes': 20,
        'seed': seed  # Set seed for reproducibility
    }

    # Update config with additional kwargs if provided
    config.update(kwargs)

    
    # Fine-tune TL model
    model_TL, val_metrics_TL = fine_tune_cox_model(
        VAE_model=finetune_VAE_TL,  # Pass the model separately
        **config  # Unpack the rest of the arguments from the config
    )

    # Fine-tune no TL model
    model_noTL, val_metrics_noTL = fine_tune_cox_model(
        VAE_model=finetune_VAE_noTL,  # Pass the model separately
        **config  # Unpack the rest of the arguments from the config
    )

    # Evaluate both models on the test set
    result_metrics_TL, latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
        model_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed)

    result_metrics_noTL, latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
        model_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed
    )

    # Add model labels to differentiate between Transfer Learning and No Transfer Learning results
    result_metrics_TL['Model'] = 'Transfer Learning'
    result_metrics_noTL['Model'] = 'No Transfer Learning'

    # Combine both results into a single DataFrame
    results_combined_df = pd.concat([result_metrics_TL, result_metrics_noTL], ignore_index=True)

    # Move the 'Model' column to the front for clarity
    results_combined_df = results_combined_df[['Model', 'Dataset', 'C-index']]

    return results_combined_df





def retrain_finetune_VAE_TL_noTL_Optuna_optimization(finetune_VAE_TL, finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed, finetune_save_dir, pretrain_model_ID, n_trials=50, **kwargs):

    #path to pre-train and fine-tune models
    models_path=f'{finetune_save_dir}/{pretrain_model_ID}'

    def objective(trial):

        global best_model, best_auc

        best_model_TL = None  # Variable to store the best TL model
        best_model_noTL = None  # Variable to store the best noTL model
        best_c_index = -float('inf')  # Initialize to a very low value to keep track of the best AUC
        
        # Hyperparameters to optimize
        add_post_latent_layers = trial.suggest_categorical('add_post_latent_layers', [True, False])
        if add_post_latent_layers:
            post_latent_layer_size = trial.suggest_categorical('post_latent_layer_size', [8, 32, 64, 128])
        else:
            post_latent_layer_size = 1  # Default value when add_post_latent_layers is False
    
        num_layers_to_retrain = trial.suggest_categorical('num_layers_to_retrain', [0, 1, 2, 3])
        num_epochs = trial.suggest_int('num_epochs', 10, 50)
        batch_size = trial.suggest_categorical('batch_size', [32])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-5)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.4)
        l1_reg_weight = trial.suggest_loguniform('l1_reg_weight', 1e-8, 1e-3)
        l2_reg_weight = trial.suggest_loguniform('l2_reg_weight', 1e-8, 1e-3)

        # Default configuration for fine-tuning, which can be overridden by kwargs
        config = {
            'X_train': X_data_train,
            'y_data_train': y_data_train[task],
            'y_event_train': y_data_train[task_event],
            'X_val': X_data_val,
            'y_data_val': y_data_val[task],
            'y_event_val': y_data_val[task_event],# Default configuration for fine-tuning
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
            'latent_passes': 20,
            'seed': seed
        }

        # Fine-tune TL model
        model_TL, val_metrics_TL = fine_tune_cox_model(
            VAE_model=finetune_VAE_TL, 
            **config  # Unpack the rest of the arguments from the config
        )

        # Fine-tune no TL model
        model_noTL, val_metrics_noTL = fine_tune_cox_model(
            VAE_model=finetune_VAE_noTL, 
            **config  # Unpack the rest of the arguments from the config
        )


        # Calculate the average AUC of TL and noTL
        # Ensure both values are scalars before calculating delta_auc
        val_c_index_TL = val_metrics_TL['C-index'].iloc[-1]
        #val_auc_noTL = val_metrics_noTL['AUC'].iloc[-1]


        if val_c_index_TL > best_c_index:
            best_c_index = val_c_index_TL
            best_model_TL = model_TL
            best_model_noTL = model_noTL

            # Save the best models
            torch.save(best_model_TL, f'{models_path}/pre-train_{pretrain_model_ID}_finetune_TL_{task}_best_COX_model.pth')
            torch.save(best_model_noTL, f'{models_path}/pre-train_{pretrain_model_ID}_finetune_noTL_{task}_best_COX_model.pth')

        # auc_avg = (val_metrics_TL['AUC'].iloc[-1] + val_metrics_noTL['AUC'].iloc[-1]) / 2
        # if auc_avg > best_auc:
        #     best_auc = auc_avg
        #     best_model_TL = model_TL
        #     best_model_noTL = model_noTL

        #     # Save the best models
        #     torch.save(best_model_TL, f'{models_path}/pre-train_{pretrain_model_ID}_finetune_TL_{task}_best_model.pth')
        #     torch.save(best_model_noTL, f'{models_path}/pre-train_{pretrain_model_ID}_finetune_noTL_{task}_best_model.pth')

        # return auc_avg
        return val_c_index_TL

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)


    # Evaluate both models on the test set
    best_model_TL=torch.load(f'{models_path}/pre-train_{pretrain_model_ID}_finetune_TL_{task}_best_COX_model.pth')

    result_metrics_TL,latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
        best_model_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed
    )

    best_model_noTL=torch.load(f'{models_path}/pre-train_{pretrain_model_ID}_finetune_noTL_{task}_best_COX_model.pth')
    result_metrics_noTL, latent_rep_train, latent_rep_val, latent_rep_test = best_finetune_model_test_eval_cox(
        best_model_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed
    )

    # Add model labels to differentiate between Transfer Learning and No Transfer Learning results
    result_metrics_TL['Model'] = 'Transfer Learning'
    result_metrics_noTL['Model'] = 'No Transfer Learning'

    # Combine both results into a single DataFrame
    results_combined_df = pd.concat([result_metrics_TL, result_metrics_noTL], ignore_index=True)

    # Move the 'Model' column to the front for clarity
    results_combined_df = results_combined_df[['Model', 'Dataset', 'C-index']]

    return study, results_combined_df



