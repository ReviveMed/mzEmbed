
''' 

this function get an VAE models and a task and return the prediction of the task using the average latent space of the VAE model


'''

import pandas as pd
import importlib
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import optuna
import imaplib


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


#importing fundtion to get encoder info and perfrom tasks 

from get_finetune_encoder import  get_finetune_input_data
from models_VAE import VAE
from latent_task_predict import log_reg_multi_class, cox_proportional_hazards



def generate_latent_space(X_data, encoder, batch_size=128):
    if isinstance(X_data, pd.DataFrame):
        x_index = X_data.index
        X_data = torch.tensor(X_data.to_numpy(), dtype=torch.float32)
    Z = torch.tensor([])
    encoder.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    with torch.inference_mode():
        for i in range(0, len(X_data), batch_size):
            # print(i, len(X_data))
            X_batch = X_data[i:i+batch_size].to(device)
            Z_batch = encoder.transform(X_batch)
            Z_batch = Z_batch.cpu()
            Z = torch.cat((Z, Z_batch), dim=0)
        Z = Z.detach().numpy()
        Z = pd.DataFrame(Z, index=x_index)
    encoder.to('cpu')
    return Z



def generate_average_latent_space(X_data, model, batch_size, num_times=10):
    latent_space_sum = None
    for i in range(num_times):
        latent_space = generate_latent_space(X_data, model, batch_size)
        if latent_space_sum is None:
            latent_space_sum = latent_space
        else:
            latent_space_sum += latent_space  # Sum the latent spaces
    latent_space_avg = latent_space_sum / num_times  # Compute the average
    return latent_space_avg





def predict_task_from_latent_avg(vae_model, task, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10):
    
    # Generate latent spaces 10 times for each dataset
    # Generate averaged latent spaces
    Z_train = generate_average_latent_space(X_data_train, vae_model, batch_size, num_times)
    Z_val = generate_average_latent_space(X_data_val, vae_model, batch_size, num_times)
    Z_test = generate_average_latent_space(X_data_test, vae_model, batch_size, num_times)


    (best_val_accuracy, best_val_auc, test_accuracy, test_auc)= log_reg_multi_class(task, Z_train, y_data_train, Z_val, y_data_val, Z_test, y_data_test)

    return best_val_accuracy, best_val_auc, test_accuracy, test_auc





def predict_survival_from_latent_avg(vae_model, task, task_event, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10):
    
    # Generate latent spaces 10 times for each dataset
    # Generate averaged latent spaces
    Z_train = generate_average_latent_space(X_data_train, vae_model, batch_size, num_times)
    Z_val = generate_average_latent_space(X_data_val, vae_model, batch_size, num_times)
    Z_test = generate_average_latent_space(X_data_test, vae_model, batch_size, num_times)


    Y_train_OS = y_data_train[task]
    Y_train_event = y_data_train[task_event]

    Y_val_OS = y_data_val[task]
    Y_val_event = y_data_val[task_event]

    Y_test_OS = y_data_test[task]
    Y_test_event = y_data_test[task_event]


    best_val_c_index, best_test_c_index, best_params= cox_proportional_hazards(Z_train, Y_train_OS, Y_train_event, Z_val, Y_val_OS, Y_val_event, Z_test, Y_test_OS, Y_test_event)

    return best_val_c_index, best_test_c_index, best_params





def main():
    
    #input data
    input_data_location='/home/leilapirhaji/PROCESSED_DATA_2'
    finetune_save_dir='/home/leilapirhaji/finetune_VAE_models' 

    #tasks to predict using encoder
    task_list_cat=['Benefit BINARY', 'Nivo Benefit BINARY', 'MSKCC BINARY', 'IMDC BINARY', 'Benefit ORDINAL', 'MSKCC ORDINAL', 'IMDC ORDINAL', 'ORR', 'Benefit', 'IMDC', 'MSKCC', 'Prior_2' ]

    #survival tasks
    task_list_survival=[ 'OS', 'PFS']


    #get fine-tuning input data 
    (X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_finetune_input_data(input_data_location)


    #loading the VAE modesl developed with and without transfer leanring
    pretrain_model_ID='RCC-37520'
    print (f'Predicting tasks using the latnet space of the VAE models with pre-train model ID: {pretrain_model_ID}') 

    #path to pre-train and fine-tune models
    models_path=f'{finetune_save_dir}/{pretrain_model_ID}'

    #finetune models files
    finetune_VAE_TL_file= f'{models_path}/Finetune_VAE_pretain_{pretrain_model_ID}_TL_True_model.pth'
    finetune_VAE_TL=torch.load(finetune_VAE_TL_file)

    finetune_VAE_noTL_file= f'{models_path}/Finetune_VAE_pretain_{pretrain_model_ID}_TL_False_model.pth'
    finetune_VAE_noTL=torch.load(finetune_VAE_noTL_file)


    #predicting tasks using the latnet space of the VAE models
    results =[]

    for task in task_list_cat:
        print (f'Predicting task: {task}')
        # predicting tasks using the latnet space of the VAE models with transfer learning 
        best_val_accuracy_TL, best_val_auc_TL, test_accuracy_TL, test_auc_TL= predict_task_from_latent_avg (finetune_VAE_TL, task, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10)

        # predicting tasks using the latnet space of the VAE models without transfer learning
        best_val_accuracy_noTL, best_val_auc_noTL, test_accuracy_noTL, test_auc_noTL= predict_task_from_latent_avg (finetune_VAE_noTL, task, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10)


        # Append the results to the list
        results.append({
            'Task': task,
            'Best Val Accuracy TL': best_val_accuracy_TL,
            'Best Val AUC TL': best_val_auc_TL,
            'Test Accuracy TL': test_accuracy_TL,
            'Test AUC TL': test_auc_TL, 
            'Best Val Accuracy NO TL': best_val_accuracy_noTL,
            'Best Val AUC NO TL': best_val_auc_noTL,
            'Test Accuracy NO TL': test_accuracy_noTL,
            'Test AUC NO TL': test_auc_noTL
        })




    # predicting survival tasks using the latnet space of the VAE models


    for task in task_list_survival:
        print (f'Predicting survival task: {task}')

        task_event= f'{task}_Event'

        # predicting survival tasks using the latnet space of the VAE models with transfer learning 
        best_val_c_index_TL, best_test_c_index_TL, best_params_TL= predict_survival_from_latent_avg (finetune_VAE_TL, task, task_event, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10)

        # predicting survival tasks using the latnet space of the VAE models without transfer learning
        best_val_c_index_noTL, best_test_c_index_noTL, best_params_noTL= predict_survival_from_latent_avg (finetune_VAE_noTL, task, task_event, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10)


        # Append the results to the list
        results.append({
            'Task': task,
            'Best Val C-Index TL': best_val_c_index_TL,
            'Best Test C-Index TL': best_test_c_index_TL,
            'Best Penalizer, TL': best_params_TL['penalizer'],
            'Best Val C-Index NO TL': best_val_c_index_noTL,
            'Best Test C-Index NO TL': best_test_c_index_noTL,
            'Best Penalizer, NO TL': best_params_noTL['penalizer']
        })


    # Convert the list of results to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(f'{models_path}/pre-train_{pretrain_model_ID}_latnet_avg_predictions.csv', index=False)

    print (f'Predictions are saved in: {models_path}/pre-train_{pretrain_model_ID}_latnet_avg_predictions.csv')











if __name__=='__main__':
    main()