'''

plot latent space of pre-trained models

'''


import os
ml_code_path='/home/leilapirhaji/mz_embed_engine/ml'
os.chdir(ml_code_path)

import pandas as pd
import importlib
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import optuna
import imaplib
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


#importing fundtion to get encoder info and perfrom tasks 
from finetune.get_finetune_encoder import  get_finetune_input_data

from models.models_VAE import VAE
from mz_embed_engine.ml.finetune.eval_finetune_latent_neptune_main import visualize_latent_space_multiple_tasks



#input data
input_data_location='/home/leilapirhaji/PROCESSED_DATA_2'
finetune_save_dir='/home/leilapirhaji/finetune_VAE_models' 

#get fine-tuning input data 
(X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_finetune_input_data(input_data_location)

print ('X_data_train:', X_data_train.shape)



#get only the directories in the finetune_save_dir
pretrain_model_list_file='/home/leilapirhaji/pretrained_models_to_finetune.txt'


print ('gettng pretrain_model_list_file')
# Read the text file into a DataFrame
pretrain_model_list = pd.read_csv(pretrain_model_list_file, header=None)[0]
pretrain_model_list = pretrain_model_list.dropna().tolist()

# pretrain_model_list=['RCC-37832', 'RCC-37783', 'RCC-38043', 'RCC-37867', 'RCC-37915', 'RCC-37922', 'RCC-37800', 'RCC-37600']




for pretrain_model_ID in pretrain_model_list:
        
    print ('pretrain_modelID:', pretrain_model_ID)

    #path to pre-train and fine-tune models
    models_path=f'{finetune_save_dir}/{pretrain_model_ID}'

    #pretrain encoder file
    #pretrain_encoder_file= f'{models_path}/{pretrain_model_ID}_encoder_state_dict.pth'

    #finetune models files
    finetune_VAE_TL_file= f'{models_path}/Finetune_VAE_pretain_{pretrain_model_ID}_TL_True_model.pth'
    finetune_VAE_TL=torch.load(finetune_VAE_TL_file)

    finetune_VAE_noTL_file= f'{models_path}/Finetune_VAE_pretain_{pretrain_model_ID}_TL_False_model.pth'
    finetune_VAE_noTL=torch.load(finetune_VAE_noTL_file)

    result_png_file_TL= f'{models_path}/finetune_VAE_{pretrain_model_ID}_TL_latent_space.png'
    result_png_file_noTL= f'{models_path}/finetune_VAE_{pretrain_model_ID}_NO_TL_latent_space.png'

    # Check if the output file already exists
    if os.path.exists(result_png_file_noTL):
        print(f'Output file {result_png_file_noTL} already exists. Skipping this model.')
        continue  # Skip to the next pretrain_modelID
    elif os.path.exists(result_png_file_TL):
        # # without transfer learning
        print ('UMAP visualization of the latent space of finetune model with NO TL')
        visualize_latent_space_multiple_tasks(finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file_noTL)
        continue
    else:
        #UMAP visualization of the latent space of pretrain model
        print ('UMAP visualization of the latent space of finetune model with TL')
        visualize_latent_space_multiple_tasks(finetune_VAE_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file_TL)

        # without transfer learning
        print ('UMAP visualization of the latent space of finetune model with NO TL')
        visualize_latent_space_multiple_tasks(finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file_noTL)
        # Free up memory after processing each model
        del finetune_VAE_TL, finetune_VAE_noTL  # Delete model objects
        gc.collect()  # Explicitly free up memory
        print(f'Finished processing model {pretrain_model_ID}, memory cleaned up.')


 


print ('done')