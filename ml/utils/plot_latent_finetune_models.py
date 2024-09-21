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



#importing fundtion to get encoder info and perfrom tasks 
from finetune.get_finetune_encoder import  get_finetune_input_data

from models.models_VAE import VAE
from finetune.eval_finetune_latent_neptune_main import visualize_latent_space_multiple_tasks



#input data
input_data_location='/home/leilapirhaji/PROCESSED_DATA'
finetune_save_dir='/home/leilapirhaji/finetune_VAE_models' 

#get fine-tuning input data 
(X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_finetune_input_data(input_data_location)

print ('X_data_train:', X_data_train.shape)



#get only the directories in the finetune_save_dir
#pretrain_model_list_file='/home/leilapirhaji/pretrained_models_to_finetune.txt'

# print ('gettng pretrain_model_list_file')
# # Read the text file into a DataFrame
# pretrain_model_list = pd.read_csv(pretrain_model_list_file, header=None)[0]
# pretrain_model_list = pretrain_model_list.dropna().tolist()

# pretrain_model_list=['RCC-37832', 'RCC-37783', 'RCC-38043', 'RCC-37867', 'RCC-37915', 'RCC-37922', 'RCC-37800', 'RCC-37600']

pretrain_id_list=[81, 51]
pretrain_name= 'pretrain_VAE_Latent_464'


for pretrain_id in pretrain_id_list:
        
    print ('pretrain_modelID:', pretrain_id)

    #path to pre-train and fine-tune models
    models_path=f'{finetune_save_dir}/{pretrain_name}/trial_{pretrain_id}'

    #path to pre-train and fine-tune models
    models_path=f'{finetune_save_dir}/{pretrain_name}/trial_{pretrain_id}'

    #finetune models files
    finetune_VAE_TL_file= f'{models_path}/finetune_VAE_TL_True_model.pth'
    finetune_VAE_TL=torch.load(finetune_VAE_TL_file)

    finetune_VAE_noTL_file= f'{models_path}/finetune_VAE_TL_False_model.pth'
    finetune_VAE_noTL=torch.load(finetune_VAE_noTL_file)


    # visulizing the latnet space of the VAE models

    # with transfer learning
    result_png_file_TL= f'{models_path}/finetune_VAE_TL_latent_space.png'
    
    # without transfer learning
    result_png_file_noTL= f'{models_path}/finetune_VAE_noTL_latent_space.png'
    


    # Check if the output file already exists
    if os.path.exists(result_png_file_noTL):
        print(f'Output file {result_png_file_noTL} already exists. Skipping this model.')
        continue  # Skip to the next pretrain_modelID
    elif os.path.exists(result_png_file_TL):
        # # without transfer learning
        print ('UMAP visualization of the latent space of finetune model with NO TL')
        visualize_latent_space_multiple_tasks(finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file_TL)
        continue
    else:
        #UMAP visualization of the latent space of pretrain model
        print ('UMAP visualization of the latent space of finetune model with TL')
        visualize_latent_space_multiple_tasks(finetune_VAE_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file_TL)

        # without transfer learning
        print ('UMAP visualization of the latent space of finetune model with NO TL')
        visualize_latent_space_multiple_tasks(finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file_TL)

        # Free up memory after processing each model
        del finetune_VAE_TL, finetune_VAE_noTL  # Delete model objects
        gc.collect()  # Explicitly free up memory
        print(f'Finished processing model trail_{pretrain_id}, memory cleaned up.')


 


print ('done')