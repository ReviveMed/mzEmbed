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


import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score



from pretrain.get_pretrain_encoder import get_pretrain_input_data, visualize_latent_space_multiple_tasks




input_data_location='/home/leilapirhaji/PROCESSED_DATA_2'

print ('input_data_location:', input_data_location)
(X_data_all, y_data_all, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_pretrain_input_data(input_data_location)

print ('X_data_all.shape:', X_data_all.shape)

finetune_save_dir='/home/leilapirhaji/finetune_VAE_models'
#get only the directories in the finetune_save_dir
pretrain_model_list_file='/home/leilapirhaji/pretrained_models_to_finetune_v2.txt'

print ('gettng pretrain_model_list_file')
# Read the text file into a DataFrame
pretrain_model_list = pd.read_csv(pretrain_model_list_file, header=None)[0]
pretrain_model_list = pretrain_model_list.dropna().tolist()




for pretrain_modelID in pretrain_model_list:
        
    print ('pretrain_modelID:', pretrain_modelID)

    embeding_path = f'{finetune_save_dir}/{pretrain_modelID}'
    output_path=f'{finetune_save_dir}/{pretrain_modelID}'
    result_file_name=f'{output_path}/{pretrain_modelID}_latent_space_visualization.png'

    # Check if the output file already exists
    if os.path.exists(result_file_name):
        print(f'Output file {result_file_name} already exists. Skipping this model.')
        continue  # Skip to the next pretrain_modelID

    Z_pretrain_all=pd.read_csv(f'{embeding_path}/Z_all_{pretrain_modelID}.csv').iloc[:, 1:]
    Z_pretrain_train=pd.read_csv(f'{embeding_path}/Z_train_{pretrain_modelID}.csv').iloc[:, 1:]
    Z_pretrain_val=pd.read_csv(f'{embeding_path}/Z_val_{pretrain_modelID}.csv').iloc[:, 1:]
    Z_pretrain_test=pd.read_csv(f'{embeding_path}/Z_test_{pretrain_modelID}.csv').iloc[:, 1:]


    #UMAP visualization of the latent space of pretrain model
    print ('UMAP visualization of the latent space of pretrain model')
    
    visualize_latent_space_multiple_tasks( pretrain_modelID, output_path, Z_pretrain_all, Z_pretrain_train, Z_pretrain_val, Z_pretrain_test, y_data_all, y_data_train, y_data_val, y_data_test)

 


print ('done')