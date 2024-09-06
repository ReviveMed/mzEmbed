'''
    this function are used for retriving an encoder from the pre-trained models
    it then use the encoder to create the latent space to the model

    Args:
        model ID: the model ID of the pre-trained model
        path_to_proccessed_data: path to the processed data e.g. f'{homedir}/PROCESSED_DATA_2'
        output_path: the path to save the encoder, for example f'{homedir}
   

    return:
        encoder
        latent space of all data, training data, validation data, and test data
        y data of all data, training data, validation data, and test data


'''


##importing the required libraries

import os
import pandas as pd
import neptune
import torch
import neptune.new as neptune
import torch
import json


#importing Jonha's funtions 
from models import get_model, Binary_Head, Dummy_Head, MultiClass_Head, MultiHead, Regression_Head, Cox_Head, get_encoder

from viz import generate_latent_space, generate_umap_embedding, generate_pca_embedding

from utils_neptune import check_neptune_existance, start_neptune_run, convert_neptune_kwargs, neptunize_dict_keys, get_latest_dataset



#setting the neptune api token
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='



def get_finetune_encoder_from_modelID(model_id, path_to_proccessed_data, output_path, ml_code_path, model_neptune_path, setup_id='pretrain', project_id = 'revivemed/RCC', encoder_kind='VAE'):

    #changing the directory to the ml code path
    os.chdir(ml_code_path)
    
    
    # Step 0: get the input data
    (X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test) = get_input_data(path_to_proccessed_data)

    # Step 1: Connect to Neptune

    # Step 1: Connect to Neptune
    run = neptune.init_run(project='revivemed/RCC', api_token= NEPTUNE_API_TOKEN, with_id=model_id)

    # Step 2: Retrieve a specific model
    model_data = run.fetch()
    model_data.keys()

    # Step 3: Retrieve the model's hyperparameters
    #the model hyperparameters are saved in model_ID/params/fit_kwargs
    fit_kwargs = run['params/fit_kwargs'].fetch()
    fit_kwargs = convert_neptune_kwargs(fit_kwargs)

    # Step 4: Retrieve the model's encoder hyperparameters
    encoder_info_file=f'{output_path}/{model_id}_encoder_info.json'
    model_encoder_file = f'{output_path}/{model_id}_encoder_state.pth'
    
    

    # Check if the file exists and remove it if necessary
    if os.path.exists(model_encoder_file):
        os.remove(model_encoder_file)
        print(f"Deleted existing file at {model_encoder_file}")

    # Download the encoder state dict and encoder info
    run[f'{model_neptune_path}/models/encoder_state'].download(model_encoder_file)

    run[f'{model_neptune_path}/models/encoder_info'].download(encoder_info_file)

    run.stop()


    with open(encoder_info_file, 'r') as file:
        encoder_info = json.load(file)

    
    #Create the Encoder Models
    # Load the encoder
    encoder_kwargs = {**encoder_info, **fit_kwargs}

    encoder = get_model(encoder_kind, **encoder_kwargs)

    # Load the model and map it to CPU
    #encoder_state_dict = torch.load(model_encoder_file, map_location=torch.device('cpu') )

    # Load the model and map it to GPU
    encoder_state_dict = torch.load(model_encoder_file )

    encoder.load_state_dict(encoder_state_dict)
    encoder.to(torch.device('cpu'))  # Explicitly move the model to CPU


    
    #getting the latent sapce
    Z_train = generate_latent_space(X_data_train, encoder)
    Z_val = generate_latent_space(X_data_val, encoder)
    Z_test = generate_latent_space(X_data_test, encoder)


    #save to csv
    Z_train.to_csv(f'{output_path}/Z_train_{model_id}.csv')
    Z_val.to_csv(f'{output_path}/Z_val_{model_id}.csv')
    Z_test.to_csv(f'{output_path}/Z_test_{model_id}.csv')

    
    

    return(encoder, Z_train, Z_val, Z_test, y_data_train, y_data_val, y_data_test)







def get_finetune_input_data(data_location):

    #defining the input datasets
    
    finetune_X_train=f'{data_location}/X_Finetune_Discovery_Train.csv'
    finetune_y_train=f'{data_location}/y_Finetune_Discovery_Train.csv'

    finetune_X_val=f'{data_location}/X_Finetune_Discovery_Val.csv'
    finetune_y_val=f'{data_location}/y_Finetune_Discovery_Val.csv'

    finetune_X_test=f'{data_location}/X_Finetune_Test.csv'
    finetune_y_test=f'{data_location}/y_Finetune_Test.csv'

    
    #loading the data
    X_data_train = pd.read_csv(finetune_X_train, index_col=0)
    y_data_train = pd.read_csv(finetune_y_train, index_col=0)

    X_data_val = pd.read_csv(finetune_X_val, index_col=0)
    y_data_val = pd.read_csv(finetune_y_val, index_col=0)

    X_data_test = pd.read_csv(finetune_X_test, index_col=0)
    y_data_test = pd.read_csv(finetune_y_test, index_col=0)


    #returning the data
    return(X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)
