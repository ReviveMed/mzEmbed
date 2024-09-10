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


#importing Jonha's funtions 
from models import get_model, Binary_Head, Dummy_Head, MultiClass_Head, MultiHead, Regression_Head, Cox_Head, get_encoder

from models_VAE import VAE


from utils_neptune import check_neptune_existance, start_neptune_run, convert_neptune_kwargs, neptunize_dict_keys, get_latest_dataset



#setting the neptune api token
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='



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




def get_pretrain_encoder_from_modelID(model_id, path_to_proccessed_data, output_path, ml_code_path, setup_id='pretrain', project_id = 'revivemed/RCC', X_size=None, encoder_kind='VAE', latent_passes=10):

    #changing the directory to the ml code path
    os.chdir(ml_code_path)
    
    
    # Step 0: get the input data
    (X_data_all, y_data_all, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test) = get_pretrain_input_data(path_to_proccessed_data)

    # Step 1: Connect to Neptune
    run = neptune.init_run(project='revivemed/RCC', api_token= NEPTUNE_API_TOKEN, with_id=model_id)

    # Step 2: Retrieve a specific model
    model_data = run.fetch()
    #input_size = model_data.get('input_size', None)
    input_size = run['input_size'].fetch()

    #get the parameters of the model
    load_kwargs = run['pretrain/original_kwargs'].fetch()
    load_kwargs = convert_neptune_kwargs(load_kwargs)


    # getting encoder kwargs
    encoder_kwargs = load_kwargs.get('encoder_kwargs', {})
    fit_kwargs = load_kwargs.get('fit_kwargs', {})

    latent_size = encoder_kwargs.get('latent_size', 8)

    if ('hidden_size_mult' in encoder_kwargs) and (encoder_kwargs['hidden_size_mult'] > 0):
        encoder_kwargs['hidden_size'] = int(encoder_kwargs['hidden_size_mult']*latent_size)
        # remove the hidden_size_mult key
        encoder_kwargs.pop('hidden_size_mult')
        hidden_size = encoder_kwargs['hidden_size']
    else:
        hidden_size = encoder_kwargs.get('hidden_size', -1)

    if input_size is None:
        try:
            input_size = run['input_size'].fetch()
        except NeptuneException:
            input_size = X_size
            run['input_size'] = input_size
    if X_size is not None:
        assert input_size == X_size



    # Load the encoder state dict
    
    model_local_path = f'{output_path}/{model_id}'
    model_encoder_file = f'{model_local_path}/{model_id}_encoder_state_dict.pth'

    os.makedirs(f'{model_local_path}',  exist_ok=True)
    
    #download the encoder state dict
    if not os.path.exists(model_encoder_file):
       # Download the encoder state dict
        run['pretrain/models/encoder_state_dict'].download(model_encoder_file)
    else: 
        print(f"Encoder state dict already exists at {model_encoder_file}")
    

    # stop the run
    run.stop()
    
    #Create the Encoder Models
    # Load the encoder
    encoder = VAE(input_size=input_size, **encoder_kwargs)

    encoder_state_dict = torch.load(model_encoder_file)
    encoder.load_state_dict(encoder_state_dict)


    #getting the latent sapce by avergaing 
    # Latent averaging with fine-tuning of the last encoder layer

    # Getting the latent space by averaging 
    Z_all = []
    Z_train = []
    Z_val = []
    Z_test = []

    for _ in range(latent_passes):  # Generate multiple latent representations
        
        # Convert DataFrames to Tensors
        latent_rep_all = torch.tensor(generate_latent_space(X_data_all, encoder).values, dtype=torch.float32)
        Z_all.append(latent_rep_all)

        latent_rep_train = torch.tensor(generate_latent_space(X_data_train, encoder).values, dtype=torch.float32)
        Z_train.append(latent_rep_train)

        latent_rep_val = torch.tensor(generate_latent_space(X_data_val, encoder).values, dtype=torch.float32)
        Z_val.append(latent_rep_val)

        latent_rep_test = torch.tensor(generate_latent_space(X_data_test, encoder).values, dtype=torch.float32)
        Z_test.append(latent_rep_test)

    # Averaging the latent spaces
    Z_all = torch.mean(torch.stack(Z_all), dim=0)
    Z_train = torch.mean(torch.stack(Z_train), dim=0)
    Z_val = torch.mean(torch.stack(Z_val), dim=0)
    Z_test = torch.mean(torch.stack(Z_test), dim=0)

    #now converting the latent space to dataframe
    Z_all = pd.DataFrame(Z_all, index=X_data_all.index)
    Z_train = pd.DataFrame(Z_train, index=X_data_train.index)
    Z_val = pd.DataFrame(Z_val, index=X_data_val.index)
    Z_test = pd.DataFrame(Z_test, index=X_data_test.index)


    #save to csv
    Z_all.to_csv(f'{model_local_path}/Z_all_{model_id}.csv')
    Z_train.to_csv(f'{model_local_path}/Z_train_{model_id}.csv')
    Z_val.to_csv(f'{model_local_path}/Z_val_{model_id}.csv')
    Z_test.to_csv(f'{model_local_path}/Z_test_{model_id}.csv')

    return(encoder, Z_all, Z_train, Z_val, Z_test, y_data_all, y_data_train, y_data_val, y_data_test)








