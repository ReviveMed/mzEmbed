import torch
import pandas as pd
import numpy as np
import os
import json
from models import get_model
from train import CompoundDataset, train_compound_model, get_end_state_eval_funcs
from misc import get_dropbox_dir, download_data_dir, download_data_file, round_to_even, encode_df_col
from utils_gcp import upload_file_to_bucket, download_file_from_bucket, check_file_exists_in_bucket
import optuna
import logging
import sys

####################################################################################
# Main functions
####################################################################################

temp_dir = '/Users/jonaheaton/Desktop/temp'

DATA_DIR = f'{temp_dir}/data'
RESULT_DIR = f'{temp_dir}/results'
TRIAL_DIR = f'{temp_dir}/trials'

goal_col = 'Nivo Benefit BINARY'

def objective(trial):

    ############################
    # load the data
    splits_subdir = "['MSKCC', 'Treatment', 'Benefit'] finetune_folds" 
    trail_dir = TRIAL_DIR
    data_dir = DATA_DIR
    result_dir = RESULT_DIR


    X_data = pd.read_csv(f'{data_dir}/X.csv', index_col=0)
    y_data = pd.read_csv(f'{data_dir}/y.csv', index_col=0)
    splits = pd.read_csv(f'{data_dir}/{splits_subdir}/splits.csv', index_col=0)

    latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
    activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(trail_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': trial.suggest_categorical('encoder_kind', ['AE', 'VAE']),
        'encoder_kwargs': {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': False, #trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            # 'hidden_sizes': [trial.suggest_int('hidden_size', 2, 100, log=True) for _ in range(trial.suggest_int('num_layers', 1, 5))],
            },
        'other_size': 1,
        # 'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        # 'y_finetune_cols': ['Benefit_encoded', 'Sex_encoded'], 
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        'num_folds': 50,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.1,
        'pretrain_batch_size': 32,
        'pretrain_head_kind': 'NA',
        'pretrain_head_kwargs' : {
            },
        'pretrain_adv_kind': 'NA',
        'pretrain_adv_kwargs' : {
            },
        'pretrain_kwargs': {
            'num_epochs': trial.suggest_int('pretrain_epochs', 50, 1000,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 1,
            'head_weight': 0,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': 25,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': {},
            'adversarial_mini_epochs': 20
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0.1,
        'finetune_batch_size': 32,
        'finetune_head_kind': 'BinaryClassifier',
        'finetune_head_kwargs' : {
            'hidden_size': 4, 
            'num_hidden_layers': 0,
            'dropout_rate': 0,
            'activation': activation,
            'use_batch_norm': False,
            'num_classes': 2,
            },
        'finetune_adv_kind': 'NA',
        'finetune_adv_kwargs' : {
            },
        'finetune_kwargs': {
            'num_epochs': trial.suggest_int('finetune_epochs', 25, 500,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': 10,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
  


    ############################
    # load the kwargs


    save_dir = kwargs.get('save_dir', None)
    encoder_kind = kwargs.get('encoder_kind', 'AE')
    encoder_kwargs = kwargs.get('encoder_kwargs', {})
    other_size = kwargs.get('other_size', 1)

    y_pretrain_cols = kwargs.get('y_pretrain_cols', None)
    y_finetune_cols = kwargs.get('y_finetune_cols', None)
    num_folds = kwargs.get('num_folds', None)

    if num_folds is None:
        num_folds = splits.shape[1]

    if save_dir is None:
        raise ValueError('save_dir must be defined')
    os.makedirs(save_dir, exist_ok=True)

    ############################
    # Filter, preprocess Data
    X_pretrain = X_data
    y_pretrain = y_data[y_pretrain_cols].astype(float)

    X_finetune = X_data.loc[splits.index]
    y_finetune = y_data.loc[splits.index][y_finetune_cols].astype(float)

    input_size = X_pretrain.shape[1]

    ############################
    # Pretrain
    pretrain_batch_size = kwargs.get('pretrain_batch_size', 32)
    pretrain_val_frac = kwargs.get('pretrain_val_frac', 0)
    pretrain_head_kind = kwargs.get('pretrain_head_kind', 'NA')
    pretrain_adv_kind = kwargs.get('pretrain_adv_kind', 'NA')
    pretrain_head_kwargs = kwargs.get('pretrain_head_kwargs', {})
    pretrain_adv_kwargs = kwargs.get('pretrain_adv_kwargs', {})

    pretrain_kwargs = kwargs.get('pretrain_kwargs', {})
    pretrain_dir = os.path.join(save_dir, 'pretrain')
    os.makedirs(pretrain_dir, exist_ok=True)
    pretrain_kwargs['save_dir'] = pretrain_dir
    encoder = get_model(encoder_kind, input_size, **encoder_kwargs)
    
    pretrain_head = get_model(pretrain_head_kind, latent_size+other_size, **pretrain_head_kwargs)
    pretrain_adv = get_model(pretrain_adv_kind, latent_size, **pretrain_adv_kwargs)

    head_col = y_pretrain_cols[0]
    adv_col = y_pretrain_cols[1]
    pretrain_dataset = CompoundDataset(X_pretrain, y_pretrain[head_col], y_pretrain[adv_col])
    if pretrain_val_frac>0:
        train_size = int((1-pretrain_val_frac) * len(pretrain_dataset))
        val_size = len(pretrain_dataset) - train_size

        pretrain_dataset, preval_dataset = torch.utils.data.random_split(pretrain_dataset, [train_size, val_size])
        val_loader = torch.utils.data.DataLoader(preval_dataset, batch_size=pretrain_batch_size, shuffle=False)

    dataloaders = {
        'train': torch.utils.data.DataLoader(pretrain_dataset, batch_size=pretrain_batch_size, shuffle=True),
    }
    if pretrain_val_frac> 0:
        dataloaders['val'] = val_loader


    if pretrain_head_kind != 'NA':
        num_classes_pretrain_head = pretrain_dataset.get_num_classes_head()
        weights_pretrain_head = pretrain_dataset.get_class_weights_head()
        pretrain_head.define_loss(class_weight=weights_pretrain_head)


    if pretrain_adv_kind != 'NA':      
        num_classes_pretrain_adv = pretrain_dataset.get_num_classes_adv()
        weights_pretrain_adv = pretrain_dataset.get_class_weights_adv()
        pretrain_adv.define_loss(class_weight=weights_pretrain_adv)

    # pretrain the model
    _, _, _, pretrain_output = train_compound_model(dataloaders, encoder, pretrain_head, pretrain_adv, **pretrain_kwargs)

    # extract the pretraining objective
    if 'val' in pretrain_output['end_state_losses']:
        pretrain_result = pretrain_output['end_state_losses']['val']['reconstruction']
    else:
        pretrain_result = pretrain_output['end_state_losses']['train']['reconstruction']

    #TODO: Add user attributes related to the pretraining
    trial.set_user_attr('reconstruction loss', pretrain_result) 

    ############################
    # Finetune
    finetune_batch_size = kwargs.get('finetune_batch_size', 32)
    finetune_val_frac = kwargs.get('finetune_val_frac', 0)
    finetune_head_kind = kwargs.get('finetune_head_kind', 'MultiClassClassifier')
    finetune_adv_kind = kwargs.get('finetune_adv_kind', 'NA')
    finetune_head_kwargs = kwargs.get('finetune_head_kwargs', {})
    finetune_adv_kwargs = kwargs.get('finetune_adv_kwargs', {})
    finetune_kwargs = kwargs.get('finetune_kwargs', {})

    finetune_dir = os.path.join(save_dir, 'finetune')
    finetune_kwargs['save_dir'] = finetune_dir
    os.makedirs(finetune_dir, exist_ok=True)

    head_col = y_finetune_cols[0]
    adv_col = y_finetune_cols[1]

    finetune_head = get_model(finetune_head_kind, latent_size+other_size, **finetune_head_kwargs)
    finetune_adv = get_model(finetune_adv_kind, latent_size, **finetune_adv_kwargs)

    #### Start the CV loop
    finetune_results = []
    for n_fold in range(num_folds):
        X_train, y_train = X_finetune.loc[~splits.iloc[:,n_fold]], y_finetune.loc[~splits.iloc[:,n_fold]]
        X_test, y_test = X_finetune.loc[splits.iloc[:,n_fold]], y_finetune.loc[splits.iloc[:,n_fold]]

        train_dataset = CompoundDataset(X_train, y_train[head_col], y_train[adv_col])
        test_dataset = CompoundDataset(X_test, y_test[head_col], y_test[adv_col])
        
        if finetune_head_kind != 'NA':
            num_classes_finetune_head = train_dataset.get_num_classes_head()
            weights_finetune_head = train_dataset.get_class_weights_head()
            finetune_head.define_loss(class_weight=weights_finetune_head)

        if finetune_adv_kind != 'NA':    
            num_classes_finetune_adv = train_dataset.get_num_classes_adv()
            weights_finetune_adv = train_dataset.get_class_weights_adv()
            finetune_adv.define_loss(class_weight=weights_finetune_adv)


        # Split the training dataset into training and validation sets
        if finetune_val_frac>0:
            train_size = int((1-finetune_val_frac) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=finetune_batch_size, shuffle=False)


        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False)
        }
        
        if finetune_val_frac> 0:
            dataloaders['val'] = val_loader

        
        # Initialize the models
        finetune_head.reset_params()
        if finetune_adv is not None:
            finetune_adv.reset_params()
        encoder.reset_params()

        encoder.load_state_dict(torch.load(os.path.join(pretrain_dir, 'encoder.pth')))
 

        # Run the train and evaluation
        _, _, _, finetune_output = train_compound_model(dataloaders, encoder, finetune_head, finetune_adv, **finetune_kwargs)


        finetune_result = finetune_output['end_state_eval']['test']['head_auc']
        finetune_results.append(finetune_result)

        # Report intermediate objective value.
        trial.report(finetune_result, step=n_fold)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()


    ############################
    # Finetune from Random
    randtune_dir = os.path.join(save_dir, 'randtune')
    os.makedirs(randtune_dir, exist_ok=True)
    randtune_kwargs = finetune_kwargs
    randtune_kwargs['save_dir'] = randtune_dir
    
    rand_results = []
    #### Start the CV loop
    for n_fold in range(num_folds):
        X_train, y_train = X_finetune.loc[~splits.iloc[:,n_fold]], y_finetune.loc[~splits.iloc[:,n_fold]]
        X_test, y_test = X_finetune.loc[splits.iloc[:,n_fold]], y_finetune.loc[splits.iloc[:,n_fold]]


        train_dataset = CompoundDataset(X_train, y_train[head_col], y_train[adv_col])
        test_dataset = CompoundDataset(X_test, y_test[head_col], y_test[adv_col])
        
        if finetune_head_kind != 'NA':
            num_classes_finetune_head = train_dataset.get_num_classes_head()
            weights_finetune_head = train_dataset.get_class_weights_head()
            finetune_head.define_loss(class_weight=weights_finetune_head)

        if finetune_adv_kind != 'NA':    
            num_classes_finetune_adv = train_dataset.get_num_classes_adv()
            weights_finetune_adv = train_dataset.get_class_weights_adv()
            finetune_adv.define_loss(class_weight=weights_finetune_adv)

        # Split the training dataset into training and validation sets
        if finetune_val_frac>0:
            train_size = int((1-finetune_val_frac) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=finetune_batch_size, shuffle=False)


        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False)
        }
        
        if finetune_val_frac> 0:
            dataloaders['val'] = val_loader

        
        # Print the first 3 parameter values of the model
        # params = list(encoder.parameters())
        # print(params[0].data)
        # print(params[1].data)
        # print(params[2].data)

        # Initialize the models
        finetune_head.reset_params()
        encoder.reset_params() 
        if finetune_adv is not None:
            finetune_adv.reset_params()
        
        # Print the first 3 parameter values of the model
        # params = list(encoder.parameters())
        # print(params[0].data)
        # print(params[1].data)
        # print(params[2].data)

        encoder.load_state_dict(torch.load(os.path.join(pretrain_dir, 'encoder.pth')))
 

        # Run the train and evaluation
        _, _, _, rand_output = train_compound_model(dataloaders, encoder, finetune_head, finetune_adv, **randtune_kwargs)

        rand_result = rand_output['end_state_eval']['test']['head_auc']
        rand_results.append(rand_result)


    ############################
    # create a summary of the results
    result_dct = {
        'pretrain': pretrain_result,
        'finetune': finetune_results,
        'randtune': rand_results,
        # 'kwargs': kwargs
    }
    result_save_file = os.path.join(save_dir, 'result.json')
    with open(result_save_file, 'w') as f:
        json.dump(result_dct, f, indent=4)

    print('finetune:', finetune_results)
    print('randtune:', rand_results)

    obj_0 = np.mean(finetune_results)
    obj_1 = obj_0 - np.mean(rand_results)
    trial.set_user_attr('rand AUC', np.mean(rand_results))

    return obj_0



####################################################################################
# Main
####################################################################################
#TODO: ideally, it would be better to attach a volume directly to the container, but for now, we will just download the data
# and upload the results to GCP. that way we can share the database with other containers and things can run in parallel

# NOTE: Optuna Issue: Trial.report is not supported for multi-objective optimization. will be difficult to prune multi-objective
# there is some work on this in the optuna github, but it is not yet implemented
# https://github.com/optuna/optuna/issues/4578

if __name__ == '__main__':




    # download the data
    # data_url = 'https://www.dropbox.com/scl/fo/fa3n7sw8fgktnz6q91ffo/h?rlkey=edbdekkhuju5r88kkdo1garmn&dl=1'
    data_url = 'https://www.dropbox.com/scl/fo/d1yqlmyafvu8tzujlg4nk/h?rlkey=zrxfapacmgb6yxsmonw4yt986&dl=1'
    data_dir = DATA_DIR
    result_dir = RESULT_DIR
    trails_dir = TRIAL_DIR
    gcp_save_loc = 'March_6_Data'
    study_name = goal_col + '_study_0'


    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(trails_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(f'{data_dir}/X.csv'):
        download_data_dir(data_url, data_dir)


    mapper_file = f'{data_dir}/mappers.json'
    if not os.path.exists(mapper_file):
        mapper_dict = {}
    else:
        with open(mapper_file, 'r') as f:
            mapper_dict = json.load(f)

    # if not os.path.exists('data/y_encoded.csv'):
    y = pd.read_csv(f'{data_dir}/y.csv', index_col=0)
    
    if 'Sex BINARY' not in y.columns:
        y, mapper = encode_df_col(y, 'Sex', suffix=' BINARY')
        mapper_dict['Sex'] = mapper

    if 'Cohort Label_encoded' not in y.columns:
        y, mapper = encode_df_col(y, 'Cohort Label')
        mapper_dict['Cohort Label'] = mapper

    if 'Study ID_encoded' not in y.columns:
        y, mapper = encode_df_col(y, 'Study ID')
        mapper_dict['Study ID'] = mapper


    with open(f'{data_dir}/mappers.json', 'w') as f:
        json.dump(mapper_dict, f, indent=4)

    y.to_csv(f'{data_dir}/y.csv')


    # Set up logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # save the study in a sqlite database located in result_dir
    storage_loc = f'{result_dir}/{study_name}.db'
    if not os.path.exists(storage_loc):
        print("checking if study exists on GCP")
        if check_file_exists_in_bucket(gcp_save_loc, f'{study_name}.db'):
            print("downloading study from GCP")
            download_file_from_bucket(gcp_save_loc, f'{study_name}.db', local_path=result_dir)

    storage_name = f'sqlite:///{storage_loc}'



    # Create a study object and optimize the objective function
    # study = optuna.create_study(directions=["maximize" , "maximize"], 
    study = optuna.create_study(direction="maximize",
                                study_name=study_name, 
                                storage=storage_name, 
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=5))
    
    try:
        study.optimize(objective, n_trials=50)
    except Exception as e:
        print(e)
    # finally:


    # we need to upload the results to GCP
    upload_file_to_bucket(storage_loc, gcp_save_loc, verbose=True)

    # optuna-dashboard sqlite:///study_3.db