import torch
import pandas as pd
import numpy as np
import os
import json
from models import get_model
from train import CompoundDataset, train_compound_model, get_end_state_eval_funcs
from misc import get_dropbox_dir, download_data_dir, download_data_file, round_to_even, encode_df_col, save_json
from utils_gcp import upload_file_to_bucket, download_file_from_bucket, check_file_exists_in_bucket, upload_path_to_bucket
import optuna
import logging
import sys
import shutil
from viz import generate_latent_space, generate_umap_embedding, generate_pca_embedding
from utils_study_kwargs import get_kwargs
from create_optuna_table import process_top_trials

####################################################################################
# Main functions
####################################################################################

# temp_dir = '/Users/jonaheaton/Desktop/temp'
BASE_DIR = '/DATA'
DATA_DIR = f'{BASE_DIR}/data'
TRIAL_DIR = f'{BASE_DIR}/trials'
storage_name = 'optuna'
USE_WEBAPP_DB = True
SAVE_TRIALS = True
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

GOAL_COL= None
STUDY_KIND = None
DEBUG = False

def objective(trial):


    if GOAL_COL is None:
        raise ValueError('GOAL_COL must be defined')
    if STUDY_KIND is None:
        raise ValueError('STUDY_KIND must be defined')

    ########################################################
    # load the data
    splits_subdir = f"{GOAL_COL} predefined_val" 
    data_dir = DATA_DIR
    study_name = trial.study.study_name
    result_dir = f'{BASE_DIR}/trials/{study_name}'
    os.makedirs(result_dir, exist_ok=True)

    X_data = pd.read_csv(f'{data_dir}/X.csv', index_col=0)
    y_data = pd.read_csv(f'{data_dir}/y.csv', index_col=0)
    
    # Get the pretrain, finetune cols
    pretrain_files = y_data[y_data['Set'] == 'Pretrain'].index.to_list()
    finetune_files = y_data[y_data['Set'] == 'Finetune'].index.to_list() # should be the same as splits

    splits_file_path = f'{data_dir}/{splits_subdir}/splits.csv'
    if os.path.exists(splits_file_path):
        splits = pd.read_csv(f'{data_dir}/{splits_subdir}/splits.csv', index_col=0)
    else:
        print('WARNING: Missing splits file')
        splits= None

    nan_data = pd.read_csv(f'{data_dir}/nans.csv', index_col=0)

    assert GOAL_COL in y_data.columns, f'{GOAL_COL} not in y_data.columns'


    ########################################################
    # Specify the model arcitecture and hyperparameters

    kwargs = get_kwargs(trial,
                        study_kind=STUDY_KIND,
                        goal_col=GOAL_COL,
                        result_dir=result_dir)
  
    if DEBUG:
        print('##############################################')
        print('DEBUG MODE')
        print('##############################################')
        print(kwargs)
        kwargs['num_folds'] = 1
        kwargs['pretrain_kwargs']['num_epochs'] = 1
        kwargs['finetune_kwargs']['num_epochs'] = 1
        trial.set_user_attr('DEBUG', True)


    ########################################################
    # Set Up

    save_dir = kwargs.get('save_dir', None)
    encoder_kind = kwargs.get('encoder_kind', 'AE')
    encoder_kwargs = kwargs.get('encoder_kwargs', {})
    other_size = kwargs.get('other_size', 1)
    latent_size = encoder_kwargs.get('latent_size', -1)

    y_pretrain_cols = kwargs.get('y_pretrain_cols', None)
    y_finetune_cols = kwargs.get('y_finetune_cols', None)
    num_folds = kwargs.get('num_folds', None)
    hold_out_str_list = kwargs.get('hold_out_str_list', [])

    if num_folds is None:
        num_folds = 5 #splits.shape[1]

    if save_dir is None:
        raise ValueError('save_dir must be defined')
    os.makedirs(save_dir, exist_ok=True)


    if splits is None:
        print('Generate Splits on the fly')
        splits = pd.DataFrame(index=finetune_files, columns=range(num_folds), dtype=int)
        splits = splits.fillna(0)
        for i in range(num_folds):
            splits[splits.sample(frac=0.2), i] = 1

    ############################
    # Filter the Peaks
            
    finetune_peak_freq_th = kwargs.get('finetune_peak_freq_th', 0)
    pretrain_peak_freq_th = kwargs.get('pretrain_peak_freq_th', 0)
    overall_peak_freq_th = kwargs.get('overall_peak_freq_th', 0)
    finetune_var_q_th = kwargs.get('finetune_var_q_th', 0)
    finetune_var_th = kwargs.get('finetune_var_th', None)

    finetune_peak_freq = 1- nan_data.loc[finetune_files].sum(axis=0)/len(finetune_files)
    pretrain_peak_freq = 1- nan_data.loc[pretrain_files].sum(axis=0)/len(pretrain_files)
    overall_peak_freq = 1- nan_data.sum(axis=0)/nan_data.shape[0]
    finetune_var = X_data.loc[finetune_files].var(axis=0)

    if (finetune_var_th is None) and (finetune_var_q_th > 0):
        finetune_var_th = finetune_var.quantile(finetune_var_q_th)
        print('finetune_var_th:', finetune_var_th)
    elif finetune_var_th is None:
        finetune_var_th = 0
     

    peak_filt_df = pd.DataFrame({
            'finetune_peak_freq': finetune_peak_freq,
            'pretrain_peak_freq': pretrain_peak_freq,
            'overall_peak_freq': overall_peak_freq,
            'finetune_var': finetune_var,
        }, index=X_data.columns)
    
    
    chosen_feats = peak_filt_df[  (peak_filt_df['finetune_peak_freq'] >= finetune_peak_freq_th)
                                & (peak_filt_df['pretrain_peak_freq'] >= pretrain_peak_freq_th)
                                & (peak_filt_df['overall_peak_freq'] >= overall_peak_freq_th)
                                & (peak_filt_df['finetune_var'] >= finetune_var_th)
                            ].index.to_list() 
    peak_filt_df['chosen'] = False
    peak_filt_df.loc[chosen_feats, 'chosen'] = True
    # peak_filt_df.to_csv(os.path.join(save_dir, 'peak_filt_df.csv'))


    input_size = len(chosen_feats)
    if latent_size < 0:
        print('WARNING: latent size not defined, setting to input size')
        latent_size = input_size
    print('input size:', input_size)
    if input_size < 5:
        raise ValueError('Not enough peaks to train the model')
    
    X_data = X_data[chosen_feats].copy()
    trial.set_user_attr('number of input peaks', input_size)

    # save the kwargs to json
    save_json(kwargs, os.path.join(save_dir, 'kwargs.json'))

    result_dct  = {}
    result_dct['kwargs'] = kwargs

    ############################
    # Filter, preprocess Data

    X_pretrain = X_data.loc[pretrain_files]
    y_pretrain = y_data.loc[pretrain_files][y_pretrain_cols].astype(float)

    X_finetune = X_data.loc[splits.index]
    y_finetune = y_data.loc[splits.index][y_finetune_cols].astype(float)

    input_size = X_pretrain.shape[1]

    ########################################################
    # Pretrain
    ########################################################

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

    if pretrain_head_kind != 'NA':
        num_classes_pretrain_head = pretrain_dataset.get_num_classes_head()
        weights_pretrain_head = pretrain_dataset.get_class_weights_head()
        pretrain_head.define_loss(class_weight=weights_pretrain_head)


    if pretrain_adv_kind != 'NA':      
        num_classes_pretrain_adv = pretrain_dataset.get_num_classes_adv()
        weights_pretrain_adv = pretrain_dataset.get_class_weights_adv()
        pretrain_adv.define_loss(class_weight=weights_pretrain_adv)

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


    # pretrain the model
    _, _, _, output = train_compound_model(dataloaders, encoder, pretrain_head, pretrain_adv, **pretrain_kwargs)

    # extract the pretraining objective
    if 'val' in output['end_state_losses']:
        pretrain_result = output['end_state_losses']['val']['reconstruction']
    else:
        pretrain_result = output['end_state_losses']['train']['reconstruction']

    #TODO: Add user attributes related to the pretraining
    trial.set_user_attr('reconstruction loss', output['end_state_losses']['val']['reconstruction'])
    trial.set_user_attr('pretrain Head AUC', output['end_state_eval']['val']['head_auc'])
    # trial.set_user_attr('pretrain Adv AUC', pretrain_output['end_state_eval']['val']['adv_auc'])
    if 'val LogisticRegression_auc' in output['sklearn_adversary_eval']:
        trial.set_user_attr('pretrain Adv AUC', output['sklearn_adversary_eval']['val LogisticRegression_auc'])

    result_dct['pretrain'] = pretrain_result


    if not DEBUG:
        try:
            Z = generate_latent_space(X_data, encoder)
            Z.to_csv(os.path.join(pretrain_dir, 'Z.csv'))

            Z_pca = generate_pca_embedding(Z)
            Z_pca.to_csv(os.path.join(pretrain_dir, 'Z_pca.csv'))

            Z_umap = generate_umap_embedding(Z)
            Z_umap.to_csv(os.path.join(pretrain_dir, 'Z_umap.csv'))
        except ValueError as e:
            print(e)
            print('Error when generating the embeddings')

    ########################################################
    # Finetune Set Up
    ########################################################

    finetune_batch_size = kwargs.get('finetune_batch_size', 32)
    finetune_val_frac = kwargs.get('finetune_val_frac', 0)
    finetune_head_kind = kwargs.get('finetune_head_kind', 'MultiClassClassifier')
    finetune_adv_kind = kwargs.get('finetune_adv_kind', 'NA')
    finetune_head_kwargs = kwargs.get('finetune_head_kwargs', {})
    finetune_adv_kwargs = kwargs.get('finetune_adv_kwargs', {})
    finetune_kwargs = kwargs.get('finetune_kwargs', {})
    early_stopping_patience = finetune_kwargs.get('early_stopping_patience', 0)

    head_col = y_finetune_cols[0]
    adv_col = y_finetune_cols[1]

    finetune_head = get_model(finetune_head_kind, latent_size+other_size, **finetune_head_kwargs)
    finetune_adv = get_model(finetune_adv_kind, latent_size, **finetune_adv_kwargs)


    ########################################################
    # Finetune from Pretrained Encoder
    ########################################################

    finetune_dir = os.path.join(save_dir, 'finetune')
    finetune_kwargs['save_dir'] = finetune_dir
    os.makedirs(finetune_dir, exist_ok=True)

    ############################
    # Start the CV loop
    ############################
    results = []
    for n_fold in range(num_folds):
        res = {}
        print('fold:', n_fold)
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
        if (finetune_val_frac>0) and (early_stopping_patience>0):
            train_size = int((1-finetune_val_frac) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=finetune_batch_size, shuffle=False)


        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False)
        }
        
        if (finetune_val_frac>0) and (early_stopping_patience>0):
            dataloaders['val'] = val_loader

        
        # Initialize the models
        finetune_head.reset_params()
        if finetune_adv is not None:
            finetune_adv.reset_params()
        encoder.reset_params()

        encoder.load_state_dict(torch.load(os.path.join(pretrain_dir, 'encoder.pth')))
 

        # Run the train and evaluation
        _, _, _, output = train_compound_model(dataloaders, encoder, finetune_head, finetune_adv, **finetune_kwargs)


        res['Val AUC'] = output['end_state_eval']['test']['head_auc']
        res['Train AUC'] = output['end_state_eval']['train']['head_auc']
        results.append(res)


    results = pd.DataFrame(results)
    trial.set_user_attr('finetune Val AUC (fit Train) avg', results['Val AUC'].mean())
    trial.set_user_attr('finetune Val AUC (fit Train) std', results['Val AUC'].std())
    trial.set_user_attr('finetune Train AUC (fit Train) avg', results['Train AUC'].mean())
    trial.set_user_attr('finetune Train AUC (fit Train) std', results['Train AUC'].std())


    obj_finetune_val = results['Val AUC'].mean()
    obj_finetune_train = results['Train AUC'].mean()

    ############################
    # Train on the full dataset, and evaluate on an independent test set,
    ############################
    if len(hold_out_str_list)> 0:
        X_train, y_train = X_finetune, y_finetune

        train_dataset = CompoundDataset(X_train, y_train[head_col], y_train[adv_col])
        
        if finetune_head_kind != 'NA':
            num_classes_finetune_head = train_dataset.get_num_classes_head()
            weights_finetune_head = train_dataset.get_class_weights_head()
            finetune_head.define_loss(class_weight=weights_finetune_head)

        if finetune_adv_kind != 'NA':    
            num_classes_finetune_adv = train_dataset.get_num_classes_adv()
            weights_finetune_adv = train_dataset.get_class_weights_adv()
            finetune_adv.define_loss(class_weight=weights_finetune_adv)


        # Split the training dataset into training and validation sets
        if (finetune_val_frac>0) and (early_stopping_patience>0):
            train_size = int((1-finetune_val_frac) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=finetune_batch_size, shuffle=False)


        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True),
        }
        
        if (finetune_val_frac>0) and (early_stopping_patience>0):
            dataloaders['val'] = val_loader

        for hold_out_str in hold_out_str_list:
            if hold_out_str == 'val':
                continue
            if hold_out_str not in y_data['Set'].unique():
                continue

            hold_out_files = y_data[y_data['Set'] == hold_out_str].index.to_list()
            X_test = X_data.loc[hold_out_files]
            y_test = y_data.loc[hold_out_files][y_finetune_cols].astype(float)
            test_dataset = CompoundDataset(X_test, y_test[head_col], y_test[adv_col])
            dataloaders[hold_out_str] = torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False)


        results = []
        for iter in range(10):
            res = {}


            # Initialize the models
            finetune_head.reset_params()
            if finetune_adv is not None:
                finetune_adv.reset_params()
            encoder.reset_params()

            encoder.load_state_dict(torch.load(os.path.join(pretrain_dir, 'encoder.pth')))


            # Run the train and evaluation
            _, _, _, output = train_compound_model(dataloaders, encoder, finetune_head, finetune_adv, **finetune_kwargs)
            res['Train AUC (fit Tr+Val)'] = output['end_state_eval']['train']['head_auc']


            for hold_out_str in hold_out_str_list:
                if hold_out_str == 'val':
                    continue
                if hold_out_str not in y_data['Set'].unique():
                    continue
                res[f'{hold_out_str} AUC (fit Tr+Val)'] = output['end_state_eval'][hold_out_str]['head_auc']
            
            results.append(res)


        results = pd.DataFrame(results)

        trial.set_user_attr('finetune Train AUC (fit Tr+Val) avg', results['Train AUC (fit Tr+Val)'].mean())
        trial.set_user_attr('finetune Train AUC (fit Tr+Val) std', results['Train AUC (fit Tr+Val)'].std())

        for hold_out_str in hold_out_str_list:
            if hold_out_str == 'val':
                continue
            if hold_out_str not in y_data['Set'].unique():
                continue
            trial.set_user_attr(f'finetune {hold_out_str} AUC (fit Tr+Val) avg', results[f'{hold_out_str} AUC (fit Tr+Val)'].mean())
            trial.set_user_attr(f'finetune {hold_out_str} AUC (fit Tr+Val) std', results[f'{hold_out_str} AUC (fit Tr+Val)'].std())



    ########################################################
    # Finetune from Random
    ########################################################

    randtune_dir = os.path.join(save_dir, 'randtune')
    os.makedirs(randtune_dir, exist_ok=True)
    randtune_kwargs = finetune_kwargs
    randtune_kwargs['save_dir'] = randtune_dir
    
    results = []

    ############################
    # Start the CV loop
    ############################
    for n_fold in range(num_folds):
        res = {}
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
        if (finetune_val_frac>0) and (early_stopping_patience>0):
            train_size = int((1-finetune_val_frac) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=finetune_batch_size, shuffle=False)


        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False)
        }
        
        if (finetune_val_frac>0) and (early_stopping_patience>0):
            dataloaders['val'] = val_loader

        # print(list(encoder.parameters())[0][:5])
        # Initialize the models
        finetune_head.reset_params()
        encoder.reset_params() 
        if finetune_adv is not None:
            finetune_adv.reset_params()
         
        # print('reset') 
        # print(list(encoder.parameters())[0][:5])
        # Run the train and evaluation
        _, _, _, output = train_compound_model(dataloaders, encoder, finetune_head, finetune_adv, **randtune_kwargs)

        res['Val AUC (fit Train)'] = output['end_state_eval']['test']['head_auc']
        res['Train AUC (fit Train)'] = output['end_state_eval']['train']['head_auc']
        results.append(res)

    
    results = pd.DataFrame(results)
    
    trial.set_user_attr('randinit Val AUC (fit Train) avg', results['Val AUC (fit Train)'].mean())
    trial.set_user_attr('randinit Val AUC (fit Train) std', results['Val AUC (fit Train)'].std())
    trial.set_user_attr('randinit Train AUC (fit Train) avg', results['Train AUC (fit Train)'].mean())
    trial.set_user_attr('randinit Train AUC (fit Train) std', results['Train AUC (fit Train)'].std())

    obj_randinit_val = results['Val AUC (fit Train)'].mean()
    obj_randinit_train = results['Train AUC (fit Train)'].mean()

    ############################
    # Train on the full dataset, and evaluate on an independent test set,
    ############################
    # if (hold_out_str) and (hold_out_str in y_data['Set'].unique()):
    if len(hold_out_str_list)> 0:
        X_train, y_train = X_finetune, y_finetune

        train_dataset = CompoundDataset(X_train, y_train[head_col], y_train[adv_col])
        
        if finetune_head_kind != 'NA':
            num_classes_finetune_head = train_dataset.get_num_classes_head()
            weights_finetune_head = train_dataset.get_class_weights_head()
            finetune_head.define_loss(class_weight=weights_finetune_head)

        if finetune_adv_kind != 'NA':    
            num_classes_finetune_adv = train_dataset.get_num_classes_adv()
            weights_finetune_adv = train_dataset.get_class_weights_adv()
            finetune_adv.define_loss(class_weight=weights_finetune_adv)

        # Split the training dataset into training and validation sets
        if (finetune_val_frac>0) and (early_stopping_patience>0):
            train_size = int((1-finetune_val_frac) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=finetune_batch_size, shuffle=False)


        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True),
        }
        
        if (finetune_val_frac>0) and (early_stopping_patience>0):
            dataloaders['val'] = val_loader


        for hold_out_str in hold_out_str_list:
            if hold_out_str == 'val':
                continue
            if hold_out_str not in y_data['Set'].unique():
                continue

            hold_out_files = y_data[y_data['Set'] == hold_out_str].index.to_list()
            X_test = X_data.loc[hold_out_files]
            y_test = y_data.loc[hold_out_files][y_finetune_cols].astype(float)
            test_dataset = CompoundDataset(X_test, y_test[head_col], y_test[adv_col])
            dataloaders[hold_out_str] = torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False)


        results = []
        for iter in range(10):
            res = {}
            # Initialize the models
            finetune_head.reset_params()
            encoder.reset_params() 
            if finetune_adv is not None:
                finetune_adv.reset_params()
            

            # Run the train and evaluation
            _, _, _, output = train_compound_model(dataloaders, encoder, finetune_head, finetune_adv, **randtune_kwargs)
            res['Train AUC (fit Tr+Val)'] = output['end_state_eval']['train']['head_auc']

            for hold_out_str in hold_out_str_list:
                if hold_out_str == 'val':
                    continue
                if hold_out_str not in y_data['Set'].unique():
                    continue
                res[f'{hold_out_str} AUC (fit Tr+Val)'] = output['end_state_eval'][hold_out_str]['head_auc']
            
            results.append(res)


        results = pd.DataFrame(results)

        trial.set_user_attr('randinit Train AUC (fit Tr+Val) avg', results['Train AUC (fit Tr+Val)'].mean())
        trial.set_user_attr('randinit Train AUC (fit Tr+Val) std', results['Train AUC (fit Tr+Val)'].std())

        for hold_out_str in hold_out_str_list:
            if hold_out_str == 'val':
                continue
            if hold_out_str not in y_data['Set'].unique():
                continue
            trial.set_user_attr(f'randinit {hold_out_str} AUC (fit Tr+Val) avg', results[f'{hold_out_str} AUC (fit Tr+Val)'].mean())
            trial.set_user_attr(f'randinit {hold_out_str} AUC (fit Tr+Val) std', results[f'{hold_out_str} AUC (fit Tr+Val)'].std())


            # result_dct[f'{hold_out_str} Rand AUC'] = finetune_output['end_state_eval'][hold_out_str]['head_auc']




    ############################
    # create a summary of the results
    ############################

    result_save_file = os.path.join(save_dir, 'result.json')
    save_json(result_dct, result_save_file)

    if obj_finetune_train < obj_finetune_val*.95:
        obj_finetune_train = 1

    return obj_finetune_val, obj_finetune_train



####################################################################################
# Main
####################################################################################
#TODO: ideally, it would be better to attach a volume directly to the container, but for now, we will just download the data
# and upload the results to GCP. that way we can share the database with other containers and things can run in parallel

# NOTE: Optuna Issue: Trial.report is not supported for multi-objective optimization. will be difficult to prune multi-objective
# there is some work on this in the optuna github, but it is not yet implemented
# https://github.com/optuna/optuna/issues/4578

if __name__ == '__main__':


    # get arguments from the command line
    if len(sys.argv) > 1:
        GOAL_COL = sys.argv[1]
    elif DEBUG:
        GOAL_COL = 'MSKCC BINARY'
    else:
        raise ValueError('GOAL_COL must be defined')

    if len(sys.argv) > 2:
        STUDY_KIND = sys.argv[2]
    elif DEBUG:
        STUDY_KIND = '_march14_S_Clas_0'  #'_study_march13_S_TGEM'
    else:
        raise ValueError('STUDY_KIND must be defined')

    if len(sys.argv) > 3:
        num_trials = int(sys.argv[3])
    elif DEBUG:
        num_trials = 1
    else:
        num_trials = 100

    # study_kind = '_study_march13_L_Clas'
    # goal_col = 'MSKCC BINARY'
    study_name = f'{GOAL_COL}{STUDY_KIND}'
    print('goal_col:', GOAL_COL)
    print('study_kind:', STUDY_KIND)
    print('study_name:', study_name)

    # download the data
    # data_url = 'https://www.dropbox.com/scl/fo/fa3n7sw8fgktnz6q91ffo/h?rlkey=edbdekkhuju5r88kkdo1garmn&dl=1'
    # data_url = 'https://www.dropbox.com/scl/fo/d1yqlmyafvu8tzujlg4nk/h?rlkey=zrxfapacmgb6yxsmonw4yt986&dl=1' # March 6 data
    data_url = 'https://www.dropbox.com/scl/fo/y4h9nyxccldu0bpc2be2f/h?rlkey=itkulmqraytn7gl86b2f1kxq6&dl=1' #March 12 data
    data_dir = DATA_DIR
    trials_dir = TRIAL_DIR
    gcp_save_loc = 'March_12_Data'



    os.makedirs(trials_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(f'{data_dir}/X.csv'):
        print('downloading data from dropbox')
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
    
    
    if USE_WEBAPP_DB:
        print('using webapp database')
        storage_name = WEBAPP_DB_LOC
    else:
        print('use local sqlite database downloaded from GCP bucket')
        # save the study in a sqlite database located in result_dir
        storage_loc = f'{data_dir}/{storage_name}.db'
        if not os.path.exists(storage_loc):
            print("checking if study exists on GCP")
            if check_file_exists_in_bucket(gcp_save_loc, f'{storage_name}.db'):
                print("downloading study from GCP")
                download_file_from_bucket(gcp_save_loc, f'{storage_name}.db', local_path=data_dir)

        storage_name = f'sqlite:///{storage_loc}'




    # Create a study object and optimize the objective function
    study = optuna.create_study(directions=["maximize" , "minimize"], 
                                study_name=study_name, 
                                storage=storage_name, 
                                load_if_exists=True)
    
    try:
        study.optimize(objective, n_trials=num_trials)#, show_progress_bar=True, timeout=3600*24*7, gc_after_trial=True)
    except Exception as e:
        print(e)
    # finally:


    if not USE_WEBAPP_DB:
        # we need to upload the results to GCP if not using the webapp DB
        upload_file_to_bucket(storage_loc, gcp_save_loc, verbose=True)

    if SAVE_TRIALS:
        upload_path_to_bucket(trials_dir, gcp_save_loc, verbose=True)
        # delete the data in the trials dir
        # os.system(f'rm -r {trials_dir}/*')
        shutil.rmtree(trials_dir)

    # optuna-dashboard sqlite:///study_3.db
        

    study_table_path = f'{data_dir}/{study_name}_table.csv'
    study_table = study.trials_dataframe()
    # study_table.to_csv('study_table.csv', index=False)
    study_table.to_csv(study_table_path, index=False)

    upload_file_to_bucket(study_table_path, gcp_save_loc)        


    # Create a summary of the top trials
    directions =  [study.directions[0].name for _ in range(len(study.directions))]
    min_cutoffs = {'values_0', 0.8}
    max_cutoffs = {}
    summary = process_top_trials(study_table, top_trial_perc=0.1, directions=directions,
                                    min_cutoffs=min_cutoffs,max_cutoffs=max_cutoffs)


    save_json(summary, f'{data_dir}/{study_name}_toptrials_summary.json')
    upload_file_to_bucket(f'{data_dir}/{study_name}_toptrials_summary.json', gcp_save_loc)