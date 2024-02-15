import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
import json
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from models import VAE, AE  # Assuming models.py contains a VAE class
# import time
from prep import ClassifierDataset,  PreTrainingDataset
# from functools import partial
from pretrain import run_train_autoencoder
from train import run_train_classifier


def round_to_even(n):
    if n % 2 == 0:
        return n
    else:
        return n + 1

def get_clean_batch_sz(len_dataset, org_batch_sz):
    # due to batch normalization, we want the batches to be as clean as possible
    curr_remainder = len_dataset % org_batch_sz
    max_iter = 100
    if org_batch_sz >= len_dataset:
        return org_batch_sz
    if (curr_remainder == 0) or (curr_remainder > org_batch_sz/2):
        return org_batch_sz
    else:
        batch_sz = org_batch_sz
        iter = 0
        while (curr_remainder != 0) and (curr_remainder < batch_sz/2) and (iter < max_iter):
            iter += 1
            if batch_sz < org_batch_sz/2:
                batch_sz = 2*org_batch_sz
            batch_sz -= 1
            curr_remainder = len_dataset % batch_sz
        if iter >= max_iter:
            print('Warning: Could not find a clean batch size')
        # print('old batch size:', org_batch_sz, 'new batch size:', batch_sz, 'remainder:', curr_remainder)
        return batch_sz


def full_train(data_dir, **kwargs):

    encoder_kind = kwargs.get('encoder_kind', 'AE')
    activation = kwargs.get('activation', 'leakyrelu')
    input_size = kwargs.get('input_size', None)
    latent_size = kwargs.get('latent_size', 64)

    encoder_hidden_size_mult = kwargs.get('encoder_hidden_size_mult', 1)
    if encoder_hidden_size_mult is None:
        encoder_hidden_size = kwargs.get('encoder_hidden_size', 64)
    else:
        encoder_hidden_size = int(encoder_hidden_size_mult * latent_size)

    encoder_hidden_layers = kwargs.get('encoder_hidden_layers', 1)
    encoder_batch_norm = kwargs.get('encoder_batch_norm', False)
    encoder_name = kwargs.get('encoder_name', 'encoder')
    encoder_act_on_latent = kwargs.get('encoder_act_on_latent', False)

    head_hidden_size = kwargs.get('head_hidden_size', 0)
    head_hidden_layers = kwargs.get('head_hidden_layers', 0)
    head_batch_norm = kwargs.get('head_batch_norm', False)
    head_num_classes = kwargs.get('head_num_classes', 2)
    head_name = kwargs.get('head_name', 'head')

    pretrain_dropout = kwargs.get('pretrain_dropout', 0)
    pretrain_epochs = kwargs.get('pretrain_epochs', 250)
    pretrain_early_stopping = kwargs.get('pretrain_early_stopping', -1)
    pretrain_lr = kwargs.get('pretrain_lr', 1e-3)
    pretrain_val_frac = kwargs.get('pretrain_val_frac', 0)
    pretrain_batch_size = kwargs.get('pretrain_batch_size', 64)
    pretrain_noise_injection = kwargs.get('pretrain_noise_injection', 0)
    # pretrain_load_num_workers = kwargs.get('pretrain_load_num_workers', 0)
    
    finetune_dropout = kwargs.get('finetune_dropout', 0.25)
    finetune_epochs = kwargs.get('finetune_epochs', 100)
    finetune_encoder_dropout = kwargs.get('finetune_encoder_dropout', finetune_dropout)
    finetune_early_stopping = kwargs.get('finetune_early_stopping', -1)
    finetune_lr = kwargs.get('finetune_lr', 1e-4)
    finetune_encoder_lr = kwargs.get('finetune_encoder_lr', finetune_lr)
    finetune_val_frac = kwargs.get('finetune_val_frac', 0)
    finetune_batch_size = kwargs.get('finetune_batch_size', 32)
    finetune_noise_injection = kwargs.get('finetune_noise_injection', 0)
    finetune_encoder_status = kwargs.get('finetune_encoder_status', 'finetune')

    finetune_label_encoder = kwargs.get('finetune_label_encoder', None)
    finetune_label_col = kwargs.get('finetune_label_col', 'MSKCC')
    finetune_n_subsets = kwargs.get('finetune_n_subsets', 5)

    verbose = kwargs.get('verbose', False)
    yesplot = kwargs.get('yesplot', False)
    save_dir = kwargs.get('save_dir', None)
    pretrain_save_dir = kwargs.get('pretrain_save_dir', save_dir)
    finetune_save_dir = kwargs.get('finetune_save_dir', save_dir)

    if finetune_label_encoder is None:
        finetune_label_encoder = {'FAVORABLE': 1, 'POOR': 0, 'INTERMEDIATE': np.nan}


    pretrained_encoder_path = os.path.join(pretrain_save_dir, encoder_name + '_model.pth')
    if finetune_encoder_status == 'random':
        print('skip pretraining since encoder inititalization should be random')

    else:
        # Load the pretrainng data
        pretrain_dataset = PreTrainingDataset(data_dir)
        if input_size is None:
            input_size = pretrain_dataset.X.shape[1]

        # create a validation set
        if (pretrain_val_frac > 0) and (pretrain_early_stopping > 0):
            val_size = int(pretrain_val_frac * len(pretrain_dataset))
            train_size = len(pretrain_dataset) - val_size
            train_dataset, val_dataset = random_split(pretrain_dataset, [train_size, val_size])
        else:
            train_dataset = pretrain_dataset
            val_dataset = None
        
        # create pretraning dataloaders
        #TODO figure out why num_workers is causing an error
        pretrain_batch_size = int(pretrain_batch_size)

        pretrain_dataloaders = {
            'train': DataLoader(train_dataset, 
                                batch_size= get_clean_batch_sz(len(train_dataset), pretrain_batch_size),
                                shuffle=True,num_workers=0),
        }
        if val_dataset is not None:
            pretrain_dataloaders['val'] = DataLoader(val_dataset, 
                                                    batch_size=get_clean_batch_sz(len(val_dataset), pretrain_batch_size),
                                                    shuffle=False)

        run_train_autoencoder(dataloaders=pretrain_dataloaders,
                            save_dir=pretrain_save_dir,
                            input_size=input_size,
                            latent_size=latent_size,
                            model_kind=encoder_kind,
                            activation=activation,
                            hidden_size = encoder_hidden_size,
                            dropout_rate = pretrain_dropout,
                            num_hidden_layers = encoder_hidden_layers,
                            use_batch_norm = encoder_batch_norm,
                            model_name = encoder_name,
                            num_epochs = pretrain_epochs,
                            learning_rate = pretrain_lr,
                            noise_factor = pretrain_noise_injection,
                            encoder_act_on_latent=encoder_act_on_latent,
                            early_stopping_patience = pretrain_early_stopping,
                            yesplot=yesplot,
                            verbose=verbose)


    # Load the finetuning data
    test_auc_list = []
    for subset_id in range(finetune_n_subsets):
        print('subset_id:', subset_id)
        finetune_dataset = ClassifierDataset(data_dir, 
                                    subset='train_{}'.format(subset_id),
                                    label_col=finetune_label_col,
                                    label_encoder=finetune_label_encoder)
        
        class_weights = 1 / torch.bincount(finetune_dataset.y.long())

        test_dataset = ClassifierDataset(data_dir, 
                                    subset='test_{}'.format(subset_id),
                                    label_col=finetune_label_col,
                                    label_encoder=finetune_label_encoder)
    
        # create a validation set
        # should this be stratified?
        if (finetune_val_frac > 0) and (finetune_early_stopping > 0):
            val_size = int(finetune_val_frac * len(finetune_dataset))
            train_size = len(finetune_dataset) - val_size
            train_dataset, val_dataset = random_split(finetune_dataset, [train_size, val_size])
            # train_dataset = finetune_dataset[train_dataset.indices]
            # val_dataset = finetune_dataset[val_dataset.indices]
        else:
            train_dataset = finetune_dataset
            val_dataset = None

        # create finetuning dataloaders
        finetune_batch_size = int(finetune_batch_size)
        finetune_dataloaders = {
            'train': DataLoader(train_dataset, 
                                batch_size= get_clean_batch_sz(len(train_dataset), finetune_batch_size),
                                shuffle=True),
            # 'train': DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True,num_workers=0),
            'test': DataLoader(test_dataset, 
                               batch_size=get_clean_batch_sz(len(test_dataset), finetune_batch_size),
                               shuffle=False),
            # 'test': DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False),
        }
        if val_dataset is not None:
            finetune_dataloaders['val'] = DataLoader(val_dataset, 
                                                     batch_size=get_clean_batch_sz(len(val_dataset), finetune_batch_size),
                                                     shuffle=False)

        model_name = head_name + '_{}'.format(subset_id)
        output_data = run_train_classifier(dataloaders=finetune_dataloaders,
                            save_dir=finetune_save_dir,
                            input_size=input_size,
                            latent_size=latent_size,
                            activation=activation,
                            hidden_size = head_hidden_size,
                            dropout_rate = finetune_dropout,
                            num_hidden_layers = head_hidden_layers,
                            use_batch_norm = head_batch_norm,
                            num_classes = head_num_classes,
                            class_weights=class_weights, #would be good idea to add this
                            model_name = model_name,
                            num_epochs = finetune_epochs,
                            learning_rate = finetune_lr,
                            early_stopping_patience = finetune_early_stopping,
                            noise_factor = finetune_noise_injection,
                            encoder_status = finetune_encoder_status,
                            encoder_kind=encoder_kind,
                            encoder_name=encoder_name,
                            encoder_learning_rate = finetune_encoder_lr,
                            encoder_hidden_size = encoder_hidden_size,
                            encoder_num_hidden_layers = encoder_hidden_layers,
                            encoder_dropout_rate = finetune_encoder_dropout,
                            encoder_activation = activation,
                            encoder_use_batch_norm = encoder_batch_norm,
                            encoder_act_on_latent = encoder_act_on_latent,
                            encoder_lr = finetune_encoder_lr,
                            pretrained_encoder_load_path = pretrained_encoder_path,
                            verbose=verbose,
                            yesplot=yesplot)

        test_auc_list.append(output_data['end_state_auroc']['test'])

    avg_cv_AUC = np.mean(test_auc_list)
    return avg_cv_AUC



#############
import optuna

def objective(trial):
    # Define the hyperparameter search space
    # trial.set_user_attr('activation', 'sigmoid')
    # trial.set_user_attr('head_hidden_size',0)
    # trial.set_user_attr('head_hidden_layers',0)
    # trial.set_user_attr('encoder_name','encoder')
    # trial.set_user_attr('pretrain_dropout',0)

    # currently looking at 15 hyperparameters to optimize

    search_space = {
        'encoder_kind': trial.suggest_categorical('encoder_kind', ['AE', 'VAE']),
        'activation': trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid']),
        'latent_size': round_to_even(trial.suggest_int('latent_size', 1, 160, log=True)),
        'encoder_hidden_size_mult': trial.suggest_float('encoder_hidden_size_mult', 1, 2.5, step=0.1),
        # 'encoder_hidden_layers': trial.suggest_int('encoder_hidden_layers', 0, 4),
        'encoder_hidden_layers': trial.suggest_int('encoder_hidden_layers', 1, 4),
        'encoder_batch_norm': trial.suggest_categorical('encoder_batch_norm', [True, False]),
        'encoder_name': 'encoder',
        'encoder_act_on_latent': True,
        'head_hidden_size': 0,
        'head_hidden_layers': 0,
        'head_batch_norm': False,
        'head_num_classes': 2,
        'head_name': 'head',
        'pretrain_dropout': 0,
        'pretrain_epochs': trial.suggest_int('pretrain_epochs', 100, 1000, step=100),
        'pretrain_early_stopping': trial.suggest_categorical('pretrain_early_stopping', [-1,20]),
        'pretrain_lr': trial.suggest_float('pretrain_lr', 1e-5, 1e-2, log=True),
        'pretrain_val_frac': 0.15,
        'pretrain_batch_size': 64,
        'pretrain_noise_injection': max(trial.suggest_float('pretrain_noise_injection', -0.05, 0.1, step=0.01),0),
        'finetune_dropout': 0,
        'finetune_epochs': trial.suggest_int('finetune_epochs', 25, 500, step=25),
        'finetune_encoder_dropout': trial.suggest_float('finetune_encoder_dropout', 0.0, 0.5, step=0.05),
        'finetune_early_stopping': trial.suggest_categorical('finetune_early_stopping', [-1,20]),
        'finetune_lr': trial.suggest_float('finetune_lr', 1e-5, 1e-2, log=True),
        'finetune_val_frac': 0.15,
        'finetune_batch_size': 32,
        'finetune_noise_injection': max(trial.suggest_float('finetune_noise_injection', -0.05, 0.25, step=0.01),0),
        'finetune_encoder_status': 'finetune',
        'finetune_label_col': 'MSKCC',
        'finetune_n_subsets': 30,
        'verbose': False,
        'yesplot': False,
    }

    if search_space['pretrain_early_stopping'] == -1:
        search_space['pretrain_val_frac'] = 0
    if search_space['finetune_early_stopping'] == -1:
        search_space['finetune_val_frac'] = 0
    search_space['finetune_encoder_lr'] = search_space['finetune_lr']

    # Run the full training process
    data_dir = '/Users/jonaheaton/Desktop/mskcc_study_feb13'

    # create a directory to using the trial id
    datetime_start_str = trial.datetime_start.strftime('%Y%m%d_%H%M%S')
    trial_id = datetime_start_str + '_' + str(trial.number).zfill(4) 
    save_dir = os.path.join(data_dir, 'models_feb13', trial_id)
    search_space['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    # save the search space to a json file
    with open(os.path.join(save_dir, 'search_space.json'), 'w') as f:
        json.dump(search_space, f, indent=4)
        
    avg_cv_AUC = full_train(data_dir, **search_space)
    print('avg_cv_AUC:', avg_cv_AUC)
    return avg_cv_AUC


if __name__ == "__main__":

    import logging
    import sys

    # Set up logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'mskcc_prediction_feb15'
    storage_name = 'sqlite:///{}.db'.format(study_name)

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=25)

    

