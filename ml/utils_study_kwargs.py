
import os
import numpy as np
import pandas as pd
import torch
import optuna

from train import CompoundDataset, get_end_state_eval_funcs



BASE_DIR = '/DATA'


def get_kwargs(trial,study_kind,goal_col,result_dir=None):

    study_name = goal_col + study_kind
    if result_dir is None:
        result_dir = f'{BASE_DIR}/trials/{study_name}'

    if 'march13' in study_kind:
        return _get_study_march13(trial,goal_col,result_dir=result_dir)

    if 'march14' in study_kind:
        kwargs = _get_study_march14(trial,goal_col,result_dir=result_dir)
        
    elif 'march15' in study_kind:
        kwargs = _get_study_march15(trial,goal_col,result_dir=result_dir)        

    elif 'pretrain_1' in study_kind:
        kwargs = _get_study_march15(trial,goal_col,result_dir=result_dir)    

    else:
        raise ValueError(f'Unrecognized study_kind: {study_kind}')

    if '_S_' in study_kind:
        kwargs = _get_fewer_peaks(kwargs)
    if '_Adv' in study_kind:
        kwargs = _get_adv_kwargs(kwargs,trial)
    if '_Clas' in study_kind:
        kwargs = _get_clf_kwargs(kwargs,trial)

    return kwargs



####################################################################
####################################################################    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
####################################################################
####################################################################    


def _get_fewer_peaks(kwargs):
    kwargs['finetune_peak_freq_th'] = 0.9
    kwargs['overall_peak_freq_th'] = 0
    kwargs['pretrain_peak_freq_th'] = 0.3
    kwargs['finetune_var_q_th'] = 0.05
    return kwargs

def _get_adv_kwargs(kwargs,trial):
    kwargs['pretrain_adv_kind'] = 'MultiClassClassifier'
    kwargs['pretrain_adv_kwargs'] = {
        'hidden_size': 4,
        'num_hidden_layers': 1,
        'dropout_rate': 0,
        'activation': 'leakyrelu',
        'use_batch_norm': False,
        'num_classes': 19,
        }
    kwargs['pretrain_kwargs']['adversary_weight'] = trial.suggest_float('pretrain_adv_weight', 0.01, 100, log=True)
    return kwargs

def _get_clf_kwargs(kwargs,trial):
    kwargs['pretrain_head_kind'] = 'MultiClassClassifier'
    kwargs['pretrain_head_kwargs'] = {
        'hidden_size': 4,
        'num_hidden_layers': 1,
        'dropout_rate': 0,
        'activation': 'leakyrelu',
        'use_batch_norm': False,
        'num_classes': 4,
        }
    kwargs['pretrain_kwargs']['head_weight'] = trial.suggest_float('pretrain_head_weight', 0.01, 100, log=True)
    return kwargs


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
####################################################################
####################################################################    



####################################################################
####################################################################    

def _get_study_march15(trial, goal_col, result_dir):

    # activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    # activation = trial.suggest_categorical('activation', ['relu', 'leakyrelu','elu'])
    
    activation = 'leakyrelu'
    encoder_kind = 'AE'
    # encoder_kind = trial.suggest_categorical('encoder_kind', ['AE', 'VAE']),

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 60, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.5, step=0.1),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.5, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
        }


    finetune_early_stopping_patience = trial.suggest_int('finetune_early_stopping_patience', 0, 20, step=10)

    if finetune_early_stopping_patience == 0:
        finetune_val_frac = 0
    else:
        finetune_val_frac = trial.suggest_float('finetune_val_frac', 0.15, 0.25, step=0.05)

    use_pretrain_weight_decay = trial.suggest_categorical('use_pretrain_weight_decay', [True, False])
    use_finetune_weight_decay = trial.suggest_categorical('use_finetune_weight_decay', [True, False])
    use_pretrain_elasticnet_reg = trial.suggest_categorical('use_elasticnet_reg', [True, False])
    use_finetune_elasticnet_reg = trial.suggest_categorical('use_elasticnet_reg', [True, False])
    if use_pretrain_weight_decay:
        pretrain_weight_decay = trial.suggest_float('pretrain_weight_decay', 0.0001, 0.1, log=True)
    else:
        pretrain_weight_decay = 0

    if use_finetune_weight_decay:
        finetune_weight_decay = trial.suggest_float('finetune_weight_decay', 0.0001, 0.1, log=True)
    else:
        finetune_weight_decay = 0

    if use_pretrain_elasticnet_reg:
        pretrain_l1_reg_weight = trial.suggest_float('pretrain_l1_reg_weight', 0.0001, 0.1, log=True)
        pretrain_l2_reg_weight = trial.suggest_float('pretrain_l2_reg_weight', 0.0001, 0.1, log=True)
    else:
        pretrain_l1_reg_weight = 0
        pretrain_l2_reg_weight = 0

    if use_finetune_elasticnet_reg:
        finetune_l1_reg_weight = trial.suggest_float('finetune_l1_reg_weight', 0.0001, 0.1, log=True)
        finetune_l2_reg_weight = trial.suggest_float('finetune_l2_reg_weight', 0.0001, 0.1, log=True)
    else:
        finetune_l1_reg_weight = 0
        finetune_l2_reg_weight = 0

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        # 'num_folds': 50,
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'hold_out_iter': 10,
        'finetune_peak_freq_th': 0,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0,
        'finetune_var_q_th': 0,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'NA',
        'pretrain_head_kwargs' : {},
        
        'pretrain_adv_kind': 'NA',
        'pretrain_adv_kwargs' : {},

        'pretrain_kwargs': {
            # 'num_epochs': trial.suggest_int('pretrain_epochs', 10, 100,log=True),
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 500,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'weight_decay': pretrain_weight_decay,
            'l1_reg_weight': pretrain_l1_reg_weight,
            'l2_reg_weight': pretrain_l2_reg_weight,
            'encoder_weight': 1,
            'head_weight': 0,
            'adversary_weight': 0,
            'noise_factor': trial.suggest_float('pretrain_noise_factor', 0, 0.25, step=0.05),
            'early_stopping_patience': trial.suggest_int('pretrain_early_stopping_patience', 0, 50, step=10),
            'loss_avg_beta': -1,
            # 'loss_avg_beta': 0,
            # 'end_state_eval_funcs': {},
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##
        'finetune_val_frac': finetune_val_frac,
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
            # 'num_epochs': 2,
            # 'num_epochs': trial.suggest_int('finetune_epochs', 5, 50,log=True),
            'num_epochs': trial.suggest_int('finetune_epochs', 10, 250,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'weight_decay': finetune_weight_decay,
            'l1_reg_weight': finetune_l1_reg_weight,
            'l2_reg_weight': finetune_l2_reg_weight,
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': trial.suggest_float('finetune_noise_factor', 0, 0.25, step=0.05),
            'early_stopping_patience': finetune_early_stopping_patience,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
    return kwargs



####################################################################
####################################################################  




def _get_study_march14(trial, goal_col, result_dir):

    # activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    activation = 'leakyrelu'
    encoder_kind = 'AE'

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
        }

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        # 'num_folds': 50,
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'finetune_peak_freq_th': 0,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0,
        'finetune_var_q_th': 0,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'NA',
        'pretrain_head_kwargs' : {},
        
        'pretrain_adv_kind': 'NA',
        'pretrain_adv_kwargs' : {},

        'pretrain_kwargs': {
            # 'num_epochs': trial.suggest_int('pretrain_epochs', 10, 100,log=True),
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 500,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 1,
            'head_weight': 0,
            'adversary_weight': 0,
            'noise_factor': trial.suggest_float('pretrain_noise_factor', 0, 0.25, step=0.05),
            'early_stopping_patience': trial.suggest_int('pretrain_early_stopping_patience', 0, 25, step=5),
            'loss_avg_beta': -1,
            # 'loss_avg_beta': 0,
            # 'end_state_eval_funcs': {},
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0.15,
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
            # 'num_epochs': 2,
            # 'num_epochs': trial.suggest_int('finetune_epochs', 5, 50,log=True),
            'num_epochs': trial.suggest_int('finetune_epochs', 5, 250,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': trial.suggest_float('finetune_noise_factor', 0, 0.25, step=0.05),
            'early_stopping_patience': trial.suggest_int('finetune_early_stopping_patience', 0, 25, step=5),
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
    return kwargs


####################################################################
####################################################################


def _get_study_march13(trial,study_kind,goal_col,result_dir=None):


    if study_kind == '_study_march13_L_Clas':
        return _get_study_march13_L_Clas(trial,goal_col,result_dir=result_dir)

    elif study_kind == '_study_march13_L_Reg':
        return _get_study_march13_L_Reg(trial,goal_col,result_dir=result_dir)
    
    elif study_kind == '_study_march13_L_Adv':
        return _get_study_march13_L_Adv(trial,goal_col,result_dir=result_dir)

    elif study_kind == '_study_march13_S_Clas':
        return _get_study_march13_S_Clas(trial,goal_col,result_dir=result_dir)
    
    elif study_kind == '_study_march13_S_Reg':
        return _get_study_march13_S_Reg(trial,goal_col,result_dir=result_dir)
    
    elif study_kind == '_study_march13_S_Adv':
        return _get_study_march13_S_Adv(trial,goal_col,result_dir=result_dir)
    
    elif study_kind == '_study_march13_S_TGEM':
        return _get_study_march13_S_TGEM(trial,goal_col,result_dir=result_dir)

####################################################################
####################################################################

def _get_study_march13_L_Reg(trial,goal_col,result_dir):
    
    activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    encoder_kind = 'AE'

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': False, #trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
        }

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        # 'num_folds': 50,
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'finetune_peak_freq_th': 0,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0,
        'finetune_var_q_th': 0,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'NA',
        'pretrain_head_kwargs' : {},
        
        'pretrain_adv_kind': 'NA',
        'pretrain_adv_kwargs' : {},


        'pretrain_kwargs': {
            # 'num_epochs': trial.suggest_int('pretrain_epochs', 10, 100,log=True),
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 500,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 1,
            'head_weight': 0,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': 5,
            'loss_avg_beta': -1,
            # 'loss_avg_beta': 0,
            # 'end_state_eval_funcs': {},
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0,
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
            # 'num_epochs': 2,
            # 'num_epochs': trial.suggest_int('finetune_epochs', 5, 50,log=True),
            'num_epochs': trial.suggest_int('finetune_epochs', 5, 250,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': -1,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
    return kwargs






####################################################################
####################################################################

def _get_study_march13_S_Reg(trial,goal_col,result_dir):

    activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    encoder_kind = 'AE'

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': False, #trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
        }

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'finetune_peak_freq_th': 0.9,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0.3,
        'finetune_var_q_th': 0.1,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'NA',
        'pretrain_head_kwargs' : {},

        
        'pretrain_adv_kind': 'NA',
        'pretrain_adv_kwargs' : {},


        'pretrain_kwargs': {
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 500,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 1,
            'head_weight': 0,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': 5,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0,
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
            'num_epochs': trial.suggest_int('finetune_epochs', 5, 250,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': -1,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
    return kwargs    


####################################################################
####################################################################

def _get_study_march13_L_Clas(trial,goal_col,result_dir):


    activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    encoder_kind = 'AE'

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': False, #trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
        }

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'finetune_peak_freq_th': 0,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0,
        'finetune_var_q_th': 0,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'MultiClassClassifier',
        'pretrain_head_kwargs' : {
            'hidden_size': 4,
            'num_hidden_layers': 1,
            'dropout_rate': 0,
            'activation': activation,
            'use_batch_norm': False,
            'num_classes': 4,
            },
        
        'pretrain_adv_kind': 'NA',
        'pretrain_adv_kwargs' : {},

        'pretrain_kwargs': {
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 500,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 1,
            'head_weight': trial.suggest_float('pretrain_head_weight',0.1, 10, log=True),
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': 5,
            'loss_avg_beta': 0,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0,
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
            'num_epochs': trial.suggest_int('finetune_epochs', 5, 250,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': -1,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }

    return kwargs


####################################################################
####################################################################

def _get_study_march13_S_Clas(trial,goal_col,result_dir):

    activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    encoder_kind = 'AE'

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': False, #trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
        }

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        # 'num_folds': 50,
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'finetune_peak_freq_th': 0.9,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0.3,
        'finetune_var_q_th': 0.05,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'MultiClassClassifier',
        'pretrain_head_kwargs' : {
            'hidden_size': 4,
            'num_hidden_layers': 1,
            'dropout_rate': 0,
            'activation': activation,
            'use_batch_norm': False,
            'num_classes': 4,
            },
        
        'pretrain_adv_kind': 'NA',
        'pretrain_adv_kwargs' : {},

        'pretrain_kwargs': {
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 500,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 1,
            'head_weight': trial.suggest_float('pretrain_head_weight',0.1, 10, log=True),
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': 5,
            'loss_avg_beta': 0,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0,
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
            'num_epochs': trial.suggest_int('finetune_epochs', 5, 250,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': -1,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
    return kwargs


####################################################################
####################################################################

def _get_study_march13_L_Adv(trial,goal_col,result_dir):
    
    activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    # encoder_kind = 'TGEM_Encoder'
    # encoder_kind = trial.suggest_categorical('encoder_kind', ['AE', 'VAE'])
    encoder_kind = 'AE'

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': False, #trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
        }

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        # 'num_folds': 50,
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'finetune_peak_freq_th': 0,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0,
        'finetune_var_q_th': 0,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'NA',
        'pretrain_head_kwargs' : {},

        'pretrain_adv_kind': 'MultiClassClassifier',
        'pretrain_adv_kwargs' : {
            'hidden_size': 4,
            'num_hidden_layers': 1,
            'dropout_rate': 0,
            'activation': activation,
            'use_batch_norm': False,
            'num_classes': 18,
            },

        'pretrain_kwargs': {
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 500,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 1,
            'head_weight': 0,
            'adversary_weight': trial.suggest_float('pretrain_adv_weight', 0.1, 10, log=True),
            'noise_factor': 0,
            'early_stopping_patience': 5,
            'loss_avg_beta': 0,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0,
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
            'num_epochs': trial.suggest_int('finetune_epochs', 5, 250,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': -1,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
    return kwargs


####################################################################
####################################################################


def _get_study_march13_S_Adv(trial,goal_col,result_dir):

    activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    encoder_kind = 'AE'

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': False, #trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
        }

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        # 'num_folds': 50,
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'finetune_peak_freq_th': 0.9,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0.3,
        'finetune_var_q_th': 0.05,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'NA',
        'pretrain_head_kwargs' : {},

        'pretrain_adv_kind': 'MultiClassClassifier',
        'pretrain_adv_kwargs' : {
            'hidden_size': 4,
            'num_hidden_layers': 1,
            'dropout_rate': 0,
            'activation': activation,
            'use_batch_norm': False,
            'num_classes': 18,
            },

        'pretrain_kwargs': {
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 500,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 1,
            'head_weight': 0,
            'adversary_weight': trial.suggest_float('pretrain_adv_weight', 0.1, 10, log=True),
            'noise_factor': 0,
            'early_stopping_patience': 5,
            'loss_avg_beta': 0,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0,
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
            'num_epochs': trial.suggest_int('finetune_epochs', 5, 250,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': -1,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
    return kwargs



####################################################################
####################################################################

def _get_study_march13_S_TGEM(trial,goal_col,result_dir):

    activation = trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid'])
    encoder_kind = 'TGEM_Encoder'

    if encoder_kind == 'AE' or encoder_kind == 'VAE':
        latent_size = trial.suggest_int('latent_size', 4, 100, log=True)
        encoder_kwargs = {
            'activation': activation,
            'latent_size': latent_size,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'use_batch_norm': False, #trial.suggest_categorical('use_batch_norm', [True, False]),
            'hidden_size': round(1.5*latent_size), #trial.suggest_float('hidden_size_mult',1,2, step=0.1)*latent_size,
            }
    else:
        latent_size = -1
        encoder_kwargs = {
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.4, step=0.1),
            'n_head': trial.suggest_int('num_hidden_layers', 1, 5),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
        }

    kwargs = {
        ################
        ## General ##

        'save_dir': os.path.join(result_dir, f'trial_{trial.datetime_start}__{trial.number}'),
        'encoder_kind': encoder_kind,
        'encoder_kwargs': encoder_kwargs,
        'other_size': 1,
        'y_pretrain_cols': ['Cohort Label_encoded', 'Study ID_encoded'],
        'y_finetune_cols': [goal_col, 'Sex BINARY'],    
        # 'num_folds': 50,
        'num_folds': 30,
        'hold_out_str_list': ['Test'],
        'finetune_peak_freq_th': 0.9,
        'overall_peak_freq_th': 0,
        'pretrain_peak_freq_th': 0.3,
        'finetune_var_q_th': 0.05,
        'finetune_var_th': None,

        ################
        ## Pretrain ##

        'pretrain_val_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
        'pretrain_batch_size': 64,
        'pretrain_head_kind': 'MultiClassClassifier',
        'pretrain_head_kwargs' : {
            'hidden_size': 4,
            'num_hidden_layers': 1,
            'dropout_rate': 0,
            'activation': activation,
            'use_batch_norm': False,
            'num_classes': 4,
            },
        
        'pretrain_adv_kind': 'NA',
        'pretrain_adv_kwargs' : {},

        'pretrain_kwargs': {
            'num_epochs': trial.suggest_int('pretrain_epochs', 10, 100,log=True),
            'lr': trial.suggest_float('pretrain_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': 5,
            'loss_avg_beta': 0,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 5,
        },

        ################
        ## Finetune ##

        'finetune_val_frac': 0,
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
            'num_epochs': trial.suggest_int('finetune_epochs', 5, 50,log=True),
            'lr': trial.suggest_float('finetune_lr', 0.0001, 0.01, log=True),
            'encoder_weight': 0,
            'head_weight': 1,
            'adversary_weight': 0,
            'noise_factor': 0,
            'early_stopping_patience': -1,
            'loss_avg_beta': -1,
            'end_state_eval_funcs': get_end_state_eval_funcs(),
            'adversarial_mini_epochs': 20
        },
    }
    return kwargs
