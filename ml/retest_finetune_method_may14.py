
from setup3 import setup_neptune_run
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune
import os
import pandas as pd
import numpy as np
import optuna
import json
from prep_study import add_runs_to_study, reuse_run, convert_neptune_kwargs, \
    make_kwargs, convert_distributions_to_suggestion, convert_model_kwargs_list_to_dict,\
    round_kwargs_to_sig




kwargs = {}
kwargs['train_kwargs'] = {}
kwargs['encoder_kwargs'] = {}
kwargs['load_model_from_run_id'] = 'RCC-2925'
kwargs['overwrite_existing_kwargs'] = True
kwargs['load_encoder_loc'] = 'pretrain'
kwargs['load_model_loc'] = False
kwargs['X_filename'] = 'X_finetune'
kwargs['y_filename'] = 'y_finetune'
kwargs['eval_name'] = 'test2'
kwargs['train_name'] = 'trainval2'
kwargs['run_training'] = True
kwargs['run_evaluation'] = True
kwargs['save_latent_space'] = True
kwargs['plot_latent_space'] = ''
kwargs['plot_latent_space_cols'] = []
kwargs['y_head_cols'] = ['OS','OS_Event']
kwargs['y_adv_cols'] = []
kwargs['upload_models_to_neptune'] = False
kwargs['num_repeats'] = 2


kwargs['head_kwargs_dict'] = {}
kwargs['adv_kwargs_dict'] = {}
kwargs['head_kwargs_list'] = [
    {
            'kind': 'Cox',
            'name': 'OS',
            'weight': 1.0,
            'y_idx': [0,1],
            'hidden_size': 4,
            'num_hidden_layers': 0,
            'dropout_rate': 0,
            'activation': 'leakyrelu',
            'use_batch_norm': False,
            }]


kwargs['encoder_kwargs']['dropout_rate'] = 0.0


# kwargs['train_kwargs']['num_epochs'] = 20
kwargs['train_kwargs']['early_stopping_patience'] = 0
kwargs['holdout_frac'] = 0
kwargs['train_kwargs']['head_weight'] = 1
kwargs['train_kwargs']['clip_grads_with_norm'] = False
kwargs['train_kwargs']['encoder_weight'] = 0
kwargs['train_kwargs']['adversary_weight'] = 0
kwargs['train_kwargs']['learning_rate'] = 0.0005414220374797538
# kwargs['train_kwargs']['learning_rate'] = 0.0001
kwargs['train_kwargs']['l2_reg_weight'] = 0
kwargs['train_kwargs']['l1_reg_weight'] = 0
kwargs['train_kwargs']['noise_factor'] =0.2
kwargs['train_kwargs']['weight_decay'] = 0
kwargs['train_kwargs']['adversarial_mini_epochs'] = 5
kwargs['train_kwargs']['adversarial_start_epoch'] = 10
kwargs['run_evaluation'] = True
kwargs['eval_kwargs'] = {}
kwargs['eval_kwargs']['sklearn_models'] = {}
kwargs['train_kwargs']['num_epochs'] = 71 

data_dir = 'DATA'
kwargs = convert_model_kwargs_list_to_dict(kwargs)

setup_id = 'RCC-2925_finetune'

_ = setup_neptune_run(data_dir,setup_id=setup_id,**kwargs)