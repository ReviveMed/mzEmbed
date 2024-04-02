NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune
import os
import optuna
import json
from prep_study import add_runs_to_study, reuse_run, convert_neptune_kwargs, \
    objective_func1, make_kwargs, convert_distributions_to_suggestion, convert_model_kwargs_list_to_dict,\
    round_kwargs_to_sig
from setup2 import setup_neptune_run
from misc import download_data_dir
from utils_neptune import  get_run_id_list, check_neptune_existance
from sklearn.linear_model import LogisticRegression
import time
from neptune.exceptions import NeptuneException





data_dir = '/DATA2'
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(data_dir+'/X_pretrain_train.csv'):
    # data_url = 'https://www.dropbox.com/scl/fo/iy2emxpwa4rkr3ad7vhc2/h?rlkey=hvhfa3ny9dlavnooka3kwvu5v&dl=1' #march 22
    data_url = 'https://www.dropbox.com/scl/fo/2xr104jnz9qda7oemrwob/h?rlkey=wy7q95pj81qpgcn7zida2xjps&dl=1' #march 29
    download_data_dir(data_url, save_dir=data_dir)

# run_id = 'RCC-1296'
run_id = 'RCC-1216'
kwargs = {}

###############################
### Plot the latent space
kwargs['overwrite_existing_kwargs'] = True
kwargs['load_model_loc'] = 'pretrain'
kwargs['run_evaluation'] = False
kwargs['run_training'] = False
kwargs['save_latent_space'] = True
kwargs['plot_latent_space'] = 'sns' #'both'
kwargs['plot_latent_space_cols'] = ['Study ID','Cohort Label','is Pediatric']

# run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

kwargs['eval_name'] = 'train'
run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

kwargs['eval_name'] = 'test'
run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

kwargs['eval_name'] = 'val'
run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

###############################
### Run the finetune with random init

# kwargs['load_encoder_loc'] = 'pretrain'
# kwargs['X_filename'] = 'X_finetune'
# kwargs['y_filename'] = 'y_finetune'
# kwargs['run_training'] = True
# kwargs['run_evaluation'] = True
# kwargs['save_latent_space'] = True
# kwargs['plot_latent_space'] = 'sns'
# kwargs['plot_latent_space_cols'] = ['MSKCC']
# kwargs['y_head_cols'] = ['MSKCC BINARY']
# kwargs['y_adv_cols'] = []


# kwargs['head_kwargs_dict'] = {}
# kwargs['adv_kwargs_dict'] = {}
# kwargs['head_kwargs_list'] = [{
#     'kind': 'Binary',
#     'name': 'MSKCC',
#     'weight': 1,
#     'y_idx': 0,
#     'hidden_size': 4,
#     'num_hidden_layers': 0,
#     'dropout_rate': 0,
#     'activation': 'leakyrelu',
#     'use_batch_norm': False,
#     'num_classes': 2,
#     }]

# kwargs['encoder_kind'] = 'AE'
# kwargs['adv_kwargs_list'] = []
# kwargs['train_kwargs'] = {}
# kwargs['train_kwargs']['num_epochs'] = 50
# kwargs['train_kwargs']['early_stopping_patience'] = 10
# kwargs['train_kwargs']['head_weight'] = 1
# kwargs['train_kwargs']['encoder_weight'] = 0
# kwargs['train_kwargs']['adversary_weight'] = 0
# kwargs['run_evaluation'] = True
# kwargs['eval_kwargs'] = {}
# kwargs['eval_kwargs']['sklearn_models'] = {}

# kwargs = convert_model_kwargs_list_to_dict(kwargs)

# # kwargs = convert_model_kwargs_list_to_dict(kwargs)
# # run_id = setup_neptune_run(data_dir,setup_id='finetune_mkscc',with_run_id=run_id,**kwargs)


# kwargs['run_random_init'] = True
# kwargs['load_model_weights'] = False
# _ = setup_neptune_run(data_dir,setup_id='randinit_mkscc',with_run_id=run_id,**kwargs)