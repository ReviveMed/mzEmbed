# run an optuna study

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune
import os
import optuna
import json
from prep_study import add_runs_to_study, get_run_id_list, reuse_run, \
    objective_func1, make_kwargs, convert_distributions_to_suggestion, convert_model_kwargs_list_to_dict
from setup2 import setup_neptune_run
from misc import download_data_dir

# BASE_DIR = '/DATA2'
# DATA_DIR = f'{BASE_DIR}/data'
# TRIAL_DIR = f'{BASE_DIR}/trials'
storage_name = 'optuna'
USE_WEBAPP_DB = True
SAVE_TRIALS = True
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'


data_dir = '/DATA2'
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(data_dir+'/X_pretrain_train.csv'):
    data_url = 'https://www.dropbox.com/scl/fo/iy2emxpwa4rkr3ad7vhc2/h?rlkey=hvhfa3ny9dlavnooka3kwvu5v&dl=1'
    download_data_dir(data_url, save_dir=data_dir)




def objective(trial):

    kwargs = make_kwargs()
    # kwargs = convert_model_kwargs_list_to_dict(kwargs)
    kwargs = convert_distributions_to_suggestion(kwargs, trial)

    run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

    kwargs['load_encoder_loc'] = 'pretrain'
    # kwargs['load_model_loc'] = 'finetune'
    kwargs['run_training'] = True
    kwargs['run_evaluation'] = True
    kwargs['save_latent_space'] = True
    kwargs['plot_latent_space'] = 'sns'
    kwargs['plot_latent_space_cols'] = ['MSKCC']
    kwargs['y_head_col'] = ['MSKCC BINARY']
    kwargs['y_adv_col'] = []

    kwargs['head_kwargs_dict'] = {}
    kwargs['adv_kwargs_list'] = {}
    kwargs['head_kwargs_list'] = [{
        'kind': 'Binary',
        'name': 'MSKCC',
        'weight': 1,
        'y_idx': 0,
        'hidden_size': 4,
        'num_hidden_layers': 0,
        'dropout_rate': 0,
        'activation': 'leakyrelu',
        'use_batch_norm': False,
        'num_classes': 2,
        }]
    
    kwargs['adv_kwargs_list'] = []
    kwargs['train_kwargs']['epochs'] = 50
    kwargs['train_kwargs']['head_weight'] = 1
    kwargs['train_kwargs']['encoder_weight'] = 1
    kwargs['train_kwargs']['adversary_weight'] = 0
    kwargs['eval_kwargs']['sklearn_models'] = {}

    # kwargs = convert_model_kwargs_list_to_dict(kwargs)
    run_id = setup_neptune_run(data_dir,setup_id='finetune_mkscc',with_run_id=run_id,**kwargs)


    return objective_func1(run_id,data_dir=data_dir)



if USE_WEBAPP_DB:
    print('using webapp database')
    storage_name = WEBAPP_DB_LOC


study_name = 'OBJ1_March27'

study = optuna.create_study(direction="maximize",
                study_name=study_name, 
                storage=storage_name, 
                load_if_exists=True)


add_runs_to_study(study)


study.optimize(objective, n_trials=2)