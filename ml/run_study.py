# run an optuna study

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune
import os
import optuna
import json
from prep_study import add_runs_to_study, get_run_id_list, reuse_run, \
    objective_func1, make_kwargs, convert_distributions_to_suggestion, convert_model_kwargs_list_to_dict,\
    round_kwargs_to_sig
from setup2 import setup_neptune_run
from misc import download_data_dir
from sklearn.linear_model import LogisticRegression

# BASE_DIR = '/DATA2'
# DATA_DIR = f'{BASE_DIR}/data'
# TRIAL_DIR = f'{BASE_DIR}/trials'
storage_name = 'optuna'
USE_WEBAPP_DB = True
SAVE_TRIALS = True
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'


STUDY_INFO_DICT = {
    'study_name': 'OBJ1_March28',
    'objective_name': 'OBJ4 equal weights (v0)',
    'recon_weight': 1,
    'isPediatric_weight': 1,
    'cohortLabel_weight': 1,
    'advStudyID_weight': 1,
}


# STUDY_INFO_DICT = {
#     'study_name': 'OBJ no Adv (v0)',
#     'objective_name': 'OBJ no Adv (v0)',
#     'recon_weight': 1,
#     'isPediatric_weight': 1,
#     'cohortLabel_weight': 1,
#     'advStudyID_weight': 0,
# }


#TODO save the study info dict to neptune metadata


data_dir = '/DATA2'
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(data_dir+'/X_pretrain_train.csv'):
    data_url = 'https://www.dropbox.com/scl/fo/iy2emxpwa4rkr3ad7vhc2/h?rlkey=hvhfa3ny9dlavnooka3kwvu5v&dl=1'
    download_data_dir(data_url, save_dir=data_dir)


def compute_objective(run_id):
    return objective_func1(run_id,
                           data_dir=data_dir,
                           objective_info_dict=STUDY_INFO_DICT)


def objective(trial):

    kwargs = make_kwargs()
    # kwargs = convert_model_kwargs_list_to_dict(kwargs)
    kwargs = convert_distributions_to_suggestion(kwargs, trial)
    kwargs = round_kwargs_to_sig(kwargs,sig_figs=2)

    kwargs['run_evaluation'] = True
    kwargs['eval_kwargs'] = {
        'sklearn_models': {
            'Adversary Logistic Regression': LogisticRegression(max_iter=10000, C=1.0, solver='lbfgs')
        }
    }
    setup_id = 'pretrain'
    run_id = setup_neptune_run(data_dir,setup_id=setup_id,**kwargs)

    kwargs['load_encoder_loc'] = setup_id
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
    
    kwargs['eval_kwargs'] = {}
    kwargs['eval_kwargs']['sklearn_models'] = {}

    # kwargs = convert_model_kwargs_list_to_dict(kwargs)
    run_id = setup_neptune_run(data_dir,setup_id='finetune_mkscc',with_run_id=run_id,**kwargs)

    trial.set_user_attr('run_id',run_id)
    trial.set_user_attr('setup_id',setup_id)

    return compute_objective(run_id)



if USE_WEBAPP_DB:
    print('using webapp database')
    storage_name = WEBAPP_DB_LOC


study_name = STUDY_INFO_DICT['study_name']

study = optuna.create_study(direction="maximize",
                study_name=study_name, 
                storage=storage_name, 
                load_if_exists=True)


# if len(study.trials) < 20:
add_runs_to_study(study,objective_func=compute_objective)


# study.optimize(objective, n_trials=1)