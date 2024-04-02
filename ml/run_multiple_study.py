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

# get user input
encoder_kind = input('Enter encoder kind (AE, VAE, TGEM_Encoder): ')
num_trials = int(input('Enter number of trials: '))

# encoder_kind = 'VAE'
# encoder_kind = 'TGEM_Encoder'

STUDY_INFO_DICT1 = {
    'objective_name': 'OBJ 1-1-1-1 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 1,
    'cohortLabel_weight': 1,
    'advStudyID_weight': 1,
}


STUDY_INFO_DICT2 = {
    'objective_name': 'OBJ 1-1-1-0 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 1,
    'cohortLabel_weight': 1,
    'advStudyID_weight': 0,
}

STUDY_INFO_DICT3 = {
    'objective_name': 'OBJ 1-1-0-1 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 1,
    'cohortLabel_weight':0,
    'advStudyID_weight': 1,
}


STUDY_INFO_DICT4 = {
    'objective_name': 'OBJ 1-0-0-0 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 0,
    'cohortLabel_weight': 0,
    'advStudyID_weight': 0,
}

#
STUDY_INFO_DICT5 = {
    'objective_name': 'OBJ 1-1-1-10 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 1,
    'cohortLabel_weight': 1,
    'advStudyID_weight': 10,
}

STUDY_INFO_DICT6 = {
    'objective_name': 'OBJ 10-1-1-1 (v1)',
    'recon_weight': 10,
    'isPediatric_weight': 1,
    'cohortLabel_weight': 1,
    'advStudyID_weight': 1,
}

STUDY_INFO_DICT7 = {
    'objective_name': 'OBJ 0-1-1-1 (v1)',
    'recon_weight': 0,
    'isPediatric_weight': 1,
    'cohortLabel_weight': 1,
    'advStudyID_weight': 1,
}

STUDY_INFO_DICT8 = {
    'objective_name': 'OBJ 1-0-0-1 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 0,
    'cohortLabel_weight':0,
    'advStudyID_weight': 1,
}

STUDY_INFO_DICT9 = {
    'objective_name': 'OBJ 1-1-0-10 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 1,
    'cohortLabel_weight':0,
    'advStudyID_weight': 10,
}

STUDY_INFO_DICT0 = {
    'objective_name': 'OBJ 10-1-0-1 (v1)',
    'recon_weight': 10,
    'isPediatric_weight': 1,
    'cohortLabel_weight':0,
    'advStudyID_weight': 1,
}

#
STUDY_INFO_DICT10 = {
    'objective_name': 'OBJ 1-0-0-10 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 0,
    'cohortLabel_weight':0,
    'advStudyID_weight': 10,
}

#
STUDY_INFO_DICT11 = {
    'objective_name': 'OBJ 1-2-2-10 (v1)',
    'recon_weight': 1,
    'isPediatric_weight': 2,
    'cohortLabel_weight':2,
    'advStudyID_weight': 10,
}


#TODO save the study info dict to neptune metadata

def main(STUDY_INFO_DICT):
    data_dir = '/DATA2'
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(data_dir+'/X_pretrain_train.csv'):
        # data_url = 'https://www.dropbox.com/scl/fo/iy2emxpwa4rkr3ad7vhc2/h?rlkey=hvhfa3ny9dlavnooka3kwvu5v&dl=1' #march 22
        data_url = 'https://www.dropbox.com/scl/fo/2xr104jnz9qda7oemrwob/h?rlkey=wy7q95pj81qpgcn7zida2xjps&dl=1' #march 29
        download_data_dir(data_url, save_dir=data_dir)


    def compute_objective(run_id):
        return objective_func1(run_id,
                            data_dir=data_dir,
                            objective_info_dict=STUDY_INFO_DICT)


    def objective(trial):

        kwargs = make_kwargs(encoder_kind=encoder_kind)
        kwargs = convert_model_kwargs_list_to_dict(kwargs)
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
        trial.set_user_attr('run_id',run_id)
        trial.set_user_attr('setup_id',setup_id)

        kwargs['load_encoder_loc'] = 'pretrain'
        # kwargs['load_model_loc'] = 'finetune'
        kwargs['X_filename'] = 'X_finetune'
        kwargs['y_filename'] = 'y_finetune'
        kwargs['run_training'] = True
        kwargs['run_evaluation'] = True
        kwargs['save_latent_space'] = True
        kwargs['plot_latent_space'] = 'sns'
        kwargs['plot_latent_space_cols'] = ['MSKCC']
        kwargs['y_head_cols'] = ['MSKCC BINARY']
        kwargs['y_adv_cols'] = []


        kwargs['head_kwargs_dict'] = {}
        kwargs['adv_kwargs_dict'] = {}
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
        kwargs['train_kwargs']['num_epochs'] = 50
        kwargs['train_kwargs']['early_stopping_patience'] = 10
        kwargs['train_kwargs']['head_weight'] = 1
        kwargs['train_kwargs']['encoder_weight'] = 0
        kwargs['train_kwargs']['adversary_weight'] = 0
        kwargs['run_evaluation'] = True
        kwargs['eval_kwargs'] = {}
        kwargs['eval_kwargs']['sklearn_models'] = {}

        kwargs = convert_model_kwargs_list_to_dict(kwargs)

        # kwargs = convert_model_kwargs_list_to_dict(kwargs)
        run_id = setup_neptune_run(data_dir,setup_id='finetune_mkscc',with_run_id=run_id,**kwargs)


        kwargs['run_random_init'] = True
        kwargs['load_model_weights'] = False
        _ = setup_neptune_run(data_dir,setup_id='randinit_mkscc',with_run_id=run_id,**kwargs)

        return compute_objective(run_id)



    if USE_WEBAPP_DB:
        print('using webapp database')
        storage_name = WEBAPP_DB_LOC

    if 'study_name' in STUDY_INFO_DICT:
        study_name = STUDY_INFO_DICT['study_name']
    else:
        if encoder_kind == 'AE':
            study_name = STUDY_INFO_DICT['objective_name']
        else:
            study_name = encoder_kind + ' ' + STUDY_INFO_DICT['objective_name']

    study = optuna.create_study(direction="maximize",
                    study_name=study_name, 
                    storage=storage_name, 
                    load_if_exists=True)


    # if len(study.trials) < 20:
    add_runs_to_study(study,
                      objective_func=compute_objective,
                      study_kwargs=make_kwargs(encoder_kind=encoder_kind),
                      run_id_list=get_run_id_list(encoder_kind=encoder_kind))


    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':

    if (encoder_kind == 'AE') or (encoder_kind == 'VAE'):
        OBJ_list = [
            STUDY_INFO_DICT5,
            STUDY_INFO_DICT6,
            STUDY_INFO_DICT7,
            STUDY_INFO_DICT8,
            STUDY_INFO_DICT9,
            STUDY_INFO_DICT0,
            STUDY_INFO_DICT10,
            STUDY_INFO_DICT11,
            STUDY_INFO_DICT1,
            STUDY_INFO_DICT2,
            STUDY_INFO_DICT3,
            STUDY_INFO_DICT4,
        ]

    elif encoder_kind == 'TGEM_Encoder':
        OBJ_list = [
            STUDY_INFO_DICT7, #no encoder
        ]

    for study_info_dict in OBJ_list:
        main(study_info_dict)