# run an optuna study

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune
import os
import optuna
import json
from prep_study import add_runs_to_study, get_run_id_list, reuse_run, \
    objective_func2, make_kwargs, convert_distributions_to_suggestion, convert_model_kwargs_list_to_dict,\
    round_kwargs_to_sig, flatten_dict, unflatten_dict, objective_func3, get_study_objective_directions, get_study_objective_keys
from setup2 import setup_neptune_run
from utils_neptune import get_latest_dataset
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier

storage_name = 'optuna'
USE_WEBAPP_DB = True
SAVE_TRIALS = True
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

ADD_EXISTING_RUNS_TO_STUDY = True
limit_add = -1 # limit the number of runs added to the study

import sys
if len(sys.argv)>1:
    encoder_kind = sys.argv[1]
else:
    encoder_kind = input('Enter encoder kind (AE, VAE, TGEM_Encoder): ')


if len(sys.argv)>2:
    num_trials = int(sys.argv[2])
else:
    num_trials = int(input('Enter number of trials: '))

# get user input
# encoder_kind = input('Enter encoder kind (AE, VAE, TGEM_Encoder): ')
# num_trials = int(input('Enter number of trials: '))

# encoder_kind = 'VAE'
# encoder_kind = 'TGEM_Encoder'

STUDY_DICT = {
    'study_name': 'Multi Obj Apr15',
    'encoder_kind': encoder_kind,
    'objectives': {
        'reconstruction_loss':{
            'weight': 1,
            'name': 'Reconstruction Loss',
            'direction': 'minimize',
            'transform': 'log10'
        },
        'Binary_isPediatric':{
            'weight': 1,
            'name': 'Pediatric Prediction',
            'direction': 'maximize'
        },
        'MultiClass_Cohort Label':{
            'weight': 1,
            'name': 'Cohort Label Prediction',
            'direction': 'maximize'
        },
        # 'MultiClass_Adv StudyID':{
        #     'weight': 1,
        #     'name': 'Adv StudyID Prediction',
        #     'direction': 'minimize'
        # },
        'Binary_isFemale':{
            'weight': 1,
            'name': 'Gender Prediction',
            'direction': 'maximize'
        },
        'Regression_Age':{
            'weight': 1,
            'name': 'Age Prediction',
            'direction': 'minimize',
            'transform': 'log10'
        },
        }
}




#TODO save the study info dict to neptune metadata

def main(STUDY_INFO_DICT):
    
    data_dir = get_latest_dataset()

    def compute_objective(run_id):
        return objective_func3(run_id,
                            data_dir=data_dir,
                            objective_keys=get_study_objective_keys(STUDY_INFO_DICT),
                            objectives_info_dict=STUDY_INFO_DICT['objectives'])


    def objective(trial):

        try:
            kwargs = make_kwargs(encoder_kind=encoder_kind)
            kwargs = convert_model_kwargs_list_to_dict(kwargs)
            kwargs = flatten_dict(kwargs) # flatten the dict for optuna compatibility
            kwargs = convert_distributions_to_suggestion(kwargs, trial) # convert the distributions to optuna suggestions
            kwargs = round_kwargs_to_sig(kwargs,sig_figs=2)
            kwargs = unflatten_dict(kwargs) # unflatten the dict for the setup function

            kwargs['run_evaluation'] = True
            kwargs['eval_kwargs'] = {
                'sklearn_models': {
                    'Adversary Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
                    # 'Adversary KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'),
                }
            }
            kwargs['study_info_dict'] = STUDY_INFO_DICT
            setup_id = 'pretrain'
            run_id = setup_neptune_run(data_dir,setup_id=setup_id,**kwargs)
            trial.set_user_attr('run_id',run_id)
            trial.set_user_attr('setup_id',setup_id)

            return compute_objective(run_id)
        
        # except Exception as e:
        except ValueError as e:
            print(e)
            # return float('nan')
            raise optuna.TrialPruned()



    if USE_WEBAPP_DB:
        print('using webapp database')
        storage_name = WEBAPP_DB_LOC

    if 'study_name' in STUDY_INFO_DICT:
        if 'encoder_kind' in STUDY_INFO_DICT:
            study_name = STUDY_INFO_DICT['study_name'] + f' {STUDY_INFO_DICT["encoder_kind"]}'
        else:
            study_name = STUDY_INFO_DICT['study_name']
    else:
        study_name = f'{encoder_kind} Study'

    
    study = optuna.create_study(directions=get_study_objective_directions(STUDY_INFO_DICT),
                    study_name=study_name, 
                    storage=storage_name, 
                    load_if_exists=True)


    if (len(study.trials) < 250) and ADD_EXISTING_RUNS_TO_STUDY:
        add_runs_to_study(study,
                        objective_func=compute_objective,
                        study_kwargs=make_kwargs(encoder_kind=encoder_kind),
                        run_id_list=get_run_id_list(encoder_kind=encoder_kind),
                        limit_add=limit_add)


    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':

    if (encoder_kind == 'AE') or (encoder_kind == 'VAE'):
        OBJ_list = [
            STUDY_DICT, #dual obj 1
        ]

    elif encoder_kind == 'TGEM_Encoder':
        OBJ_list = [
            STUDY_DICT, #no encoder
        ]

    for study_info_dict in OBJ_list:
        main(study_info_dict)