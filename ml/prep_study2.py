import neptune
import numpy as np
import optuna
import json
from utils_neptune import get_run_id_list, check_neptune_existance
from setup3 import setup_neptune_run
from misc import round_to_sig
from prep_run import convert_neptune_kwargs, dict_diff, dict_diff_cleanup, flatten_dict, convert_model_kwargs_list_to_dict
# from optuna.distributions import json_to_distribution, check_distribution_compatibility, distribution_to_json
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='
from neptune.utils import stringify_unsupported
from prep_run import make_kwargs_set
import os
import pandas as pd
import anndata as ad

# import neptune exceptions
from neptune.exceptions import NeptuneException #, NeptuneServerError

#########################################################################################


def get_default_study_kwargs(head_kwargs_dict,adv_kwargs_dict,**kwargs):
    encoder_kind = kwargs.get('encoder_kind','VAE')
    run_kwargs = make_kwargs_set(encoder_kind=encoder_kind,
                head_kwargs_dict=head_kwargs_dict,
                adv_kwargs_dict=adv_kwargs_dict,

                latent_size=None, latent_size_min=96, latent_size_max=128, latent_size_step=4,
                hidden_size=-1, hidden_size_min=16, hidden_size_max=64, hidden_size_step=16,
                hidden_size_mult=1.5, hidden_size_mult_min=1.25, hidden_size_mult_max=2, hidden_size_mult_step=0.25,
                num_hidden_layers=None, num_hidden_layers_min=2, num_hidden_layers_max=3, num_hidden_layers_step=1,
                
                num_attention_heads=-1, num_attention_heads_min=1, num_attention_heads_max=5, num_attention_heads_step=1,
                num_decoder_layers=-1, num_decoder_layers_min=1, num_decoder_layers_max=5, num_decoder_layers_step=1,
                embed_dim=-1, embed_dim_min=4, embed_dim_max=64, embed_dim_step=4,
                decoder_embed_dim=-1, decoder_embed_dim_min=4, decoder_embed_dim_max=64, decoder_embed_dim_step=4,
                default_hidden_fraction=-1, default_hidden_fraction_min=0, default_hidden_fraction_max=0.5, default_hidden_fraction_step=0.1,

                dropout_rate=0, dropout_rate_min=0, dropout_rate_max=0.5, dropout_rate_step=0.1,
                encoder_weight=1.0, encoder_weight_min=0, encoder_weight_max=2, encoder_weight_step=0.5,
                head_weight=1.0, head_weight_min=0, head_weight_max=2, head_weight_step=0.5,
                adv_weight=1.0, adv_weight_min=0, adv_weight_max=2, adv_weight_step=0.5,

                task_head_weight=None, task_head_weight_min=0.25, task_head_weight_max=2, task_head_weight_step=0.25,
                task_adv_weight=-1, task_adv_weight_min=0, task_adv_weight_max=10, task_adv_weight_step=0.1,
                task_hidden_size=4, task_hidden_size_min=4, task_hidden_size_max=64, task_hidden_size_step=4,
                task_num_hidden_layers=1, task_num_hidden_layers_min=1, task_num_hidden_layers_max=3, task_num_hidden_layers_step=1,

                weight_decay=None, weight_decay_min=1e-5, weight_decay_max=1e-2, weight_decay_step=0.00001, weight_decay_log=True,
                l1_reg_weight=0, l1_reg_weight_min=0, l1_reg_weight_max=0.01, l1_reg_weight_step=0.0001, l1_reg_weight_log=False,
                l2_reg_weight=0, l2_reg_weight_min=0, l2_reg_weight_max=0.01, l2_reg_weight_step=0.0001, l2_reg_weight_log=False,
                
                batch_size=96, batch_size_min=16,batch_size_max=128,batch_size_step=16,
                noise_factor=0.1, noise_factor_min=0.01, noise_factor_max=0.1, noise_factor_step=0.01,
                num_epochs=None, num_epochs_min=50, num_epochs_max=400, num_epochs_step=25, num_epochs_log=False,
                learning_rate=None, learning_rate_min=0.00001, learning_rate_max=0.005, learning_rate_step=None,learning_rate_log=True,
                early_stopping_patience=0, early_stopping_patience_min=0, early_stopping_patience_max=50, early_stopping_patience_step=5,
                adversarial_start_epoch=0, adversarial_start_epoch_min=-1, adversarial_start_epoch_max=10, adversarial_start_epoch_step=2,
                )
    
    return run_kwargs


def create_optuna_study(study_info_dict,get_kwargs_wrapper,head_kwargs_dict,adv_kwargs_dict,compute_objective):
    use_webapp_db = study_info_dict.get('use_webapp_db',True)
    if use_webapp_db:
        print('using webapp database')
        storage_name = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'
    else:
        raise NotImplementedError('Only webapp database is supported')

    encoder_kind = study_info_dict.get('encoder_kind','')
    add_existing_runs_to_study = study_info_dict.get('add_existing_runs_to_study',False)
    acceptable_tag = study_info_dict.get('acceptable_tag','v4')
    limit_add = study_info_dict.get('limit_add',-1)

    if 'study_name' in study_info_dict:
        if 'encoder_kind' in study_info_dict:
            study_name = study_info_dict['study_name'] + f' {study_info_dict["encoder_kind"]}'
        else:
            study_name = study_info_dict['study_name']
    else:
        study_name = f'{encoder_kind} Study'

    study = optuna.create_study(directions=get_study_objective_directions(study_info_dict),
                    study_name=study_name, 
                    storage=storage_name, 
                    load_if_exists=True)

    if (len(study.trials) < 250) and add_existing_runs_to_study:
        # raise NotImplementedError('ADD_EXISTING_RUNS_TO_STUDY not implemented in this script')
        add_runs_to_study(study,
            objective_func=compute_objective,
            study_kwargs=get_kwargs_wrapper(head_kwargs_dict,adv_kwargs_dict),
            run_id_list=get_run_id_list(encoder_kind=encoder_kind,tag=acceptable_tag),
            limit_add=limit_add)

    return study


def add_runs_to_study(study,run_id_list=None,study_kwargs=None,
                      objective_func=None,limit_add=-1):
    if run_id_list is None:
        run_id_list = get_run_id_list()

    print('number of runs: ', len(run_id_list))
    
    if limit_add>-1:
        if len(run_id_list) > limit_add:
            run_id_list = run_id_list[:limit_add]

    if study_kwargs is None:
        raise ValueError("Study kwargs is None")
    study_kwargs = convert_model_kwargs_list_to_dict(study_kwargs,style=2)
        # study_kwargs = convert_model_kwargs_list_to_dict(study_kwargs,style=2)

    if objective_func is None:
        # objective_func = lambda x: objective_func1(x,data_dir='/DATA2')
        raise ValueError("Objective function is None")

    num_runs_to_check = len(run_id_list)
    print('number of runs attempt to add to study: ', num_runs_to_check)
    # shuffle the run_id_list
    np.random.shuffle(run_id_list)

    already_in_study = 0
    newely_added_to_study = 0
    skipped_due_to_missing_eval = 0
    skipped_due_to_distribution_mismatch = 0
    skipped_due_to_neptune_error = 0

    for run_id in run_id_list:
        #TODO test this code
        #check if the trial is already in the study by looking at the user attributes
        if run_id in [t.user_attrs['run_id'] for t in study.trials if 'run_id' in t.user_attrs]:
            print(f"Run {run_id} is already in the study")
            already_in_study += 1
            continue

        print('adding {} to study'.format(run_id))
        try:
            trial = reuse_run(run_id, study_kwargs=study_kwargs, objective_func=objective_func)
            study.add_trial(trial)
            newely_added_to_study += 1
        except ValueError as e:
            print(f"Error with run {run_id}: {e}")
            error_text = e.args[0]
            if 'no eval output' in error_text:
                skipped_due_to_missing_eval += 1
            elif 'Some values are not distributions' in error_text:
                skipped_due_to_distribution_mismatch += 1
            continue
        except NeptuneException as e:
            print(f"Error with run {run_id}: {e}")
            skipped_due_to_neptune_error += 1
            continue
        # except NeptuneServerError as e:
        #     print(f"Error with run {run_id}: {e}")
        #     continue


    print('###################')
    print('###################')
    print('Number of runs attempted to add to study: ', num_runs_to_check)
    print(f"Number of runs already in study: {already_in_study}")
    print(f"Number of runs newly added to study: {newely_added_to_study}")
    print(f"Number of runs skipped due to missing eval: {skipped_due_to_missing_eval}")
    print(f"Number of runs skipped due to distribution mismatch: {skipped_due_to_distribution_mismatch}")
    print(f"Number of runs skipped due to neptune error: {skipped_due_to_neptune_error}")

    tot_num_trials_in_study = len(study.trials)
    print(f"Total number of trials in study: {tot_num_trials_in_study}")
    print('###################')


    return


########################################################################################
########################################################################################

########################################################################################


def get_default_kwarg_val_dict():
    default_val_dict = {
        'head_kwargs_dict__Regression_Age__weight': 0,
        'head_kwargs_dict__Binary_isFemale__weight': 0,
        'head_kwargs_dict__BMI__weight': 0,
        'train_kwargs__l1_reg_weight': 0,
        'train_kwargs__optimizer_name': 'adam',
        'train_kwargs__adversarial_start_epoch': -1,
        'num_repeats': 1,
        'head_kwargs_dict__Smoking__weight': 0,
        'head_kwargs_dict__Cohort-Label__weight': 0,
        'head_kwargs_dict__is-Pediatric__weight': 0,
    }

    return default_val_dict



def reuse_run(run_id,study_kwargs=None,objective_func=None,ignore_keys_list=None,
              default_kwarg_val_dict=None,verbose=1,setup_id='pretrain',project_id='revivemed/RCC',
              neptune_api_token=NEPTUNE_API_TOKEN):
    if study_kwargs is None:
        raise ValueError("Study kwargs is None")

    if ignore_keys_list is None:
        ignore_keys_list = ['run_evaluation','save_latent_space','plot_latent_space_cols','plot_latent_space',\
            'eval_kwargs','run_training','overwrite_existing_kwargs',\
            'load_model_loc','y_head_cols','eval_name','train_name','study_info_dict','y_fit_file','source run_id',
            'X_fit_file','X_eval_file','y_eval_file','num_repeats','y_adv_cols','head_kwargs_dict','adv_kwargs_dict','fit_kwargs__dropout_rate']


    if default_kwarg_val_dict is None:
        default_kwarg_val_dict = get_default_kwarg_val_dict()

    run = neptune.init_run(project=project_id,
                        api_token=neptune_api_token,
                        with_id=run_id,
                        mode='read-only')
    print(run_id)
    pretrain_kwargs = run[f'{setup_id}/original_kwargs'].fetch()
    run.stop()

    pretrain_kwargs = convert_neptune_kwargs(pretrain_kwargs)
    # print(pretrain_kwargs['head_kwargs_list'][1]['weight'])
    
    #TODO test this
    pretrain_kwargs = convert_model_kwargs_list_to_dict(pretrain_kwargs)
    study_kwargs = convert_model_kwargs_list_to_dict(study_kwargs)

    diff = dict_diff(flatten_dict(study_kwargs), flatten_dict(pretrain_kwargs))
    # diff_clean = dict_diff_cleanup(diff)

    # check that the first value of each tuple is a distribution
    yes_raise = False
    diff_clean = {}
    for k, v in diff.items():
        yes_ignore = False
        if not isinstance(v[0], optuna.distributions.BaseDistribution):

            for ignore_key in ignore_keys_list:
                if k.startswith(ignore_key):
                    if verbose>1:
                        print(f"Ignoring key {k}")
                    yes_ignore = True
                    break
            
            if (not yes_ignore):
                yes_raise = True
                if verbose > 0:
                    print(f"Value {v} for key {k} is not a distribution")

        else:
            if v[1] is None:
                if k in default_kwarg_val_dict:
                    diff_clean[k] = (v[0], default_kwarg_val_dict[k])
                else:
                    if verbose>0:
                        print(f"Value {v} for key {k} is None")
                    yes_raise = True
                    diff_clean[k] = v
            else:
                diff_clean[k] = v

    if yes_raise:
        raise ValueError("Some values are not distributions or are missing")
        
    params = {k: v[1] for k, v in diff_clean.items()}
    # print(params)
    distributions = {k: v[0] for k, v in diff_clean.items()}
    
    objective_val = objective_func(run_id)

    if objective_val is None:
        raise ValueError("Objective function returned None")
    
    if len(objective_val) == 1:
        trial = optuna.create_trial(
            params=params, 
            distributions=distributions, 
            value=objective_val,
            user_attrs={'run_id': run_id, 'setup_id': setup_id})
    else:
        trial = optuna.create_trial(
            params=params, 
            distributions=distributions, 
            values=objective_val,
            user_attrs={'run_id': run_id, 'setup_id': setup_id})
    
    print('Adding run {} to study [{}] with parameters {}'.format(run_id,objective_val,params))

    return trial


########################################################################################

def get_study_objective_keys(study_info_dict):
    objective_keys = list(study_info_dict['objectives'].keys())
    return sorted(objective_keys)

def get_study_objective_directions(study_info_dict,convert_to_int=False):
    objective_keys = get_study_objective_keys(study_info_dict)
    directions = [study_info_dict['objectives'][k]['direction'] for k in objective_keys]
    if convert_to_int:
        directions = np.array([1 if d=='minimize' else -1 for d in directions])
    return directions

def get_study_objective_weights(study_info_dict):
    objective_keys = get_study_objective_keys(study_info_dict)
    weights = []
    for k in objective_keys:
        if 'weight' in study_info_dict['objectives'][k]:
            weights.append(study_info_dict['objectives'][k]['weight'])
        else:
            weights.append(1)
    weights = np.array(weights)
    return weights


def objective_func4(run_id,study_info_dict,
                    project_id='revivemed/RCC',
                    neptune_api_token=NEPTUNE_API_TOKEN,
                    setup_id='pretrain',
                    eval_file_id='Pretrain_Discovery_Val',
                    sum_objectives=False):

    objectives_info_dict = study_info_dict['objectives']
    objective_keys =  get_study_objective_keys(study_info_dict)


    run = neptune.init_run(project=project_id,
                    api_token=neptune_api_token,
                    with_id=run_id,
                    capture_stdout=False,
                    capture_stderr=False,
                    capture_hardware_metrics=False,
                    mode='read-only')


    obj_vals = []
    try:
        # if setup_id in run.get_structure()
        pretrain_output = run[setup_id].fetch()
    
    except NeptuneException as e:
        print(f"Error with run {run_id}: {e}")
        run.stop()
        raise ValueError(f"Error with run {run_id}: {e}")

    run_struc = run.get_structure()

    if not setup_id in run_struc:
        run.stop()
        raise ValueError(f"Error with run {run_id}: no {setup_id} output")
    
    if not 'eval' in run_struc[setup_id]:
        run.stop()
        raise ValueError(f"Error with run {run_id}: no eval output")
    
    if not eval_file_id in run_struc[setup_id]['eval']:
        run.stop()
        raise ValueError(f"Error with run {run_id}: no {eval_file_id} output")

    # eval_res = run['pretrain/eval/val'].fetch()
    eval_loc = f'{setup_id}/eval/{eval_file_id}'
    eval_res = run_struc[setup_id]['eval'][eval_file_id]

    for objective_key in objective_keys:

        if objective_key not in eval_res.keys():
            print(f'no exact match of {objective_key} in eval_res')
            eval_key_matches = []
            for eval_key in eval_res.keys():
                if objective_key in eval_key:
                    eval_key_matches.append(eval_key)
            
            if len(eval_key_matches) == 0:
                # raise ValueError(f"Objective {objective_key} not in eval results")
                print(f"Objective {objective_key} not in eval results, use default value")
                obj_val = objectives_info_dict[objective_key]['default_value']
        else:
            eval_key_matches = [objective_key]
        

        if len(eval_key_matches) > 0:
            print(f'Objective {objective_key} will be the average of values from {eval_key_matches}')
            val0_list = []
            for eval_key in eval_key_matches:
                if isinstance(eval_res[eval_key],dict):
                    sub_keys = list(eval_res[eval_key].keys())
                    for sub_key in sub_keys:
                        try:
                            val0 = run[f'{eval_loc}/{eval_key}/{sub_key}'].fetch_last()
                        except NeptuneException:
                            val0 = run[f'{eval_loc}/{eval_key}/{sub_key}'].fetch()
                        val0_list.append(val0)

                else:
                    try:
                        val0 = run[f'{eval_loc}/{eval_key}'].fetch_last()
                    except NeptuneException:
                        val0 = run[f'{eval_loc}/{eval_key}'].fetch()
                    val0_list.append(val0)

            obj_val = np.mean(val0_list)


        if objective_key in objectives_info_dict:
            if 'transform' in objectives_info_dict[objective_key]:
                transform_str = objectives_info_dict[objective_key]['transform']
                if transform_str == 'log10':
                    obj_val = np.log10(obj_val)
                elif transform_str == 'neg':
                    obj_val = -1*obj_val 
                elif transform_str == 'neglog10':
                    obj_val = -1*np.log10(obj_val)

        obj_vals.append(obj_val)
    # else:
    #     run.stop()
    #     raise ValueError(f"no evaluation results for {run_id}")

    if sum_objectives:
        obj_dirs = get_study_objective_directions(study_info_dict)
        obj_weights = get_study_objective_weights(study_info_dict)
        obj_vals = np.array(obj_vals)
        obj_sum = np.sum(obj_dirs*obj_weights*obj_vals)
        run.stop()
        return obj_sum

    run.stop()
    return tuple(obj_vals)


