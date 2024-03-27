NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune

import optuna
import json

from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
# from optuna.distributions import json_to_distribution, check_distribution_compatibility, distribution_to_json

#########################################################################################



def add_runs_to_study(study,run_id_list=None,study_kwargs=None,objective_func=None):
    if run_id_list is None:
        run_id_list = get_run_id_list()

    if study_kwargs is None:
        study_kwargs = make_kwargs()

    if objective_func is None:
        objective_func = objective_func1


    for run_id in run_id_list:
        trial = reuse_run(run_id, study_kwargs=study_kwargs, objective_func=objective_func)
        study.add_trial(trial)

    return


########################################################################################
########################################################################################

def get_run_id_list():

    project = neptune.init_project(
        project='revivemed/RCC',
        mode="read-only",
    )

    runs_table_df = project.fetch_runs_table(tag=['v3.1'],state='inactive').to_pandas()
    run_id_list = runs_table_df['sys/id'].tolist()
    return run_id_list


########################################################################################

def reuse_run(run_id,study_kwargs=None,objective_func=None):
    if study_kwargs is None:
        study_kwargs = make_kwargs()

    run = neptune.init_run(project='revivemed/RCC',
                        api_token=NEPTUNE_API_TOKEN,
                        with_id=run_id,
                        mode='read-only')
    print(run_id)
    pretrain_kwargs = run['pretrain/kwargs'].fetch()
    run.stop()

    pretrain_kwargs = convert_neptune_kwargs(pretrain_kwargs)
    # print(pretrain_kwargs['head_kwargs_list'][1]['weight'])
    
    diff = dict_diff(flatten_dict(study_kwargs), flatten_dict(pretrain_kwargs))
    diff_clean = dict_diff_cleanup(diff)

    # check that the first value of each tuple is a distribution
    yes_raise = False
    for k, v in diff_clean.items():
        if not isinstance(v[0], optuna.distributions.BaseDistribution):
            yes_raise = True
            print(f"Value {v} for key {k} is not a distribution")
    
    if yes_raise:
        raise ValueError("Some values are not distributions")
        
    params = {k: v[1] for k, v in diff_clean.items()}
    # print(params)
    distributions = {k: v[0] for k, v in diff_clean.items()}
    
    objective_val = objective_func(run_id)

    if objective_val is None:
        raise ValueError("Objective function returned None")
    

    trial = optuna.create_trial(
        params=params, 
        distributions=distributions, 
        value=objective_val)
    
    return trial


########################################################################################


def objective_func1(run_id,recompute_eval=False):

    obj_val = None
    recon_weight = 1
    isPediatric_weight = 1
    cohortLabel_weight = 1
    advStudyID_weight = 1
    objective_name = 'OBJECTIVE equal weights'


    run = neptune.init_run(project='revivemed/RCC',
                    api_token=NEPTUNE_API_TOKEN,
                    with_id=run_id)
    
    if recompute_eval:
        raise NotImplementedError("Need to recompute")

    pretrain_output = run['pretrain'].fetch()
    
    if 'eval' in pretrain_output:
        eval_res = pretrain_output['eval']['val']

        recon_loss = eval_res['reconstruction_loss']
        if 'Binary_isPediatric' in eval_res:
            isPediatric_auc = eval_res['Binary_isPediatric']['AUROC (micro)']
        else:
            isPediatric_auc = 0.5

        if 'MultiClass_CohortLabel' in eval_res:
            cohortLabel_auc = eval_res['MultiClass_CohortLabel']['AUROC (ovo, macro)']
        else:
            cohortLabel_auc = 0.5

        if 'MultiClass_AdvStudyID' in eval_res:
            advStudyID_auc = eval_res['MultiClass_AdvStudyID']['AUROC (ovo, macro)']
        else:
            advStudyID_auc = 0.5

        obj_val = -1*(recon_weight)*recon_loss \
            + (isPediatric_weight)*isPediatric_auc \
            + (cohortLabel_weight)*cohortLabel_auc \
            + -1*(advStudyID_weight)*advStudyID_auc

        run[f'objectives/{objective_name}'] = obj_val 

        run.stop()
    else:
        # run the evaluation!
        run.stop()
        raise ValueError("No eval results")
        
    return obj_val


########################################################################################

def make_kwargs():
    activation = 'leakyrelu'
    latent_size = IntDistribution(4, 100, log=True)
    num_hidden_layers = IntDistribution(1, 5)
    cohort_label_weight = FloatDistribution(0,10)
    head_weight = FloatDistribution(0,10)
    # head_weight = FloatDistribution(0.1, 10, log=True)
    adv_weight = FloatDistribution(0,10)
    encoder_kwargs = {
                'activation': activation,
                'latent_size': latent_size,
                'num_hidden_layers': num_hidden_layers,
                'dropout_rate': 0.2,
                'use_batch_norm': False,
                # 'hidden_size': int(1.5*latent_size),
                'hidden_size_mult' : 1.5
                }

    kwargs = {
                    ################
                    ## General ##
                    'encoder_kind': 'AE',
                    'encoder_kwargs': encoder_kwargs,
                    'other_size': 1,
                    'y_head_cols' : ['is Pediatric','Cohort Label ENC'],
                    'y_adv_cols' : ['Study ID ENC'],

                    ################
                    ## Pretrain ##

                    'holdout_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
                    'batch_size': 64,
                    
                    'head_kwargs_list': [
                        {
                            'kind': 'Binary',
                            'name': 'isPediatric',
                            'y_idx': 0,
                            'weight': 1.0,
                            'hidden_size': 4,
                            'num_hidden_layers': 1,
                            'dropout_rate': 0,
                            'activation': 'leakyrelu',
                            'use_batch_norm': False,
                            'num_classes': 2,
                        },
                        {
                            'kind': 'MultiClass',
                            'name': 'Cohort Label',
                            'y_idx': 1,
                            'weight': cohort_label_weight,
                            'hidden_size': 4,
                            'num_hidden_layers': 1,
                            'dropout_rate': 0,
                            'activation': 'leakyrelu',
                            'use_batch_norm': False,
                            'num_classes': 4,
                        },
                    ],
                    
                    'adv_kwargs_list': [
                        {
                            'kind': 'MultiClass',
                            'name': 'Adv StudyID',
                            'y_idx': 0,
                            'weight': 1.0, 
                            'hidden_size': 4,
                            'num_hidden_layers': 1,
                            'dropout_rate': 0,
                            'activation': 'leakyrelu',
                            'use_batch_norm': False,
                            'num_classes': 19,
                        },
                    ],

                    'train_kwargs': {
                        # 'num_epochs': trial.suggest_int('pretrain_epochs', 10, 100,log=True),
                        'num_epochs': 100,
                        'lr': 0.01,
                        'weight_decay': 0,
                        'l1_reg_weight': 0,
                        'l2_reg_weight': 0.001,
                        'encoder_weight': 1,
                        'head_weight': head_weight,
                        'adversary_weight': adv_weight,
                        'noise_factor': 0.1,
                        'early_stopping_patience': 20,
                        'adversarial_mini_epochs': 5,
                    },

        }
    return kwargs


########################################################################################
########################################################################################
# Helper Functions
########################################################################################
########################################################################################

def convert_distributions_to_json(obj):
    if isinstance(obj, optuna.distributions.BaseDistribution):
        return optuna.distributions.distribution_to_json(obj)
    elif isinstance(obj, dict):
        return {k: convert_distributions_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_distributions_to_json(v) for v in obj]
    elif callable(obj):
        return str(obj)
    else:
        return obj

def convert_json_to_distributions(obj):
    if isinstance(obj, str):
        try:
            return optuna.distributions.json_to_distribution(obj)
        except:
            return obj
    elif isinstance(obj, dict):
        return {k: convert_json_to_distributions(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_to_distributions(v) for v in obj]
    else:
        return obj
    

def convert_distributions_to_suggestion(obj,trial,name=None):
    if isinstance(obj, optuna.distributions.BaseDistribution):
        print('base')
        return trial._suggest(name, obj)
    elif isinstance(obj, dict):
        print('dict')
        return {k: convert_distributions_to_suggestion(v,trial,name=k) for k, v in obj.items()}
    elif isinstance(obj, list):
        print('list')
        return [convert_distributions_to_suggestion(v,trial) for v in obj]
    else:
        return obj    
    
def convert_neptune_kwargs(kwargs):
    if isinstance(kwargs, dict):
        return {k: convert_neptune_kwargs(v) for k, v in kwargs.items()}
    elif isinstance(kwargs, str):
        try:
            return eval(kwargs)
        except:
            return kwargs
    else:
        return kwargs    
    

def flatten_dict(d, parent_key='', sep='__'):
        items = []
        if not isinstance(d, dict):
            return {parent_key: d}
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v,list):
                for i, item in enumerate(v):
                    items.extend(flatten_dict(item, new_key + sep + str(i), sep=sep).items())            
            else:
                items.append((new_key, v))
        return dict(items)

def unflatten_dict(d, sep='__'):
    out = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = out
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    for key, value in out.items():
        if 'list' in key:
            out[key] = list(value.values())    
    return out



def dict_diff(d1, d2):
    diff = {}
    for k in d1.keys() & d2.keys():
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            sub_diff = dict_diff(d1[k], d2[k])
            if sub_diff:
                diff[k] = sub_diff
        elif isinstance(d1[k], list) and isinstance(d2[k], list):
            if d1[k] != d2[k]:
                diff[k] = (d1[k], d2[k])
        elif d1[k] != d2[k]:
            diff[k] = (d1[k], d2[k])
    for k in d1.keys() - d2.keys():
        diff[k] = (d1[k], None)
    for k in d2.keys() - d1.keys():
        diff[k] = (None, d2[k])
    return diff


def dict_diff_cleanup(diff,ignore_keys_list=None):
    if ignore_keys_list is None:
        ignore_keys_list = ['run_evaluation','save_latent_space','plot_latent_space_cols','plot_latent_space',\
                    'eval_kwargs','train_kwargs__eval_funcs','run_training','encoder_kwargs__hidden_size']

    diff_clean = {}
    for key, val in diff.items():
        for ignore_key in ignore_keys_list:
            if key.startswith(ignore_key):
                break
        else:
            diff_clean[key] = val

    return diff_clean