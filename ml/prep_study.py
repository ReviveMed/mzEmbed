import neptune
import numpy as np
import optuna
import json
from utils_neptune import get_run_id_list, check_neptune_existance
from setup2 import setup_neptune_run
from misc import round_to_sig
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
# from optuna.distributions import json_to_distribution, check_distribution_compatibility, distribution_to_json
from sklearn.linear_model import LogisticRegression
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='
from neptune.utils import stringify_unsupported

# import neptune exceptions
from neptune.exceptions import NeptuneException #, NeptuneServerError

#########################################################################################


def cleanup_runs(run_id_list=None):
    if run_id_list is None:
        run_id_list = get_run_id_list()

    for run_id in run_id_list:
        print('cleanup run: ', run_id)
        run = neptune.init_run(project='revivemed/RCC',
                        api_token=NEPTUNE_API_TOKEN,
                        with_id=run_id,
                        capture_stdout=False,
                        capture_stderr=False,
                        capture_hardware_metrics=False)
        
        if check_neptune_existance(run,'pretrain/original_kwargs'):
            existing_kwargs = run['pretrain/original_kwargs'].fetch()
            existing_kwargs = convert_neptune_kwargs(existing_kwargs)
        else:
            existing_kwargs = run['pretrain/kwargs'].fetch()
            existing_kwargs = convert_neptune_kwargs(existing_kwargs)
            run['pretrain/original_kwargs'] = stringify_unsupported(existing_kwargs)

        kwargs = convert_model_kwargs_list_to_dict(existing_kwargs,style=2)
        del run['pretrain/kwargs']
        run['pretrain/kwargs'] = stringify_unsupported(kwargs)

        # run['sys/failed'] = False
        run.stop()

    return



def add_runs_to_study(study,run_id_list=None,study_kwargs=None,objective_func=None,limit_add=0):
    if run_id_list is None:
        run_id_list = get_run_id_list()

    print('number of runs: ', len(run_id_list))
    
    if limit_add>0:
        if len(run_id_list) > limit_add:
            run_id_list = run_id_list[:limit_add]

    if study_kwargs is None:
        study_kwargs = make_kwargs()
    study_kwargs = convert_model_kwargs_list_to_dict(study_kwargs,style=2)
        # study_kwargs = convert_model_kwargs_list_to_dict(study_kwargs,style=2)

    if objective_func is None:
        # objective_func = lambda x: objective_func1(x,data_dir='/DATA2')
        raise ValueError("Objective function is None")

    
    for run_id in run_id_list:
        #TODO test this code
        #check if the trial is already in the study by looking at the user attributes
        if run_id in [t.user_attrs['run_id'] for t in study.trials if 'run_id' in t.user_attrs]:
            print(f"Run {run_id} is already in the study")
            continue

        print('adding {} to study'.format(run_id))
        try:
            trial = reuse_run(run_id, study_kwargs=study_kwargs, objective_func=objective_func)
            study.add_trial(trial)
        except ValueError as e:
            print(f"Error with run {run_id}: {e}")
            continue
        except NeptuneException as e:
            print(f"Error with run {run_id}: {e}")
            continue
        # except NeptuneServerError as e:
        #     print(f"Error with run {run_id}: {e}")
        #     continue

    return


########################################################################################
########################################################################################


########################################################################################

def reuse_run(run_id,study_kwargs=None,objective_func=None):
    if study_kwargs is None:
        study_kwargs = make_kwargs()

    run = neptune.init_run(project='revivemed/RCC',
                        api_token=NEPTUNE_API_TOKEN,
                        with_id=run_id,
                        mode='read-only')
    setup_id = 'pretrain'
    print(run_id)
    pretrain_kwargs = run[f'{setup_id}/kwargs'].fetch()
    run.stop()

    pretrain_kwargs = convert_neptune_kwargs(pretrain_kwargs)
    # print(pretrain_kwargs['head_kwargs_list'][1]['weight'])
    
    #TODO test this
    pretrain_kwargs = convert_model_kwargs_list_to_dict(pretrain_kwargs)
    study_kwargs = convert_model_kwargs_list_to_dict(study_kwargs)

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
    
    return trial


########################################################################################


def objective_func2(run_id,data_dir,recompute_eval=False,objective_info_dict_list=None):

    obj_vals = []
    for objective_info_dict in objective_info_dict_list:
        obj_val = objective_func1(run_id,data_dir,recompute_eval=recompute_eval,objective_info_dict=objective_info_dict)
        obj_vals.append(obj_val)
    
    return tuple(obj_vals)


def objective_func1(run_id,data_dir,recompute_eval=False,objective_info_dict=None):

    obj_val = None
    if objective_info_dict is None:
        recon_weight = 1
        isPediatric_weight = 1
        cohortLabel_weight = 1
        advStudyID_weight = 1
        isFemale_weight = 0
        objective_name = 'OBJ4 equal weights (v0)'

    recon_weight = objective_info_dict.get('recon_weight',1)
    isPediatric_weight = objective_info_dict.get('isPediatric_weight',1)
    cohortLabel_weight = objective_info_dict.get('cohortLabel_weight',1)
    advStudyID_weight = objective_info_dict.get('advStudyID_weight',1)
    isFemale_weight = objective_info_dict.get('isFemale_weight',0)
    objective_name = objective_info_dict.get('objective_name','OBJ4 equal weights (v0)')


    run = neptune.init_run(project='revivemed/RCC',
                    api_token=NEPTUNE_API_TOKEN,
                    with_id=run_id,
                    capture_stdout=False,
                    capture_stderr=False,
                    capture_hardware_metrics=False)
    try:
        pretrain_output = run['pretrain'].fetch()

        if (recompute_eval) or ('eval' not in pretrain_output):
            
            kwargs = convert_neptune_kwargs(pretrain_output['kwargs'])
            kwargs['overwrite_existing_kwargs'] = True
            kwargs['load_model_loc'] = 'pretrain'
            kwargs['run_training'] = False
            kwargs['run_evaluation'] = True
            kwargs['save_latent_space'] = True
            kwargs['plot_latent_space'] = 'seaborn'
            kwargs['eval_kwargs'] = {
                'sklearn_models': {
                    'Adversary Logistic Regression': LogisticRegression(max_iter=10000, C=1.0, solver='lbfgs')
                }
            }

            setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)
            #raise NotImplementedError("Need to recompute")
    except NeptuneException as e:
        print(f"Error with run {run_id}: {e}")
        run.stop()
        raise ValueError(f"Error with run {run_id}: {e}")

    
    if 'eval' in pretrain_output:
        eval_res = pretrain_output['eval']['val']

        recon_loss = eval_res['reconstruction_loss']
        if 'Binary_isPediatric' in eval_res:
            isPediatric_auc = eval_res['Binary_isPediatric']['AUROC (micro)']
        else:
            isPediatric_auc = 0.5

        if 'MultiClass_Cohort Label' in eval_res:
            cohortLabel_auc = eval_res['MultiClass_Cohort Label']['AUROC (ovo, macro)']
        else:
            cohortLabel_auc = 0.5

        if 'MultiClass_Adv StudyID' in eval_res:
            advStudyID_auc = eval_res['MultiClass_Adv StudyID']['AUROC (ovo, macro)']
        else:
            advStudyID_auc = 0.5

        if 'Binary_isFemale' in eval_res:
            isFemale_auc = eval_res['Binary_isFemale']['AUROC (micro)']
        else:
            isFemale_auc = 0.5

        obj_val = -1*(recon_weight)*recon_loss \
            + (isPediatric_weight)*isPediatric_auc \
            + (cohortLabel_weight)*cohortLabel_auc \
            + -1*(advStudyID_weight)*advStudyID_auc \
            + (isFemale_weight)*isFemale_auc

        run[f'objectives/{objective_name}'] = obj_val 

        run.stop()
    else:
        # run the evaluation!?
        # add tags to the run
        run["sys/tags"].add("no eval") 
        run.stop()
        raise ValueError("No eval results")
        
    return obj_val


########################################################################################

def make_kwargs(sig_figs=2,encoder_kind='AE'):
    activation = 'leakyrelu'
    latent_size = IntDistribution(8, 64, step=4)
    num_hidden_layers = IntDistribution(2, 10)
    cohort_label_weight = FloatDistribution(0,2,step=0.1) #10
    isfemale_weight = FloatDistribution(0.1,10,step=0.1) #20
    ispediatric_weight = FloatDistribution(0.1,10,step=0.1) #10
    head_weight = FloatDistribution(0,10,step=0.1) # 10
    adv_weight = FloatDistribution(0.1,50,step=0.1) #50
    
    if encoder_kind in ['AE']:
        encoder_kwargs = {
                    'activation': activation,
                    'latent_size': latent_size,
                    'num_hidden_layers': num_hidden_layers,
                    'dropout_rate': FloatDistribution(0, 0.5, step=0.1),
                    'use_batch_norm': False,
                    # 'hidden_size': int(1.5*latent_size),
                    'hidden_size_mult' : 1.5
                    }
        encoder_weight = FloatDistribution(0,5,step=0.1)
        num_epochs_min = 50
        num_epochs_max = 300
        num_epochs_step = 10
        adversarial_mini_epochs = 5
        early_stopping_patience_step = 10
        early_stopping_patience_max = 50
        l2_reg_weight = FloatDistribution(0, 0.01, step=0.0001)
        l1_reg_weight = FloatDistribution(0, 0.01, step=0.0001)

    elif encoder_kind == 'VAE':        
        encoder_kwargs = {
                    'activation': activation,
                    'latent_size': latent_size,
                    'num_hidden_layers': num_hidden_layers,
                    'dropout_rate': 0,
                    'use_batch_norm': False,
                    # 'hidden_size': int(1.5*latent_size),
                    'hidden_size_mult' : 1.5
                    }
        encoder_weight = FloatDistribution(0,5,step=0.1)
        num_epochs_min = 50
        num_epochs_max = 200
        num_epochs_step = 10
        adversarial_mini_epochs = 5
        early_stopping_patience_step = 10
        early_stopping_patience_max = 50
        l2_reg_weight = 0 # loss explodes if not 0
        l1_reg_weight = 0

    elif encoder_kind == 'TGEM_Encoder':
        encoder_kwargs = {
                    'activation': 'linear',
                    'n_head': IntDistribution(2, 5, step=1),
                    'n_layers': IntDistribution(2, 3, step=1),
                    'dropout_rate': FloatDistribution(0, 0.5, step=0.1),
                    }
        encoder_weight = 0
        num_epochs_min = 10
        num_epochs_max = 50
        num_epochs_step = 1
        adversarial_mini_epochs = 1
        early_stopping_patience_step = 5
        early_stopping_patience_max = 20
        l2_reg_weight = 0
        l1_reg_weight = 0

    kwargs = {
                ################
                ## General ##
                'encoder_kind': encoder_kind,
                'encoder_kwargs': encoder_kwargs,
                'other_size': 1,
                'y_head_cols' : ['is Pediatric','Cohort Label ENC', 'is Female'],
                'y_adv_cols' : ['Study ID ENC'],
                'train_name': 'train',
                'eval_name': 'val',

                ################
                ## Pretrain ##

                'holdout_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
                'batch_size': 64,
                
                'head_kwargs_list': [
                    {
                        'kind': 'Binary',
                        'name': 'isPediatric',
                        'y_idx': 0,
                        'weight': ispediatric_weight,
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
                    {
                        'kind': 'Binary',
                        'name': 'isFemale',
                        'y_idx': 2,
                        'weight': isfemale_weight,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
                        'num_classes': 2,
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
                    'num_epochs': IntDistribution(num_epochs_min, num_epochs_max, step=num_epochs_step),
                    'lr': FloatDistribution(0.0001, 0.05, log=True),
                    # 'lr': 0.01,
                    'weight_decay': 0,
                    'l1_reg_weight': l1_reg_weight,
                    # 'l2_reg_weight': 0.001,
                    'l2_reg_weight': l2_reg_weight,
                    'encoder_weight': encoder_weight,
                    'head_weight': head_weight,
                    'adversary_weight': adv_weight,
                    'noise_factor': 0.1,
                    'early_stopping_patience': IntDistribution(0, early_stopping_patience_max, step=early_stopping_patience_step),
                    'adversarial_mini_epochs': adversarial_mini_epochs,
                },

        }
    
    kwargs = round_kwargs_to_sig(kwargs,sig_figs=sig_figs)
    return kwargs


def round_kwargs_to_sig(val,sig_figs=2,key=None):
    if sig_figs is None:
        return val

    # for key, val in kwargs.items():
    if isinstance(val, dict):
        for k, v in val.items():
            val[k] = round_kwargs_to_sig(v, sig_figs=sig_figs, key=k)
        return val
    
    elif isinstance(val, list):
        return [round_kwargs_to_sig(v, sig_figs=sig_figs) for v in val]
    
    elif isinstance(val, float):
        new_val = round_to_sig(val, sig_figs=sig_figs)
        if np.abs(new_val - val) > 1e-6:
            print(f"Rounded {key} from {val} to {new_val}")
        return new_val
    else:
        return val


def convert_model_kwargs_list_to_dict(kwargs,style=2):

    if 'head_kwargs_dict' not in kwargs:
        kwargs['head_kwargs_dict'] = {}
    if 'adv_kwargs_dict' not in kwargs:
        kwargs['adv_kwargs_dict'] = {}

    for key in ['head_kwargs_list','adv_kwargs_list']:
        if key not in kwargs:
            # print(f"Key {key} not in kwargs")
            continue
        val = kwargs[key]
        new_key = key.replace('kwargs_list','kwargs_dict')
        if style == 0:
            kwargs[new_key].update({f'{i}': v for i, v in enumerate(val)})
        elif style == 1:
            kwargs[new_key].update({f'{v["name"]}': v for i, v in enumerate(val)})
        elif style == 2:                
            kwargs[new_key].update({f'{v["kind"]}_{v["name"]}': v for i, v in enumerate(val)})

        del kwargs[key]

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
        # print('base')
        return trial._suggest(name, obj)
    elif isinstance(obj, dict):
        # print('dict')
        return {k: convert_distributions_to_suggestion(v,trial,name=k) for k, v in obj.items()}
    elif isinstance(obj, list):
        # print('list')
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
                if len(v) == 0:
                    items.append((new_key, v))
                elif isinstance(v[0], dict):
                    for i, item in enumerate(v):
                        items.extend(flatten_dict(item, new_key + sep + str(i), sep=sep).items())            
                else:
                    items.append((new_key, v))
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
                    'eval_kwargs','train_kwargs__eval_funcs','run_training','encoder_kwargs__hidden_size','overwrite_existing_kwargs',\
                    'load_model_loc']
        new_ignore_keys_list = ['y_head_cols','head_kwargs_dict__Binary_isFemale','eval_name','train_name']
                                # 'head_kwargs_dict__MultiClass_Cohort','head_kwargs_dict__Binary_isPediatric',\
                                # 'head_kwargs_dict__MultiClass_Cohort'
                                
        ignore_keys_list.extend(new_ignore_keys_list)

    diff_clean = {}
    for key, val in diff.items():
        for ignore_key in ignore_keys_list:
            if key.startswith(ignore_key):
                break
        else:
            diff_clean[key] = val

    return diff_clean



########################################################################################
########################################################################################

if __name__ == '__main__':
    cleanup_runs()
    # run_id_list = get_run_id_list()
    # print(run_id_list)
    # for run_id in run_id_list:
    #     reuse_run(run_id)
    #