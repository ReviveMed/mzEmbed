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
    raise NotImplementedError("This function needs to be updated, replacing kwargs with original_kwargs")

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



def add_runs_to_study(study,run_id_list=None,study_kwargs=None,objective_func=None,limit_add=-1):
    if run_id_list is None:
        run_id_list = get_run_id_list()

    print('number of runs: ', len(run_id_list))
    
    if limit_add>-1:
        if len(run_id_list) > limit_add:
            run_id_list = run_id_list[:limit_add]

    if study_kwargs is None:
        study_kwargs = make_kwargs()
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
        'train_kwargs__l1_reg_weight': 0,
        'train_kwargs__optimizer_name': 'adam',
        'train_kwargs__adversarial_start_epoch': -1,
    }

    return default_val_dict



def reuse_run(run_id,study_kwargs=None,objective_func=None,ignore_keys_list=None,
              default_kwarg_val_dict=None,verbose=1):
    if study_kwargs is None:
        study_kwargs = make_kwargs()

    if ignore_keys_list is None:
        ignore_keys_list = ['run_evaluation','save_latent_space','plot_latent_space_cols','plot_latent_space',\
            'eval_kwargs','train_kwargs__eval_funcs','run_training','encoder_kwargs__hidden_size','overwrite_existing_kwargs',\
            'load_model_loc','y_head_cols','head_kwargs_dict__Binary_isFemale','eval_name','train_name','head_kwargs_dict__Regression_Age',\
            'study_info_dict']


    if default_kwarg_val_dict is None:
        default_kwarg_val_dict = get_default_kwarg_val_dict()

    run = neptune.init_run(project='revivemed/RCC',
                        api_token=NEPTUNE_API_TOKEN,
                        with_id=run_id,
                        mode='read-only')
    setup_id = 'pretrain'
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

def get_study_objective_directions(study_info_dict):
    objective_keys = get_study_objective_keys(study_info_dict)
    directions = [study_info_dict['objectives'][k]['direction'] for k in objective_keys]
    return directions


def objective_func3(run_id,data_dir,recompute_eval=False,objective_keys=None,objectives_info_dict=None):

    if objectives_info_dict is None:
        objectives_info_dict = {}

    if objective_keys is None:
        objective_keys = ['reconstruction_loss','Binary_isPediatric','MultiClass_Cohort Label','MultiClass_Adv StudyID','Binary_isFemale','Regression_Age']
    objective_keys = sorted(objective_keys)

    default_objective_vals_dict = {
        'reconstruction_loss': 99999,
        'Binary_isPediatric': 0.5,
        'MultiClass_Cohort Label': 0.5,
        'MultiClass_Adv StudyID': 0.5,
        'Binary_isFemale': 0.5,
        'Regression_Age': 9999
    }

    run = neptune.init_run(project='revivemed/RCC',
                    api_token=NEPTUNE_API_TOKEN,
                    with_id=run_id,
                    capture_stdout=False,
                    capture_stderr=False,
                    capture_hardware_metrics=False,
                    mode='read-only')


    obj_vals = []
    try:
        # if 'pretrain' in run.get_structure()
        pretrain_output = run['pretrain'].fetch()
    
    except NeptuneException as e:
        print(f"Error with run {run_id}: {e}")
        run.stop()
        raise ValueError(f"Error with run {run_id}: {e}")

    run_struc = run.get_structure()

    if not 'pretrain' in run_struc:
        run.stop()
        raise ValueError(f"Error with run {run_id}: no pretrain output")
    
    if not 'eval' in run_struc['pretrain']:
        run.stop()
        raise ValueError(f"Error with run {run_id}: no eval output")
    
    if not 'val' in run_struc['pretrain']['eval']:
        run.stop()
        raise ValueError(f"Error with run {run_id}: no val output")

    # eval_res = run['pretrain/eval/val'].fetch()
    eval_loc = 'pretrain/eval/val'
    eval_res = run_struc['pretrain']['eval']['val']

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
                obj_val = default_objective_vals_dict[objective_key]    
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

        # elif len(eval_key_matches) > 1:
        #     print(f'Objective {objective_key} will be the average of values from {eval_key_matches}')
        #     obj_vals = []
        #     for eval_key in eval_key_matches:
        #         if isinstance(eval_res[eval_key],dict):
        #             sub_keys = list(eval_res[eval_key].keys())
        #             if len(sub_keys) == 1:
        #                 obj_val = eval_res[eval_key][sub_keys[0]]
        #             else:
        #                 # get the average of the sub_keys
        #                 obj_val = np.mean([eval_res[eval_key][k] for k in sub_keys])

        #         else:
        #             obj_val = eval_res[eval_key]
        #             # if objective_key == 'reconstruction_loss':
        #                 # obj_val = np.log10(obj_val)
        #         obj_vals.append(obj_val)
        #     obj_val = np.mean(obj_vals)
        # else:
        #     print(f'set {objective_key} obj_val to nan')
        #     obj_val = float('nan')


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

    run.stop()
    return tuple(obj_vals)


########################################################################################
# if the variable is single value, then use the single value, if it is None, then use the min,max,step with optuna, if it is a list, then pick from the list

def make_kwargs_set(sig_figs=2,
                    encoder_kind='VAE',
                    activation_func= 'leakyrelu',
                    head_kwargs_dict={},
                    adv_kwargs_dict={},

                    batch_size=None, batch_size_min=16,batch_size_max=64,batch_size_step=16,
                    noise_factor=None, noise_factor_min=0.01, noise_factor_max=0.1, noise_factor_step=0.01,
                    latent_size=None, latent_size_min=4, latent_size_max=128, latent_size_step=4,
                    hidden_size=None, hidden_size_min=16, hidden_size_max=64, hidden_size_step=16,
                    hidden_size_mult=None, hidden_size_mult_min=0, hidden_size_mult_max=2, hidden_size_mult_step=0.5,
                    num_hidden_layers=None, num_hidden_layers_min=1, num_hidden_layers_max=10, num_hidden_layers_step=1,
                    
                    num_attention_heads=None, num_attention_heads_min=1, num_attention_heads_max=5, num_attention_heads_step=1,
                    num_decoder_layers=None, num_decoder_layers_min=1, num_decoder_layers_max=5, num_decoder_layers_step=1,
                    decoder_embed_dim=None, decoder_embed_dim_min=4, decoder_embed_dim_max=64, decoder_embed_dim_step=4,
                    default_hidden_fraction=None, default_hidden_fraction_min=0, default_hidden_fraction_max=0.5, default_hidden_fraction_step=0.1,

                    dropout_rate=None, dropout_rate_min=0, dropout_rate_max=0.5, dropout_rate_step=0.1,
                    encoder_weight=None, encoder_weight_min=0, encoder_weight_max=5, encoder_weight_step=0.1,
                    head_weight=None, head_weight_min=0, head_weight_max=10, head_weight_step=0.1,
                    adv_weight=None, adv_weight_min=0, adv_weight_max=10, adv_weight_step=0.1,

                    task_head_weight=None, task_head_weight_min=0, task_head_weight_max=10, task_head_weight_step=0.1,
                    task_adv_weight=None, task_adv_weight_min=0, task_adv_weight_max=10, task_adv_weight_step=0.1,
                    task_hidden_size=None, task_hidden_size_min=4, task_hidden_size_max=64, task_hidden_size_step=4,
                    task_num_hidden_layers=None, task_num_hidden_layers_min=1, task_num_hidden_layers_max=5, task_num_hidden_layers_step=1,

                    weight_decay=None, weight_decay_min=0, weight_decay_max=0.0005, weight_decay_step=0.00001, weight_decay_log=False,
                    l1_reg_weight=None, l1_reg_weight_min=0, l1_reg_weight_max=0.01, l1_reg_weight_step=0.0001, l1_reg_weight_log=False,
                    l2_reg_weight=None, l2_reg_weight_min=0, l2_reg_weight_max=0.01, l2_reg_weight_step=0.0001, l2_reg_weight_log=False,
                    num_epochs=None, num_epochs_min=50, num_epochs_max=300, num_epochs_step=10, num_epochs_log=False,
                    learning_rate=None, learning_rate_min=0.0001, learning_rate_max=0.05, learning_rate_step=None,learning_rate_log=True,
                    early_stopping_patience=None, early_stopping_patience_min=0, early_stopping_patience_max=50, early_stopping_patience_step=5,
                    adversarial_start_epoch=None, adversarial_start_epoch_min=-1, adversarial_start_epoch_max=10, adversarial_start_epoch_step=2,
                    ):


    if batch_size is None:
        batch_size = IntDistribution(batch_size_min, batch_size_max, step=batch_size_step)
    elif isinstance(batch_size, list):
        batch_size = CategoricalDistribution(batch_size)

    if noise_factor is None:
        noise_factor = FloatDistribution(noise_factor_min, noise_factor_max, step=noise_factor_step)
    elif isinstance(noise_factor, list):
        noise_factor = CategoricalDistribution(noise_factor)

    if encoder_kind in ['AE','VAE']:
        if latent_size is None:
            latent_size = IntDistribution(latent_size_min, latent_size_max, step=latent_size_step)
        elif isinstance(latent_size, list):
            latent_size = CategoricalDistribution(latent_size)

    if hidden_size is None:
        hidden_size = IntDistribution(hidden_size_min, hidden_size_max, step=hidden_size_step)
    elif isinstance(hidden_size, list):
        hidden_size = CategoricalDistribution(hidden_size)
    elif hidden_size == -1:
        if (hidden_size_mult is None):
            hidden_size_mult = FloatDistribution(hidden_size_mult_min, hidden_size_mult_max, step=hidden_size_mult_step)
        elif isinstance(hidden_size_mult, list):
            hidden_size_mult = CategoricalDistribution(hidden_size_mult)

    if num_hidden_layers is None:
        num_hidden_layers = IntDistribution(num_hidden_layers_min, num_hidden_layers_max, step=num_hidden_layers_step)
    elif isinstance(num_hidden_layers, list):
        num_hidden_layers = CategoricalDistribution(num_hidden_layers)

    if encoder_kind in ['metabFoundation']:

        if num_attention_heads is None:
            num_attention_heads = IntDistribution(num_attention_heads_min, num_attention_heads_max, step=num_attention_heads_step)
        elif isinstance(num_attention_heads, list):
            num_attention_heads = CategoricalDistribution(num_attention_heads)

        if num_decoder_layers is None:
            num_decoder_layers = IntDistribution(num_decoder_layers_min, num_decoder_layers_max, step=num_decoder_layers_step)
        elif isinstance(num_decoder_layers, list):
            num_decoder_layers = CategoricalDistribution(num_decoder_layers)

        if decoder_embed_dim is None:
            decoder_embed_dim = IntDistribution(decoder_embed_dim_min, decoder_embed_dim_max, step=decoder_embed_dim_step)
        elif isinstance(decoder_embed_dim, list):
            decoder_embed_dim = CategoricalDistribution(decoder_embed_dim)

        if default_hidden_fraction is None:
            default_hidden_fraction = FloatDistribution(default_hidden_fraction_min, default_hidden_fraction_max, step=default_hidden_fraction_step)
        elif isinstance(default_hidden_fraction, list):
            default_hidden_fraction = CategoricalDistribution(default_hidden_fraction)

    if dropout_rate is None:
        dropout_rate = FloatDistribution(dropout_rate_min, dropout_rate_max, step=dropout_rate_step)
    elif isinstance(dropout_rate, list):
        dropout_rate = CategoricalDistribution(dropout_rate)

    if encoder_weight is None:
        encoder_weight = FloatDistribution(encoder_weight_min, encoder_weight_max, step=encoder_weight_step)
    elif isinstance(encoder_weight, list):
        encoder_weight = CategoricalDistribution(encoder_weight)

    if len(head_kwargs_dict) > 0:
        if head_weight is None:
            head_weight = FloatDistribution(head_weight_min, head_weight_max, step=head_weight_step)
        elif isinstance(head_weight, list):
            head_weight = CategoricalDistribution(head_weight)
    
    for head_kwargs_key in head_kwargs_dict.keys():
        if head_weight is None:
            head_kwargs_dict[head_kwargs_key].update({'weight': FloatDistribution(task_head_weight_min, task_head_weight_max, step=task_head_weight_step)})
        elif isinstance(head_weight, list):
            head_kwargs_dict[head_kwargs_key].update({'weight': CategoricalDistribution(task_head_weight)})
        else:
            head_kwargs_dict[head_kwargs_key].update({'weight': task_head_weight})
            
    
    else:
        head_weight = 0

    if len(adv_kwargs_dict) >0:
        if adv_weight is None:
            adv_weight = FloatDistribution(adv_weight_min, adv_weight_max, step=adv_weight_step)
        elif isinstance(adv_weight, list):
            adv_weight = CategoricalDistribution(adv_weight)
    else:
        adv_weight = 0
            


    pass



########################################################################################

def make_kwargs(sig_figs=2,encoder_kind='AE',choose_from_distribution=True):
    activation = 'leakyrelu'
    # batch_size = 64
    # batch_size = 32
    batch_size = 24
    noise_factor = 0.1

    # these are the defaults if you don't want to choose from a distribution
    if not choose_from_distribution:
        latent_size = 16
        num_hidden_layers = 2
        dropout_rate = 0.2
        encoder_weight = 1
        head_weight = 0
        adv_weight = 0
        cohort_label_weight = 0
        isfemale_weight = 0
        ispediatric_weight = 0
        age_weight = 0
        use_l1_reg = False
        use_l2_reg = False
        l2_reg_weight = 0
        l1_reg_weight = 0
        num_epochs = 100
        # lr = 0.001
        lr = 0.0005
        optimizer_name = 'adamw'
        early_stopping_patience = 0
        weight_decay = 0.00001
        adversarial_start_epoch = 0
        num_decoder_layers = -1
        decoder_embed_dim = -1
        default_hidden_fraction = 0.0

        if 'MA_Encoder' in encoder_kind:
            hidden_size = 16
            hidden_size_mult = 0
            num_attention_heads = 2
        elif 'metabFoundation' in encoder_kind:
            num_attention_heads = 2
            num_hidden_layers = 4
            num_decoder_layers = 2
            default_hidden_fraction = 0.2
            
            hidden_size = 96
            latent_size = 2*hidden_size
            decoder_embed_dim = 16
            num_epochs =300

            # head_weight = 1.0
            head_weight = 0
            adv_weight = 0
            cohort_label_weight = 1.0
            isfemale_weight = 10.0
            ispediatric_weight = 1.0
            age_weight = 0.1

        else:
            num_attention_heads = 0
            hidden_size_mult = 1.5
            hidden_size = 0


    if choose_from_distribution:
        cohort_label_weight = FloatDistribution(0,10,step=0.1) #10
        isfemale_weight = FloatDistribution(0,20,step=0.1) #20
        ispediatric_weight = FloatDistribution(0,10,step=0.1) #10
        # head_weight = FloatDistribution(0,10,step=0.1) # 10
        head_weight = 1
        # adv_weight = FloatDistribution(0,10,step=0.1) #50
        adv_weight = 0
        age_weight = FloatDistribution(0,10,step=0.1) #10
    

    if (encoder_kind in ['AE','VAE']) or ('MA_Encoder' in encoder_kind):
        if choose_from_distribution:
            latent_size = IntDistribution(4, 128, step=1)
            if encoder_kind == 'VAE':  
                activation = 'leakyrelu'
                num_hidden_layers = IntDistribution(1, 5) 
                dropout_rate = FloatDistribution(0, 0.3, step=0.15)
                use_l1_reg = False
                use_l2_reg = False
                l2_reg_weight = 0 # loss explodes if not 0
                l1_reg_weight = 0
                num_attention_heads = 0
                hidden_size_mult = 1.5
                hidden_size = 0


            elif 'MA_Encoder' in encoder_kind:
                activation = 'relu'
                num_attention_heads = IntDistribution(2, 5, step=1)
                num_hidden_layers = IntDistribution(2, 4, step=2)
                hidden_size = IntDistribution(16, 64, step=16)
                hidden_size_mult = 0
                use_l1_reg =False
                use_l2_reg = False
                l2_reg_weight = 0
                l1_reg_weight = 0
            else:
                activation = 'leakyrelu'
                num_hidden_layers = IntDistribution(1, 10)
                dropout_rate = FloatDistribution(0, 0.5, step=0.1)
                # use_l1_reg = CategoricalDistribution([True, False])
                # use_l2_reg = CategoricalDistribution([True, False])
                l2_reg_weight = FloatDistribution(0, 0.01, step=0.0001)
                l1_reg_weight = FloatDistribution(0, 0.01, step=0.0001)
                num_attention_heads = 0
                hidden_size_mult = 1.5
                hidden_size = 0
            
            encoder_weight = FloatDistribution(0,5,step=0.1)

        encoder_kwargs = {
                    'activation': activation,
                    'latent_size': latent_size,
                    'num_hidden_layers': num_hidden_layers,
                    'dropout_rate': dropout_rate,
                    'use_batch_norm': False,
                    'hidden_size': hidden_size,
                    'hidden_size_mult' : hidden_size_mult,
                    'num_attention_heads': num_attention_heads,
                    'num_decoder_layers': num_decoder_layers,
                    'decoder_embed_dim': decoder_embed_dim,
                    'default_hidden_fraction': default_hidden_fraction
                    }
        

        num_epochs_min = 50
        num_epochs_max = 300
        num_epochs_step = 10
        early_stopping_patience_step = 10
        early_stopping_patience_max = 50

    elif encoder_kind == 'metabFoundation':
        if choose_from_distribution:
            num_attention_heads = IntDistribution(2, 5, step=1)
            num_hidden_layers = IntDistribution(2, 4, step=2)
            dropout_rate = FloatDistribution(0, 0.5, step=0.1)
            hidden_size = IntDistribution(16, 64, step=16)
        
        encoder_kwargs = {
                    'num_attention_heads': num_attention_heads,
                    'hidden_size': hidden_size,
                    'latent_size': latent_size,
                    'num_hidden_layers': num_hidden_layers,
                    'dropout_rate': dropout_rate,
                    }

        encoder_weight = 1
        num_epochs_min = 10
        num_epochs_max = 100
        num_epochs_step = 10
        early_stopping_patience_step = 5
        early_stopping_patience_max = 20
        l2_reg_weight = 0
        l1_reg_weight = 0


    elif encoder_kind == 'TGEM_Encoder':
        if choose_from_distribution:
            n_head = IntDistribution(2, 5, step=1)
            n_layers = IntDistribution(2, 3, step=1)
            dropout_rate = FloatDistribution(0, 0.5, step=0.1)
        
        encoder_kwargs = {
                    'activation': 'linear',
                    'n_head': n_head,
                    'n_layers': n_layers,
                    'dropout_rate': dropout_rate,
                    }
        
        encoder_weight = 0
        num_epochs_min = 10
        num_epochs_max = 50
        num_epochs_step = 1
        early_stopping_patience_step = 5
        early_stopping_patience_max = 20
        l2_reg_weight = 0
        l1_reg_weight = 0


    if choose_from_distribution:
        num_epochs = IntDistribution(num_epochs_min, num_epochs_max, step=num_epochs_step)
        lr = FloatDistribution(0.0001, 0.05, log=True)
        # optimizer_name = CategoricalDistribution(['adam','adamw'])
        optimizer_name = 'adamw'
        weight_decay = FloatDistribution(0, 0.0005, step=0.00001)
        early_stopping_patience = IntDistribution(0, early_stopping_patience_max, step=early_stopping_patience_step)
        adversarial_start_epoch = IntDistribution(-1, 10, step=2)



    kwargs = {
                ################
                ## General ##
                'encoder_kind': encoder_kind,
                'encoder_kwargs': encoder_kwargs,
                'other_size': 1,
                'y_head_cols' : ['is Pediatric','Cohort Label ENC', 'is Female','Age'],
                'y_adv_cols' : ['Study ID ENC'],
                'train_name': 'train',
                'eval_name': 'val',

                ################
                ## Pretrain ##

                'holdout_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
                'batch_size': batch_size,
                
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
                    {
                        'kind': 'Regression',
                        'name': 'Age',
                        'y_idx': 3,
                        'weight': age_weight,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
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


                'fit_kwargs': {
                    'optimizer_name': optimizer_name,
                    'num_epochs': num_epochs,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    # 'use_l1_reg': use_l1_reg,
                    # 'use_l2_reg': use_l2_reg,
                    'l1_reg_weight': l1_reg_weight,
                    'l2_reg_weight': l2_reg_weight,
                    'encoder_weight': encoder_weight,
                    'head_weight': head_weight,
                    'adversary_weight': adv_weight,
                    'noise_factor': noise_factor,
                    'early_stopping_patience': early_stopping_patience,
                    # 'adversarial_mini_epochs': adversarial_mini_epochs,
                    'adversarial_start_epoch': adversarial_start_epoch
                },

        }
    
    kwargs = round_kwargs_to_sig(kwargs,sig_figs=sig_figs)
    kwargs = convert_model_kwargs_list_to_dict(kwargs,style=2)
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
                    'eval_kwargs','train_kwargs__eval_funcs','fit_kwargs__eval_funcs','run_training','encoder_kwargs__hidden_size','overwrite_existing_kwargs',\
                    'load_model_loc','study_info_dict']
        new_ignore_keys_list = ['y_head_cols','head_kwargs_dict__Binary_isFemale','eval_name','train_name',
                                'head_kwargs_dict__Regression_Age']
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
