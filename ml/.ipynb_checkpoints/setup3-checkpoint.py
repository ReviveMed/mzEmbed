# setup a neptune run


import torch
import pandas as pd
import numpy as np
import os
import json
from models import get_model, Binary_Head, Dummy_Head, MultiClass_Head, MultiHead, Regression_Head, Cox_Head, get_encoder

from train4 import CompoundDataset, train_compound_model, get_end_state_eval_funcs, evaluate_compound_model, create_dataloaders, create_dataloaders_old


import neptune
from neptune.utils import stringify_unsupported
from utils_neptune import check_neptune_existance, start_neptune_run, convert_neptune_kwargs, neptunize_dict_keys, get_latest_dataset
from neptune_pytorch import NeptuneLogger
from viz import generate_latent_space, generate_umap_embedding, generate_pca_embedding
from misc import assign_color_map, get_color_map, get_clean_batch_sz
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from neptune.exceptions import NeptuneException


# set up neptune


def setup_neptune_run(data_dir,setup_id,with_run_id=None,run=None,
                      neptune_mode='async',
                      yes_logging = False,
                      project_id = 'revivemed/RCC',
                      save_kwargs_to_neptune=False,
                      restart_run=False,
                      neptune_api_token=None,
                      tags=['v3.3'],**kwargs):
    print(setup_id)
    print(kwargs)

    if run is None:
        run, is_run_new = start_neptune_run(with_run_id=with_run_id,
                                            neptune_mode=neptune_mode,
                                            tags=tags,
                                            project_id=project_id,
                                            api_token=neptune_api_token,
                                            yes_logging=yes_logging)
        ret_run_id = True
    else:
        is_run_new = False
        ret_run_id = False
    run['sys/failed'] = False
    run["info/state"] = 'Active'
    if not is_run_new:
        # setup_is_new = not check_neptune_existance(run,f'{setup_id}/kwargs')
        # if setup_is_new:
        setup_is_new = not check_neptune_existance(run,f'{setup_id}/original_kwargs')
    else:
        setup_is_new = False
    
    print(f'is_run_new: {is_run_new}, setup_is_new: {setup_is_new}')

    ##### New addition
    if (not setup_is_new) and restart_run:
        del run[f'{setup_id}']
        run.wait()
        setup_is_new = True
    ##### ##### #####

    if (not is_run_new) and (not setup_is_new):
        overwrite_existing_kwargs = kwargs.get('overwrite_existing_kwargs', False)
        update_existing_kwargs = kwargs.get('update_existing_kwargs', False)
        ignore_existing_kwargs = kwargs.get('ignore_existing_kwargs', False)
        if overwrite_existing_kwargs:
            print(f'Overwriting existing {setup_id} in run {run["sys/id"].fetch()}')
            existing_kwargs = run[f'{setup_id}/original_kwargs'].fetch()
            existing_kwargs = convert_neptune_kwargs(existing_kwargs)
            # run[f'{setup_id}/original_kwargs'] = stringify_unsupported(kwargs)
            #TODO: log the existing kwargs to a file
            new_kwargs = {**existing_kwargs}
            new_kwargs.update(kwargs)
            kwargs = new_kwargs


    run_id = run["sys/id"].fetch()
    try:
        print(f'initalize run {run_id}')
        load_model_from_run_id = kwargs.get('load_model_from_run_id', None)
        load_model_loc = kwargs.get('load_model_loc', False)
        load_encoder_loc = kwargs.get('load_encoder_loc', load_model_loc)
        if load_model_loc:
            print('override the load_encoder_loc with load_model_loc')
            load_encoder_loc = load_model_loc
        load_head_loc = kwargs.get('load_head_loc', load_model_loc)
        load_adv_loc = kwargs.get('load_adv_loc', load_model_loc)
        y_head_cols = kwargs.get('y_head_cols', [])
        y_adv_cols = kwargs.get('y_adv_cols', [])
        
        if load_model_from_run_id:
            pretrained_run = neptune.init_run(project=project_id,
                                            api_token=neptune_api_token,
                                            with_id=load_model_from_run_id,
                                            capture_stdout=False,
                                            capture_stderr=False,
                                            capture_hardware_metrics=False,
                                            mode='read-only')
            
            
            # existing_kwargs = pretrained_run[f'{setup_id}/original_kwargs'].fetch()
            # existing_kwargs = convert_neptune_kwargs(existing_kwargs)
            # # run[f'{setup_id}/original_kwargs'] = stringify_unsupported(kwargs)
            # #TODO: log the existing kwargs to a file
            # new_kwargs = {**existing_kwargs}
            # new_kwargs.update(kwargs)
            # kwargs = new_kwargs

        else:
            pretrained_run = run

        if isinstance(y_head_cols, dict):
            y_head_cols = y_head_cols.values()
        if isinstance(y_adv_cols, dict):
            y_adv_cols = y_adv_cols.values()

        if not isinstance(y_head_cols, list):
            y_head_cols = [y_head_cols]

        if not isinstance(y_adv_cols, list):
            y_adv_cols = [y_adv_cols]
        
        if load_encoder_loc:
            print('loading pretrained encoder info, overwriting encoder_kwargs')
            load_kwargs = pretrained_run[f'{load_encoder_loc}/original_kwargs'].fetch()
            load_kwargs = convert_neptune_kwargs(load_kwargs)
            # kwargs['encoder_kwargs'].update(load_kwargs['encoder_kwargs'])
            if 'encoder_kind' in kwargs:
                if kwargs['encoder_kind'] != load_kwargs['encoder_kind']:
                    raise ValueError(f'Encoder kind mismatch: {kwargs["encoder_kind"]} vs {load_kwargs["encoder_kind"]}')
            else:
                kwargs['encoder_kind'] = load_kwargs['encoder_kind']
            
            encoder_kwargs = load_kwargs.get('encoder_kwargs', {})
            fit_kwargs = kwargs.get('fit_kwargs', {})
            if 'dropout_rate' in fit_kwargs:
                encoder_kwargs['dropout_rate'] = fit_kwargs['dropout_rate']
            elif 'encoder_kwargs' in kwargs:
                if 'dropout_rate' in kwargs['encoder_kwargs']:
                    encoder_kwargs['dropout_rate'] = kwargs['encoder_kwargs']['dropout_rate']
          
            # if overwrite_existing_kwargs:
            # encoder_kwargs.update(kwargs.get('encoder_kwargs', {}))
            kwargs['encoder_kwargs'] = encoder_kwargs
            print('encoder_kwargs:', kwargs['encoder_kwargs'])

        if load_head_loc:
            print('loading pretrained heads, overwriting head_kwargs_list')
            load_kwargs = pretrained_run[f'{load_head_loc}/original_kwargs'].fetch()
            kwargs['head_kwargs_dict'] = load_kwargs.get('head_kwargs_dict', {})
            kwargs['head_kwargs_list'] = eval(load_kwargs.get('head_kwargs_list', '[]'))
            # assert len(kwargs['head_kwargs_list']) <= len(y_head_cols)
            kwargs['y_head_cols'] = eval(load_kwargs.get('y_head_cols', []))
            y_head_cols = kwargs['y_head_cols']

        if load_adv_loc:
            print('loading pretrained advs, overwriting adv_kwargs_list')
            load_kwargs = pretrained_run[f'{load_adv_loc}/original_kwargs'].fetch()
            kwargs['adv_kwargs_dict'] = load_kwargs.get('adv_kwargs_dict', {})
            kwargs['adv_kwargs_list'] = eval(load_kwargs.get('adv_kwargs_list', '[]'))
            # assert len(kwargs['adv_kwargs_list']) <= len(y_adv_cols)
            kwargs['y_adv_cols'] = eval(load_kwargs.get('y_adv_cols', []))
            y_adv_cols = kwargs['y_adv_cols']

        if save_kwargs_to_neptune:
            #Set to True for debugging
            if check_neptune_existance(run,f'{setup_id}/kwargs'):
                if not overwrite_existing_kwargs:
                    raise ValueError(f'{setup_id} already exists in run {run_id} and overwrite_existing_kwargs is False')
                else:
                    print(f'Overwriting existing {setup_id} in run {run_id}')
                    del run[f'{setup_id}/kwargs']
                    run[f'{setup_id}/kwargs'] = stringify_unsupported(kwargs)
            else:
                run[f'{setup_id}/kwargs'] = stringify_unsupported(kwargs)
        
        if is_run_new or setup_is_new:
            run[f'{setup_id}/original_kwargs'] = stringify_unsupported(kwargs)

        local_dir = kwargs.get('local_dir', f'~/output')
        local_dir = os.path.expanduser(local_dir)
        save_dir = f'{local_dir}/{run_id}'
        os.makedirs(save_dir, exist_ok=True)

        y_code_keys = []
        y_codes = kwargs.get('y_codes', None)
        if (y_codes is None) and (load_model_loc):
            if 'y_codes_keys' in pretrained_run[f'{load_model_loc}/datasets'].fetch():
                y_code_keys = pretrained_run[f'{load_model_loc}/datasets/y_codes_keys'].fetch_values()['value'].to_list()
                print('existing y_code_keys:', y_code_keys)
                y_codes = {}
                for key in y_code_keys:
                    if key in y_codes.keys():
                        continue
                    y_codes[key] = pretrained_run[f'{load_model_loc}/datasets/y_codes/{key}'].fetch_values()['value'].to_list()
        else:
            y_code_keys = []

        if load_model_from_run_id:
            pretrain_save_dir = f'{local_dir}/{load_model_from_run_id}'
            os.makedirs(pretrain_save_dir, exist_ok=True)
            os.makedirs(os.path.join(pretrain_save_dir,load_encoder_loc), exist_ok=True)
            
            pretrain_local_path = f'{pretrain_save_dir}/{load_encoder_loc}/encoder_state_dict.pth'
            if not os.path.exists(pretrain_local_path):
                pretrained_run[f'{load_encoder_loc}/models/encoder_state_dict'].download(pretrain_local_path)
            pretrained_run.stop()
        else:
            pretrain_save_dir = f'{local_dir}/{run_id}'


    except Exception as e:
        run['sys/tags'].add('init failed')
        run["info/state"] = 'Inactive'
        run.stop()
        raise e

    ####################################
    ##### Load the Data ######
    run_training = kwargs.get('run_training', True)
    run_evaluation = kwargs.get('run_evaluation', True)
    save_latent_space = kwargs.get('save_latent_space', False)
    use_subset_of_features = kwargs.get('use_subset_of_features', False)
    random_seed = kwargs.get('random_seed', 42)
    remove_y_nans = kwargs.get('remove_y_nans', False)
    remove_y_nans_strict = kwargs.get('remove_y_nans_strict', False)
    np.random.seed(random_seed)
    
    try:
        print('loading data')
        if os.path.exists(f'{data_dir}/hash.txt'):
            dataset_hash = open(f'{data_dir}/hash.txt','r').read()
            run[f'{setup_id}/datasets/hash'] = dataset_hash

        X_fit_file = kwargs.get('X_fit_file',None)
        y_fit_file = kwargs.get('y_fit_file',None)
        X_eval_file = kwargs.get('X_eval_file',None)
        y_eval_file = kwargs.get('y_eval_file',None)
        
        train_name = kwargs.get('train_name', 'train')
        eval_name = kwargs.get('eval_name', 'val')

        X_filename_prefix = kwargs.get('X_filename_prefix',None)
        if (X_filename_prefix is None) and ('X_filename' in kwargs):
            print('Warning argument "X_filename" is depreciated')
            X_filename_prefix = kwargs.get('X_filename', 'X_pretrain')
        elif X_filename_prefix is None:
            X_filename_prefix = 'X_pretrain'

        y_filename_prefix = kwargs.get('y_filename_prefix', None)
        if (y_filename_prefix is None) and ('y_filename' in kwargs):
            print('Warning argument "y_filename" is depreciated')
            y_filename_prefix = kwargs.get('y_filename', 'y_pretrain')
        elif y_filename_prefix is None:
            y_filename_prefix = X_filename_prefix.replace('X','y')


        if X_fit_file is None:
            X_fit_file = f'{data_dir}/{X_filename_prefix}_{train_name}.csv'
        if y_fit_file is None:
            y_fit_file = f'{data_dir}/{y_filename_prefix}_{train_name}.csv'
        if X_eval_file is None:
            X_eval_file = f'{data_dir}/{X_filename_prefix}_{eval_name}.csv'
        if y_eval_file is None:
            y_eval_file = f'{data_dir}/{y_filename_prefix}_{eval_name}.csv'
        
        # nan_filename = kwargs.get('nan_filename', 'nans')
        X_size = None
        fit_dataset = None
        eval_dataset = None

        if run_training or run_evaluation:
            X_data_train = pd.read_csv(X_fit_file, index_col=0)
            y_data_train = pd.read_csv(y_fit_file, index_col=0)
            X_size = X_data_train.shape[1]
            run[f'{setup_id}/datasets/X_fit'].track_files(X_fit_file)
            run[f'{setup_id}/datasets/y_fit'].track_files(y_fit_file)
        # nan_data = pd.read_csv(f'{data_dir}/{nan_filename}_{train_name}.csv', index_col=0)
        else:
            X_data_train = pd.DataFrame()
            y_data_train = pd.DataFrame()

        if run_evaluation or save_latent_space:
            X_data_eval = pd.read_csv(X_eval_file, index_col=0)
            y_data_eval = pd.read_csv(y_eval_file, index_col=0)
            run[f'{setup_id}/datasets/X_eval'].track_files(X_eval_file)
            run[f'{setup_id}/datasets/y_eval'].track_files(y_eval_file)
            X_size = X_data_eval.shape[1]
        # nan_data = pd.read_csv(f'{data_dir}/{nan_filename}_{eval_name}.csv', index_col=0)
        else:
            X_data_eval = pd.DataFrame()
            y_data_eval = pd.DataFrame()

        if use_subset_of_features:
            feature_subset = kwargs.get('feature_subset', 0.25)
            if isinstance(feature_subset, float):
                # choose random subset of features
                feature_subset = np.random.choice(X_data_train.columns, int(feature_subset*X_data_train.shape[1]), replace=False)
            if X_data_train.shape[1] > 0:
                X_data_train = X_data_train[feature_subset]
            if X_data_eval.shape[1] > 0:
                X_data_eval = X_data_eval[feature_subset]
            X_size = len(feature_subset)

        if remove_y_nans:
            print('removing rows with all nan in y')
            if y_data_train.shape[0] > 0:
                y_data_train.dropna(subset=y_head_cols+y_adv_cols, how='all', axis=0, inplace=True)
                X_data_train = X_data_train.loc[y_data_train.index].copy()
            if y_data_eval.shape[0] > 0:
                y_data_eval.dropna(subset=y_head_cols+y_adv_cols, how='all', axis=0, inplace=True)
                X_data_eval = X_data_eval.loc[y_data_eval.index].copy()
        if remove_y_nans_strict:
            print('removing rows with any nan in y')
            if y_data_train.shape[0] > 0:
                y_data_train.dropna(subset=y_head_cols+y_adv_cols, how='any', axis=0, inplace=True)
                X_data_train = X_data_train.loc[y_data_train.index].copy()
            if y_data_eval.shape[0] > 0:
                y_data_eval.dropna(subset=y_head_cols+y_adv_cols, how='any', axis=0, inplace=True)
                X_data_eval = X_data_eval.loc[y_data_eval.index].copy()

        ####################################
        ##### Create the DataLoaders ######
        batch_size = kwargs.get('batch_size', 32)
        holdout_frac = kwargs.get('holdout_frac', 0)
        yes_clean_batches = kwargs.get('yes_clean_batches', False)
        fit_kwargs = kwargs.get('fit_kwargs', {})
        if (not fit_kwargs) and 'train_kwargs' in kwargs:
            print('Warning: using train_kwargs instead of fit_kwargs. train_kwargs is depreciated')
        early_stopping_patience = fit_kwargs.get('early_stopping_patience', 0)
        scheduler_kind = fit_kwargs.get('scheduler_kind', None)


        if (holdout_frac > 0) and (early_stopping_patience < 1) and (scheduler_kind is None):
            # raise ValueError('holdout_frac > 0 and early_stopping_patience < 1 is not recommended')
            print('holdout_frac > 0 and early_stopping_patience < 1 is not recommended, set hold out frac to 0')
            print('UNLESS you are using a scheduler, in which case the holdout_frac is used for the scheduler')
            holdout_frac = 0

        if run_training or run_evaluation:
            fit_dataset = CompoundDataset(X_data_train,y_data_train[y_head_cols], y_data_train[y_adv_cols],y_codes=y_codes)
            y_codes = fit_dataset.y_codes
            for key, val in y_codes.items():
                if key in y_code_keys:
                    continue
                if len(val) < 2:
                    continue
                run[f'{setup_id}/datasets/y_codes_keys'].append(key)
                run[f'{setup_id}/datasets/y_codes/{key}'].extend(val)
                y_code_keys.append(key)
            eval_dataset = CompoundDataset(X_data_eval,y_data_eval[y_head_cols], y_data_eval[y_adv_cols],y_codes=y_codes)
            if yes_clean_batches:
                print('attempt to clean batch size so last batch is large')
                batch_size = get_clean_batch_sz(X_data_train.shape[0], batch_size)
            # stratify on the adversarial column (stratify=2)
            # this is probably not the most memory effecient method, would be better to do stratification before creating the dataset
            # fit_loader_dct = create_dataloaders(fit_dataset, batch_size, holdout_frac, set_name=train_name, stratify=2)
            fit_loader_dct = create_dataloaders_old(fit_dataset, batch_size, holdout_frac, set_name=train_name)
            eval_loader_dct = create_dataloaders(eval_dataset, batch_size, set_name = eval_name)
            eval_loader_dct.update(fit_loader_dct)

            ########################################
            ##### Custom Evaluation for OS predictions
            #Eval on the EVER OS data when 
            #the model is trained on the NIVO OS data
            if len(y_head_cols) == len(y_adv_cols):

                head_kwargs_dict = kwargs.get('head_kwargs_dict', {})
                if head_kwargs_dict:
                    head_kwargs_list = [head_kwargs_dict[k] for k in head_kwargs_dict.keys()]
                else:
                    head_kwargs_list = kwargs.get('head_kwargs_list', [])

                adv_kwargs_dict = kwargs.get('adv_kwargs_dict', {})
                if adv_kwargs_dict:
                    adv_kwargs_list = [adv_kwargs_dict[k] for k in adv_kwargs_dict.keys()]
                else:
                    adv_kwargs_list = kwargs.get('adv_kwargs_list', [])

                if len(head_kwargs_list) == len(adv_kwargs_list):
                    
                    head_kinds = [h['kind'] for h in head_kwargs_list]
                    adv_kinds = [a['kind'] for a in adv_kwargs_list]
                    adv_names = [a['name'] for a in adv_kwargs_list]
                    
                    for h_kind,a_kind,adv_name in zip(head_kinds,adv_kinds,adv_names):
                        if h_kind == a_kind:
                            print(f'Creating custom evaluation for {h_kind} on {adv_name}')
                            # y_head_cols_temp = y_head_cols.copy()
                            #TODO replace the y_head columns that correspond to h with the y_adv columns that correspond to a
                            
                            fit_dataset2 = CompoundDataset(X_data_train,y_data_train[y_adv_cols], y_data_train[y_adv_cols],y_codes=y_codes)
                            eval_dataset2 = CompoundDataset(X_data_eval,y_data_eval[y_adv_cols], y_data_eval[y_adv_cols],y_codes=y_codes)
                            fit_loader_dct2 = create_dataloaders_old(fit_dataset2, batch_size, set_name=train_name+'_'+adv_name)
                            eval_loader_dct2 = create_dataloaders(eval_dataset2, batch_size, set_name = eval_name+'_'+adv_name)
                            eval_loader_dct.update(fit_loader_dct2)
                            eval_loader_dct.update(eval_loader_dct2)

            ########################################

    except Exception as e:
        run['sys/tags'].add('data-load failed')
        run["info/state"] = 'Inactive'
        run.stop()
        raise e


    ########################################################################
    ###### Repeated Training and Evaluation

    recompute_evaluation = kwargs.get('recompute_evaluation', False)
    num_repeats = kwargs.get('num_repeats', 1)
    yes_save_model = True
    
    #######################################
    ############# Encoder Preamble
    encoder = None

    encoder_kind = kwargs.get('encoder_kind', 'AE')
    encoder_kwargs = kwargs.get('encoder_kwargs', {})
    other_input_size = kwargs.get('other_input_size', 1)
    load_model_weights = kwargs.get('load_model_weights', True)
    

    latent_size = encoder_kwargs.get('latent_size', 8)
    input_size = kwargs.get('input_size', None)
    if ('hidden_size_mult' in encoder_kwargs) and (encoder_kwargs['hidden_size_mult'] > 0):
        encoder_kwargs['hidden_size'] = int(encoder_kwargs['hidden_size_mult']*latent_size)
        # remove the hidden_size_mult key
        encoder_kwargs.pop('hidden_size_mult')
        hidden_size = encoder_kwargs['hidden_size']
    else:
        hidden_size = encoder_kwargs.get('hidden_size', -1)
    
    if input_size is None:
        try:
            input_size = run['input_size'].fetch()
        except NeptuneException:
            input_size = X_size
            run['input_size'] = input_size
    if X_size is not None:
        assert input_size == X_size
    # if input_size is None:


    if (encoder_kind == 'TGEM_Encoder'): # or (encoder_kind == 'metabFoundation'):
        latent_size = input_size
    elif encoder_kind == 'metabFoundation':
        hidden_size = encoder_kwargs.get('embed_dim',encoder_kwargs.get('hidden_size', -1))
        latent_size = 2*hidden_size
        # encoder_kwargs['latent_size'] = latent_size

    ############################################
    
    
    run_random_init = kwargs.get('run_random_init', False) # this should be redundant with load_model_weights
    upload_models_to_neptune = kwargs.get('upload_models_to_neptune', True)
    upload_models_to_gcp = kwargs.get('upload_models_to_gcp', False)
    eval_kwargs = kwargs.get('eval_kwargs', {})
    plot_latent_space = kwargs.get('plot_latent_space', '')


    if run_random_init:
        print('running with random initialization')
        load_model_weights = False

    print('run_random_init:', run_random_init)
    print('load_model_weights:', load_model_weights)

    eval_prefix = f'{setup_id}/eval'
    eval_kwargs['prefix'] = eval_prefix

    try:
        eval_recon_loss_history = run[f'{eval_prefix}/{eval_name}/reconstruction_loss'].fetch_values()
        num_existing_repeats = len(eval_recon_loss_history)
        print(f'Already computed {num_existing_repeats} train/eval repeats')
    except NeptuneException:
        num_existing_repeats = 0

    if (num_existing_repeats == 1 ) and (recompute_evaluation):
        # delete the existing evaluation results, if there is only one evaluation
        # if there is more than one evaluation, then there has been multiple fit iterations 
        # and unless they were all saved, we won't be able to recompute the all evaluations
        del run[f'{eval_prefix}/{eval_name}']
        run.wait()
        num_existing_repeats = 0

    if num_existing_repeats > num_repeats:
        print('Already computed over {} train/eval repeats'.format(num_repeats))

    num_repeats = num_repeats - num_existing_repeats
    if num_repeats < 0:
        num_repeats = 0

    for ii_repeat in range(num_repeats):
        
        if num_repeats > 1:
            print(f'{setup_id}, train/eval repeat: {num_existing_repeats+ii_repeat}')
            yes_save_model = False
            if (not run_evaluation):
                print('why are you running training multiple times without any evaluation?')

        if ii_repeat == num_repeats-1:
            yes_save_model = True


        ####################################
        ###### Create the Encoder Models ######
        try:
            if run_training or run_evaluation or save_latent_space or plot_latent_space:
                print('creating models')
                
                encoder = get_model(encoder_kind, input_size, **encoder_kwargs)


                if (load_encoder_loc) and (load_model_weights):
                    os.makedirs(os.path.join(pretrain_save_dir,load_encoder_loc), exist_ok=True)
                    local_path = f'{pretrain_save_dir}/{load_encoder_loc}/encoder_state_dict.pth'
                    if not os.path.exists(local_path):
                        run[f'{load_encoder_loc}/models/encoder_state_dict'].download(local_path)
                    print('load encoder weights from ',local_path)
                    encoder_state_dict = torch.load(local_path)
                    encoder.load_state_dict(encoder_state_dict)
                else:
                    if 'init_layers' in dir(encoder):
                        encoder.init_layers()
                    # encoder.reset_params()
                    print('encoder random initialized')

                ####################################
                ###### Create the Head Models ######

                head_kwargs_dict = kwargs.get('head_kwargs_dict', {})
                if head_kwargs_dict:
                    head_kwargs_list = [head_kwargs_dict[k] for k in head_kwargs_dict.keys()]
                else:
                    head_kwargs_list = kwargs.get('head_kwargs_list', [])

                head_list = []
                for head_kwargs in head_kwargs_list:
                    
                    head_kind = head_kwargs.get('kind', 'NA')
                    if 'input_size' not in head_kwargs:
                        head_kwargs['input_size'] = latent_size+other_input_size

                    if head_kind == 'Binary':
                        head_list.append(Binary_Head(**head_kwargs))
                    elif head_kind == 'MultiClass':
                        head_list.append(MultiClass_Head(**head_kwargs))
                    elif head_kind == 'Regression':
                        head_list.append(Regression_Head(**head_kwargs))
                    elif head_kind == 'Cox':
                        head_list.append(Cox_Head(**head_kwargs))
                    elif head_kind == 'NA':
                        head_list.append(Dummy_Head())
                    else:
                        raise ValueError(f'Invalid head_kind: {head_kind}')


                head = MultiHead(head_list)

                # confirm that the heads and adv use the correct columns
                y_head_col_array = np.array(y_head_cols)

                for h in head_list:
                    cols = y_head_col_array[h.y_idx]
                    print(f'{h.kind} {h.name} uses columns: {cols}')

                # Assign the class weights to the head and adv models so the loss is balanced
                if fit_dataset is not None:
                    #TODO: loadd the class weights from the model json
                    head.update_class_weights(fit_dataset.y_head)


                if load_head_loc and load_model_weights:
                    head_file_ids = head.get_file_ids()
                    for head_file_id in head_file_ids:
                        # local_path = f'{save_dir}/{load_head_loc}/{head_file_id}_state.pt'
                        local_path = f'{pretrain_save_dir}/{load_head_loc}/{head_file_id}_state.pt'
                        if not os.path.exists(local_path):
                            run[f'{load_head_loc}/models/{head_file_id}_state'].download(f'{pretrain_save_dir}/{load_head_loc}/{head_file_id}_state.pt')
                            run[f'{load_head_loc}/models/{head_file_id}_info'].download(f'{pretrain_save_dir}/{load_head_loc}/{head_file_id}_info.json')
                        # head_state_dict = torch.load(f'{save_dir}/{load_head_loc}/{head_file_id}_state.pt')
                    head.load_state_from_path(f'{pretrain_save_dir}/{load_head_loc}')

                ####################################
                ###### Create the Adversarial Models ######

                adv_kwargs_dict = kwargs.get('adv_kwargs_dict', {})
                if adv_kwargs_dict:
                    adv_kwargs_list = [adv_kwargs_dict[k] for k in adv_kwargs_dict.keys()]
                else:
                    adv_kwargs_list = kwargs.get('adv_kwargs_list', [])

                adv_list = []
                for adv_kwargs in adv_kwargs_list:
                    adv_kind = adv_kwargs.get('kind', 'NA')

                    if 'input_size' not in adv_kwargs:
                        adv_kwargs['input_size'] = latent_size

                    if adv_kind == 'Binary':
                        adv_list.append(Binary_Head(**adv_kwargs))
                    elif adv_kind == 'MultiClass':
                        adv_list.append(MultiClass_Head(**adv_kwargs))
                    elif adv_kind == 'Regression':
                        adv_list.append(Regression_Head(**adv_kwargs))
                    elif adv_kind == 'Cox':
                        adv_list.append(Cox_Head(**adv_kwargs))
                    elif adv_kind == 'NA':
                        adv_list.append(Dummy_Head())
                    else:
                        raise ValueError(f'Invalid adv_kind: {adv_kind}')

                adv = MultiHead(adv_list)

                y_adv_col_array = np.array(y_adv_cols)
                for a in adv_list:
                    cols = y_adv_col_array[a.y_idx]
                    print(f'{a.kind} {a.name} uses columns: {cols}')

                if fit_dataset is not None:
                    #TODO load the class weights from the model info json
                    adv.update_class_weights(fit_dataset.y_adv)

                if load_adv_loc and load_model_weights:
                    adv_file_ids = adv.get_file_ids()
                    for adv_file_id in adv_file_ids:
                        # local_path = f'{save_dir}/{load_adv_loc}/{adv_file_id}_state.pt'
                        local_path = f'{pretrain_save_dir}/{load_adv_loc}/{adv_file_id}_state.pt'
                        if not os.path.exists(local_path):
                            run[f'{load_adv_loc}/models/{adv_file_id}_state'].download(f'{pretrain_save_dir}/{load_adv_loc}/{adv_file_id}_state.pt')
                            run[f'{load_adv_loc}/models/{adv_file_id}_info'].download(f'{pretrain_save_dir}/{load_adv_loc}/{adv_file_id}_info.json')
                        # adv_state_dict = torch.load(f'{save_dir}/{load_adv_loc}/{adv_file_id}_state.pt')
                    adv.load_state_from_path(f'{pretrain_save_dir}/{load_adv_loc}')
                
        except Exception as e:
            run['sys/tags'].add('model-creation failed')
            run["info/state"] = 'Inactive'
            run.stop()
            raise e

        ####################################
        ###### Train the Models ######
        if run_training:
            try:
                print('training models')
            
                # for advanced model logging
                # if encoder_kind == 'AE' or encoder_kind == 'VAE':
                #     log_freq = 5
                # elif encoder_kind == 'TGEM_Encoder':
                #     log_freq = 1
                # else:
                #     log_freq = 0

                # if log_freq > 0:
                #     npt_logger = NeptuneLogger(
                #         run=run,
                #         model=encoder,
                #         log_model_diagram=False,
                #         log_gradients=False,
                #         log_parameters=False,
                #         log_freq=log_freq,
                #     )

                if run_random_init:
                    # encoder.reset_params()
                    encoder.init_layers()
                    head.reset_params()
                    adv.reset_params()


                fit_kwargs = kwargs.get('fit_kwargs', {})
                if (not fit_kwargs) and 'train_kwargs' in kwargs:
                    print('Warning: using train_kwargs instead of fit_kwargs. depreceate train_kwargs')
                    fit_kwargs = kwargs.get('train_kwargs',{})
                fit_kwargs['prefix'] = f'{setup_id}/fit'
                if 'train_name' not in fit_kwargs:
                    fit_kwargs['train_name'] = train_name
                encoder, head, adv = train_compound_model(fit_loader_dct, 
                                                        encoder, head, adv, 
                                                        run=run, **fit_kwargs)
                if encoder is None:
                    raise ValueError('Encoder is None after training')


                if yes_save_model:
                    # log the models
                    # run[f'{setup_id}/encoder'] = npt_logger.log_model('encoder')
                    # run[f'{setup_id}/head'] = npt_logger.log_model(head)
                    # run[f'{setup_id}/adv'] = npt_logger.log_model(adv)
                    os.makedirs(os.path.join(save_dir,setup_id), exist_ok=True)

                    # alternative way to log models
                    torch.save(encoder.state_dict(), f'{save_dir}/{setup_id}_encoder_state_dict.pth')
                    encoder.save_state_to_path(f'{save_dir}/{setup_id}')
                    encoder.save_info(f'{save_dir}/{setup_id}')
                    # torch.save(head.state_dict(), f'{save_dir}/{setup_id}_head_state_dict.pth')
                    # torch.save(adv.state_dict(), f'{save_dir}/{setup_id}_adv_state_dict.pth')
                    if upload_models_to_neptune:
                        encoder_file_id = encoder.get_file_ids()[0]
                        run[f'{setup_id}/models/encoder_state_dict'].upload(f'{save_dir}/{setup_id}_encoder_state_dict.pth')
                        # run[f'{setup_id}/models/encoder_state'].upload(f'{save_dir}/{setup_id}/{encoder_file_id}_state.pt')
                        run[f'{setup_id}/models/encoder_info'].upload(f'{save_dir}/{setup_id}/{encoder_file_id}_info.json')
                    # run[f'{setup_id}/models/head_state_dict'].upload(f'{save_dir}/{setup_id}_head_state_dict.pth')
                    # run[f'{setup_id}/models/adv_state_dict'].upload(f'{save_dir}/{setup_id}_adv_state_dict.pth')
                        run.wait()

                    if upload_models_to_gcp:
                        raise NotImplementedError('upload_models_to_gcp not implemented')

                    # New method for saving the heads and advs
                    #TODO need to test
                    # head.save_state_to_path(f'{save_dir}/{setup_id}',save_name='head')
                    # adv.save_state_to_path(f'{save_dir}/{setup_id}',save_name='adv')
                    # head.save_info(f'{save_dir}/{setup_id}',save_name='head')
                    # adv.save_info(f'{save_dir}/{setup_id}',save_name='adv')
                    # if upload_models_to_neptune:
                    #     run[f'{setup_id}/models/head_state'].upload(f'{save_dir}/{setup_id}/head_state.pt')
                    #     run[f'{setup_id}/models/adv_state'].upload(f'{save_dir}/{setup_id}/adv_state.pt')
                    #     run[f'{setup_id}/models/head_info'].upload(f'{save_dir}/{setup_id}/head_info.json')
                    #     run[f'{setup_id}/models/adv_info'].upload(f'{save_dir}/{setup_id}/adv_info.json')
                    

                    head.save_state_to_path(f'{save_dir}/{setup_id}')
                    adv.save_state_to_path(f'{save_dir}/{setup_id}')
                    head.save_info(f'{save_dir}/{setup_id}')
                    adv.save_info(f'{save_dir}/{setup_id}')

                    if upload_models_to_neptune:
                        head_file_ids = head.get_file_ids()
                        for head_file_id in head_file_ids:
                            run[f'{setup_id}/models/{head_file_id}_state'].upload(f'{save_dir}/{setup_id}/{head_file_id}_state.pt')
                            run[f'{setup_id}/models/{head_file_id}_info'].upload(f'{save_dir}/{setup_id}/{head_file_id}_info.json')
                            run.wait()
                        adv_file_ids = adv.get_file_ids()
                        for adv_file_id in adv_file_ids:
                            run[f'{setup_id}/models/{adv_file_id}_state'].upload(f'{save_dir}/{setup_id}/{adv_file_id}_state.pt')
                            run[f'{setup_id}/models/{adv_file_id}_info'].upload(f'{save_dir}/{setup_id}/{adv_file_id}_info.json')
                            run.wait()

                    if upload_models_to_gcp:
                        raise NotImplementedError('upload_models_to_gcp not implemented')

            except Exception as e:
                run['sys/tags'].add('training failed')
                run["info/state"] = 'Inactive'
                run.stop()
                raise e


        ####################################
        ###### Evaluate the Models ######

        if run_evaluation:
            try:
                print('evaluating models')
                evaluate_compound_model(eval_loader_dct, 
                                        encoder, head, adv, 
                                        run=run, **eval_kwargs)

                # save a history of evaluation results
                # As of April 11th, the history is saved in the eval function
                # run.wait()
                # eval_res = run[f'{setup_id}/eval'].fetch()
                # eval_dict = neptunize_dict_keys(eval_res, f'eval')
                # for key, val in eval_dict.items():
                #     run[f'{setup_id}/history/{key}'].append(val)
                

            except Exception as e:
                run['sys/tags'].add('evaluation failed')
                run["info/state"] = 'Inactive'
                run.stop()
                raise e

    ####################################
    ##### Create an evaluation summary that averages ######
    run_struc = run.get_structure()
    avg_iter_count = 0

    if run_evaluation:
        run.wait()
        for set_name in eval_loader_dct.keys():
            for key in run_struc[setup_id]['eval'][set_name].keys():
                val_array = run[f'{setup_id}/eval/{set_name}/{key}'].fetch_values()
                run[f'{setup_id}/avg/{set_name} {key}'] = round(val_array['value'].mean(),5)
                if avg_iter_count == 0:
                    avg_iter_count = len(val_array['value'])

        # for key in run_struc[setup_id]['eval'][train_name].keys():
        #     val_array = run[f'{setup_id}/eval/{train_name}/{key}'].fetch_values()
        #     run[f'{setup_id}/avg/{train_name} {key}'] = round(val_array['value'].mean(),5)

        run[f'{setup_id}/avg/iter_count'] = avg_iter_count
    
    ####################################
    ###### Generate the Latent Space ######
    print('generating latent space plots')
    try:
        if encoder is None:
            print('No encoder created, attempt to load from save location')
            encoder = get_model(encoder_kind, input_size, **encoder_kwargs)
            encoder_save_loc = f'{save_dir}/{setup_id}_encoder_state_dict.pth'
            if (load_encoder_loc) and (load_model_weights):
                print(f'loading encoder from {load_encoder_loc}')
                os.makedirs(os.path.join(save_dir,load_encoder_loc), exist_ok=True)
                local_path = f'{save_dir}/{load_encoder_loc}/encoder_state_dict.pth'
                if not os.path.exists(local_path):
                    run[f'{load_encoder_loc}/models/encoder_state_dict'].download(local_path)
                encoder_state_dict = torch.load(local_path)
                encoder.load_state_dict(encoder_state_dict)
            elif os.path.exists(encoder_save_loc):
                print(f'loading encoder from {encoder_save_loc}')
                encoder_state_dict = torch.load(encoder_save_loc)
                encoder.load_state_dict(encoder_state_dict)

            elif f'{setup_id}/models/encoder_state_dict' in run.get_structure():
                print(f'loading encoder from neptune')
                run[f'{setup_id}/models/encoder_state_dict'].download(encoder_save_loc)
                encoder_state_dict = torch.load(encoder_save_loc)
                encoder.load_state_dict(encoder_state_dict)
            else:
                raise ValueError('Encoder is None, no saves were found, cannot generate latent space plots')

        Z_embed_savepath = os.path.join(save_dir, f'Z_embed_{eval_name}.csv')

        if save_latent_space or plot_latent_space:
            
            if check_neptune_existance(run,f'{setup_id}/Z_{eval_name}'):
                print(f'Z_{eval_name} already exists in {setup_id} of run {run_id}')
            
            else:
                Z = generate_latent_space(X_data_eval, encoder)
                Z.to_csv(os.path.join(save_dir, f'Z_{eval_name}.csv'))

                Z_pca = generate_pca_embedding(Z,n_components=4)
                Z_pca.to_csv(os.path.join(save_dir, f'Z_pca_{eval_name}.csv'))
                Z_pca.columns = [f'PCA{i+1}' for i in range(Z_pca.shape[1])]

                Z_umap = generate_umap_embedding(Z)
                Z_umap.to_csv(os.path.join(save_dir, f'Z_umap_{eval_name}.csv'))
                Z_umap.columns = [f'UMAP{i+1}' for i in range(Z_umap.shape[1])]

                Z_embed = pd.concat([Z_pca, Z_umap], axis=1)
                # Z_embed = Z_embed.join(y_data_eval)
                Z_embed.to_csv(Z_embed_savepath)
                run[f'{setup_id}/Z_embed_{eval_name}'].upload(Z_embed_savepath)
            run.wait()

            if check_neptune_existance(run,f'{setup_id}/Z_{train_name}'):
                print(f'Z_{train_name} already exists in {setup_id} of run {run_id}')
            
            else:
                Z_embed_train_savepath = os.path.join(save_dir, f'Z_embed_{train_name}.csv')
                if (not os.path.exists(Z_embed_train_savepath)):
                    if X_data_train is None:
                        X_data_train = pd.read_csv(f'{data_dir}/{X_filename_prefix}_{train_name}.csv', index_col=0)
                        y_data_train = pd.read_csv(f'{data_dir}/{y_filename_prefix}_{train_name}.csv', index_col=0)

                    Z = generate_latent_space(X_data_train, encoder)
                    Z.to_csv(os.path.join(save_dir, f'Z_{train_name}.csv'))

                    Z_pca = generate_pca_embedding(Z,n_components=4)
                    Z_pca.to_csv(os.path.join(save_dir, f'Z_pca_{train_name}.csv'))
                    Z_pca.columns = [f'PCA{i+1}' for i in range(Z_pca.shape[1])]

                    Z_umap = generate_umap_embedding(Z)
                    Z_umap.to_csv(os.path.join(save_dir, f'Z_umap_{train_name}.csv'))
                    Z_umap.columns = [f'UMAP{i+1}' for i in range(Z_umap.shape[1])]

                    Z_embed = pd.concat([Z_pca, Z_umap], axis=1)
                    # Z_embed = Z_embed.join(y_data_train)
                    Z_embed.to_csv(Z_embed_train_savepath)
                    run[f'{setup_id}/Z_embed_{train_name}'].upload(Z_embed_train_savepath)

            run.wait()

        plot_latent_space_cols = kwargs.get('plot_latent_space_cols', y_head_cols+y_adv_cols)
        yes_plot_pca = kwargs.get('yes_plot_pca', False)
        print('plot_latent_space:', plot_latent_space)
        print('plot_latent_space_cols:', plot_latent_space_cols)
        if plot_latent_space:
            if os.path.exists(Z_embed_savepath):
                Z_embed = pd.read_csv(Z_embed_savepath, index_col=0)
            else:
                # check if the Z_embed file is in neptune
                if not check_neptune_existance(run,f'{setup_id}/Z_embed_{eval_name}'):
                    raise ValueError(f'No Z_embed_{eval_name} file found in run {run_id}')

                # download the Z_embed file from neptune
                run[f'{setup_id}/Z_embed_{eval_name}'].download(Z_embed_savepath)
                Z_embed = pd.read_csv(Z_embed_savepath, index_col=0)

            missing_cols = [col for col in y_data_eval.columns if col not in Z_embed.columns]
            if len(missing_cols) > 0:
                print(f'Adding metadata columns to Z_embed: {missing_cols}')
                Z_embed = Z_embed.join(y_data_eval[missing_cols])
                # Z_embed.to_csv(Z_embed_savepath)
                # run[f'{setup_id}/Z_embed_{eval_name}'].upload(Z_embed_savepath)



            if (plot_latent_space=='seaborn') or (plot_latent_space=='both') or (plot_latent_space=='sns'):

                for hue_col in plot_latent_space_cols:
                    if hue_col not in Z_embed.columns:
                        print(f'{hue_col} not in Z_embed columns')
                        continue


                    # palette = get_color_map(Z_embed[hue_col].nunique())
                    # Get the counts for each instance of the hue column, and the corresponding colormap
                    Z_count_sum = (~Z_embed[hue_col].isnull()).sum()
                    print(f'Number of samples in {eval_name}: {Z_count_sum}')
                    if Z_count_sum < 10:
                        print('too few to plot')
                        continue
                    if Z_embed[hue_col].nunique() > 30:
                        # if more than 30 unique values, then assume its continuous
                        palette = 'flare'
                        Z_counts = None
                    else:
                        # if fewer than 30 unique values, then assume its categorical
                        # palette = get_color_map(Z_embed[hue_col].nunique())
                        palette = assign_color_map(Z_embed[hue_col].unique())
                        Z_counts = Z_embed[hue_col].value_counts()

                    plot_title = f'{setup_id} Latent Space of {eval_name} (N={Z_count_sum})'
                    # choose the marker size based on the number of nonnan values
                    # marker_sz = 10/(1+np.log(Z_count_sum))
                    marker_sz = 100/np.sqrt(Z_count_sum)

                    ## PCA ##
                    if yes_plot_pca:
                        fig = sns.scatterplot(data=Z_embed, x='PCA1', y='PCA2', hue=hue_col, palette=palette,s=marker_sz)
                        # place the legend outside the plot
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                        
                        # edit the legend to include the number of samples in each cohort
                        handles, labels = fig.get_legend_handles_labels()
                        

                        # Add the counts to the legend if hue_col is categorical
                        if Z_counts is not None:
                            # new_labels = [f'{label} ({Z_embed[Z_embed[hue_col]==label].shape[0]})' for label in labels]
                            new_labels = []
                            for label in labels:
                                # new_labels.append(f'{label} ({Z_counts.loc[eval(label)]})')
                                try:
                                    new_labels.append(f'{label} ({Z_counts.loc[label]})')
                                except KeyError:
                                    new_labels.append(f'{label} ({Z_counts.loc[eval(label)]})')
                        else:
                            new_labels = labels


                        # make the size of the markers in the handles larger
                        for handle in handles:
                            # print(dir(handle))
                            handle.set_markersize(10)
                            # handle._sizes = [100]
                        
                        plt.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue_col)
                        plt.title(plot_title)
                        plt.savefig(os.path.join(save_dir, f'Z_pca_{hue_col}_{eval_name}.png'), bbox_inches='tight')
                        run[f'{setup_id}/sns_Z_pca_{hue_col}_{eval_name}'].upload(os.path.join(save_dir, f'Z_pca_{hue_col}_{eval_name}.png'))
                        plt.close()

                    ## UMAP ##
                    # remove 'nan' from palette
                    if 'nan' in palette:
                        palette.pop('nan')
                    print(f"palette is {palette}")
                    fig = sns.scatterplot(data=Z_embed, x='UMAP1', y='UMAP2', hue=hue_col, palette=palette,s=marker_sz)
                    # place the legend outside the plot
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                    # edit the legend to include the number of samples in each cohort
                    handles, labels = fig.get_legend_handles_labels()

                    # Add the counts to the legend if hue_col is categorical
                    if Z_counts is not None:
                        # new_labels = [f'{label} ({Z_embed[Z_embed[hue_col]==label].shape[0]})' for label in labels]
                        new_labels = []
                        for label in labels:
                            # new_labels.append(f'{label} ({Z_counts.loc[eval(label)]})')
                            try:
                                new_labels.append(f'{label} ({Z_counts.loc[label]})')
                            except KeyError:
                                new_labels.append(f'{label} ({Z_counts.loc[eval(label)]})')
                    else:
                        new_labels = labels

                    # make the size of the markers in the handles larger
                    for handle in handles:
                        # print(dir(handle))
                        handle.set_markersize(10)
                        # handle._sizes = [100]
                    
                    plt.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue_col)

                    plt.title(plot_title)
                    plt.savefig(os.path.join(save_dir, f'Z_umap_{hue_col}_{eval_name}.png'), bbox_inches='tight', dpi=300)
                    run[f'{setup_id}/sns_Z_umap_{hue_col}_{eval_name}'].upload(os.path.join(save_dir, f'Z_umap_{hue_col}_{eval_name}.png'))
                    plt.close()

            if (plot_latent_space=='plotly') or (plot_latent_space=='both') or (plot_latent_space=='px'):
                for hue_col in plot_latent_space_cols:
                    marker_sz = 5
                    if yes_plot_pca:
                        plotly_fig = px.scatter(Z_embed, x='PCA1', y='PCA2', color=hue_col, title=f'PCA {hue_col}')
                        plotly_fig.update_traces(marker=dict(size=2*marker_sz))
                        run[f'{setup_id}/px_Z_pca_{hue_col}_{eval_name}'].upload(plotly_fig)
                        plt.close()

                    plotly_fig = px.scatter(Z_embed, x='UMAP1', y='UMAP2', color=hue_col)
                    plotly_fig.update_traces(marker=dict(size=2*marker_sz))
                    run[f'{setup_id}/px_Z_umap_{hue_col}_{eval_name}'].upload(plotly_fig)
                    plt.close()

            run.wait()

    except Exception as e:
        run['sys/tags'].add('plotting failed')
        run["info/state"] = 'Inactive'
        run.stop()
        raise e

    run['sys/failed'] = False
    if ret_run_id:
        run["info/state"] = 'Inactive'
        run.stop()
        return run_id
    else:
        print('Returning Neptune Run, it has NOT been stopped')
        return run

