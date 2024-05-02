# setup a neptune run


import torch
import pandas as pd
import numpy as np
import os
import json
from models import get_model, Binary_Head, Dummy_Head, MultiClass_Head, MultiHead, Regression_Head, Cox_Head

# from train3 import CompoundDataset, train_compound_model, get_end_state_eval_funcs, evaluate_compound_model, create_dataloaders, create_dataloaders_old
from train4 import CompoundDataset, train_compound_model, get_end_state_eval_funcs, evaluate_compound_model, create_dataloaders, create_dataloaders_old
print('WARNING: using Train4 not Train3')


import neptune
from neptune.utils import stringify_unsupported
from utils_neptune import check_neptune_existance, start_neptune_run, convert_neptune_kwargs, neptunize_dict_keys, get_latest_dataset
from neptune_pytorch import NeptuneLogger
from viz import generate_latent_space, generate_umap_embedding, generate_pca_embedding
from misc import assign_color_map, get_color_map
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from neptune.exceptions import NeptuneException


# set up neptune
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='



def setup_neptune_run(data_dir,setup_id,with_run_id=None,run=None,
                      neptune_mode='async',
                      yes_logging = False,
                      save_kwargs_to_neptune=False,
                      restart_run=False,
                      tags=['v3.3'],**kwargs):
    print(setup_id)
    if run is None:
        run, is_run_new = start_neptune_run(with_run_id=with_run_id,
                                            neptune_mode=neptune_mode,
                                            tags=tags,
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
        load_head_loc = kwargs.get('load_head_loc', load_model_loc)
        load_adv_loc = kwargs.get('load_adv_loc', load_model_loc)
        y_head_cols = kwargs.get('y_head_cols', [])
        y_adv_cols = kwargs.get('y_adv_cols', [])
        
        if load_model_from_run_id:
            pretrained_run = neptune.init_run(project='revivemed/RCC',
                                            api_token=NEPTUNE_API_TOKEN,
                                            with_id=load_model_from_run_id,
                                            capture_stdout=False,
                                            capture_stderr=False,
                                            capture_hardware_metrics=False,
                                            mode='read-only')
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
            encoder_kwargs.update(kwargs.get('encoder_kwargs', {}))
            kwargs['encoder_kwargs'] = encoder_kwargs
            print('encoder_kwargs:', kwargs['encoder_kwargs'])

        if load_head_loc:
            print('loading pretrained heads, overwriting head_kwargs_list')
            load_kwargs = pretrained_run[f'{load_head_loc}/original_kwargs'].fetch()
            kwargs['head_kwargs_dict'] = load_kwargs.get('head_kwargs_dict', {})
            kwargs['head_kwargs_list'] = eval(load_kwargs.get('head_kwargs_list', '[]'))
            # assert len(kwargs['head_kwargs_list']) <= len(y_head_cols)

        if load_adv_loc:
            print('loading pretrained advs, overwriting adv_kwargs_list')
            load_kwargs = pretrained_run[f'{load_adv_loc}/original_kwargs'].fetch()
            kwargs['adv_kwargs_dict'] = load_kwargs.get('adv_kwargs_dict', {})
            kwargs['adv_kwargs_list'] = eval(load_kwargs.get('adv_kwargs_list', '[]'))
            # assert len(kwargs['adv_kwargs_list']) <= len(y_adv_cols)

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
    save_latent_space = kwargs.get('save_latent_space', True)
    
    try:
        print('loading data')
        if os.path.exists(f'{data_dir}/hash.txt'):
            dataset_hash = open(f'{data_dir}/hash.txt','r').read()
            run[f'{setup_id}/datasets/hash'] = dataset_hash

        X_filename = kwargs.get('X_filename', 'X_pretrain')
        y_filename = kwargs.get('y_filename', 'y_pretrain')
        # nan_filename = kwargs.get('nan_filename', 'nans')
        train_name = kwargs.get('train_name', 'train')
        eval_name = kwargs.get('eval_name', 'val')
        X_size = None
        train_dataset = None
        eval_dataset = None

        if run_training or run_evaluation:
            X_data_train = pd.read_csv(f'{data_dir}/{X_filename}_{train_name}.csv', index_col=0)
            y_data_train = pd.read_csv(f'{data_dir}/{y_filename}_{train_name}.csv', index_col=0)
            X_size = X_data_train.shape[1]
            run[f'{setup_id}/datasets/X_train'].track_files(f'{data_dir}/{X_filename}_{train_name}.csv')
            run[f'{setup_id}/datasets/y_train'].track_files(f'{data_dir}/{y_filename}_{train_name}.csv')
        # nan_data = pd.read_csv(f'{data_dir}/{nan_filename}_{train_name}.csv', index_col=0)
        else:
            X_data_train = None

        if run_evaluation or save_latent_space:
            X_data_eval = pd.read_csv(f'{data_dir}/{X_filename}_{eval_name}.csv', index_col=0)
            y_data_eval = pd.read_csv(f'{data_dir}/{y_filename}_{eval_name}.csv', index_col=0)
            run[f'{setup_id}/datasets/X_eval'].track_files(f'{data_dir}/{X_filename}_{eval_name}.csv')
            run[f'{setup_id}/datasets/y_eval'].track_files(f'{data_dir}/{y_filename}_{eval_name}.csv')
            X_size = X_data_eval.shape[1]
        # nan_data = pd.read_csv(f'{data_dir}/{nan_filename}_{eval_name}.csv', index_col=0)


        ####################################
        ##### Create the DataLoaders ######
        batch_size = kwargs.get('batch_size', 32)
        holdout_frac = kwargs.get('holdout_frac', 0)
        train_kwargs = kwargs.get('train_kwargs', {})
        early_stopping_patience = train_kwargs.get('early_stopping_patience', 0)
        scheduler_kind = train_kwargs.get('scheduler_kind', None)

        if (holdout_frac > 0) and (early_stopping_patience < 1) and (scheduler_kind is None):
            # raise ValueError('holdout_frac > 0 and early_stopping_patience < 1 is not recommended')
            print('holdout_frac > 0 and early_stopping_patience < 1 is not recommended, set hold out frac to 0')
            print('UNLESS you are using a scheduler, in which case the holdout_frac is used for the scheduler')
            holdout_frac = 0

        if run_training or run_evaluation:
            train_dataset = CompoundDataset(X_data_train,y_data_train[y_head_cols], y_data_train[y_adv_cols])
            eval_dataset = CompoundDataset(X_data_eval,y_data_eval[y_head_cols], y_data_eval[y_adv_cols])

            # stratify on the adversarial column (stratify=2)
            # this is probably not the most memory effecient method, would be better to do stratification before creating the dataset
            # train_loader_dct = create_dataloaders(train_dataset, batch_size, holdout_frac, set_name=train_name, stratify=2)
            train_loader_dct = create_dataloaders_old(train_dataset, batch_size, holdout_frac, set_name=train_name)
            eval_loader_dct = create_dataloaders(eval_dataset, batch_size, set_name = eval_name)
            eval_loader_dct.update(train_loader_dct)

            ########################################
            ##### Custom Evaluation for OS predictions
            #Eval on the EVER OS data when 
            #the model is trained on the NEVER OS data
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
                            
                            train_dataset2 = CompoundDataset(X_data_train,y_data_train[y_adv_cols], y_data_train[y_adv_cols])
                            eval_dataset2 = CompoundDataset(X_data_eval,y_data_eval[y_adv_cols], y_data_eval[y_adv_cols])
                            train_loader_dct2 = create_dataloaders_old(train_dataset2, batch_size, set_name=train_name+'_'+adv_name)
                            eval_loader_dct2 = create_dataloaders(eval_dataset2, batch_size, set_name = eval_name+'_'+adv_name)
                            eval_loader_dct.update(train_loader_dct2)
                            eval_loader_dct.update(eval_loader_dct2)

            ########################################

    except Exception as e:
        run['sys/tags'].add('data-load failed')
        run["info/state"] = 'Inactive'
        run.stop()
        raise e


    ########################################################################
    ###### Repeated Training and Evaluation

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
    if 'hidden_size_mult' in encoder_kwargs:
        encoder_kwargs['hidden_size'] = int(encoder_kwargs['hidden_size_mult']*latent_size)
        # remove the hidden_size_mult key
        encoder_kwargs.pop('hidden_size_mult')
    
    if input_size is None:
        try:
            input_size = run['input_size'].fetch()
        except NeptuneException:
            input_size = X_size
            run['input_size'] = input_size
    if X_size is not None:
        assert input_size == X_size
    # if input_size is None:


    if encoder_kind == 'TGEM_Encoder':
        latent_size = input_size

    ############################################
    
    
    
    run_random_init = kwargs.get('run_random_init', False) # this should be redundant with load_model_weights
    upload_models_to_neptune = kwargs.get('upload_models_to_neptune', True)
    upload_models_to_gcp = kwargs.get('upload_models_to_gcp', False)
    eval_kwargs = kwargs.get('eval_kwargs', {})

    eval_prefix = f'{setup_id}/eval'
    eval_kwargs['prefix'] = eval_prefix

    try:
        eval_recon_loss_history = run[f'{eval_prefix}/{eval_name}/reconstruction_loss'].fetch_values()
        num_existing_repeats = len(eval_recon_loss_history)
        print(f'Already computed {num_existing_repeats} train/eval repeats')
    except NeptuneException:
        num_existing_repeats = 0

    if num_existing_repeats > num_repeats:
        print('Already computed over {} train/eval repeats'.format(num_repeats))

    num_repeats = num_repeats - num_existing_repeats
    if num_repeats < 0:
        num_repeats = 0

    for ii_repeat in range(num_repeats):
        
        if num_repeats > 1:
            print(f'train/eval repeat: {ii_repeat}')
            yes_save_model = False
            if (not run_evaluation):
                print('why are you running training multiple times without any evaluation?')

        if ii_repeat == num_repeats-1:
            yes_save_model = True


        ####################################
        ###### Create the Encoder Models ######
        try:
            if run_training or run_evaluation or save_latent_space:
                print('creating models')
                
                encoder = get_model(encoder_kind, input_size, **encoder_kwargs)


                if (load_encoder_loc) and (load_model_weights):
                    os.makedirs(os.path.join(pretrain_save_dir,load_encoder_loc), exist_ok=True)
                    local_path = f'{pretrain_save_dir}/{load_encoder_loc}/encoder_state_dict.pth'
                    if not os.path.exists(local_path):
                        run[f'{load_encoder_loc}/models/encoder_state_dict'].download(local_path)
                    encoder_state_dict = torch.load(local_path)
                    encoder.load_state_dict(encoder_state_dict)
                else:
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
                if train_dataset is not None:
                    #TODO: loadd the class weights from the model json
                    head.update_class_weights(train_dataset.y_head)


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

                if train_dataset is not None:
                    #TODO load the class weights from the model info json
                    adv.update_class_weights(train_dataset.y_adv)

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


                train_kwargs = kwargs.get('train_kwargs', {})
                train_kwargs['prefix'] = f'{setup_id}/train2'
                encoder, head, adv = train_compound_model(train_loader_dct, 
                                                        encoder, head, adv, 
                                                        run=run, **train_kwargs)
                if encoder is None:
                    raise ValueError('Encoder is None after training')


                if yes_save_model:
                    # log the models
                    # run[f'{setup_id}/encoder'] = npt_logger.log_model('encoder')
                    # run[f'{setup_id}/head'] = npt_logger.log_model(head)
                    # run[f'{setup_id}/adv'] = npt_logger.log_model(adv)

                    # alternative way to log models
                    torch.save(encoder.state_dict(), f'{save_dir}/{setup_id}_encoder_state_dict.pth')
                    encoder.save_info(f'{save_dir}/{setup_id}')
                    # torch.save(head.state_dict(), f'{save_dir}/{setup_id}_head_state_dict.pth')
                    # torch.save(adv.state_dict(), f'{save_dir}/{setup_id}_adv_state_dict.pth')
                    if upload_models_to_neptune:
                        run[f'{setup_id}/models/encoder_state_dict'].upload(f'{save_dir}/{setup_id}_encoder_state_dict.pth')
                        run[f'{setup_id}/models/encoder_info'].upload(f'{save_dir}/{setup_id}_encoder_info.json')
                    # run[f'{setup_id}/models/head_state_dict'].upload(f'{save_dir}/{setup_id}_head_state_dict.pth')
                    # run[f'{setup_id}/models/adv_state_dict'].upload(f'{save_dir}/{setup_id}_adv_state_dict.pth')
                        run.wait()

                    if upload_models_to_gcp:
                        raise NotImplementedError('upload_models_to_gcp not implemented')

                    os.makedirs(os.path.join(save_dir,setup_id), exist_ok=True)
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

        save_latent_space = kwargs.get('save_latent_space', True)
        Z_embed_savepath = os.path.join(save_dir, f'Z_embed_{eval_name}.csv')

        if save_latent_space:
            
            if check_neptune_existance(run,f'{setup_id}/Z_{eval_name}'):
                print(f'Z_{eval_name} already exists in {setup_id} of run {run_id}')
            
            else:
                Z = generate_latent_space(X_data_eval, encoder)
                Z.to_csv(os.path.join(save_dir, f'Z_{eval_name}.csv'))

                Z_pca = generate_pca_embedding(Z)
                Z_pca.to_csv(os.path.join(save_dir, f'Z_pca_{eval_name}.csv'))
                Z_pca.columns = [f'PCA{i+1}' for i in range(Z_pca.shape[1])]

                Z_umap = generate_umap_embedding(Z)
                Z_umap.to_csv(os.path.join(save_dir, f'Z_umap_{eval_name}.csv'))
                Z_umap.columns = [f'UMAP{i+1}' for i in range(Z_umap.shape[1])]

                Z_embed = pd.concat([Z_pca, Z_umap], axis=1)
                Z_embed = Z_embed.join(y_data_eval)
                Z_embed.to_csv(Z_embed_savepath)
                run[f'{setup_id}/Z_embed_{eval_name}'].upload(Z_embed_savepath)
            run.wait()

            if check_neptune_existance(run,f'{setup_id}/Z_{train_name}'):
                print(f'Z_{train_name} already exists in {setup_id} of run {run_id}')
            
            else:
                Z_embed_train_savepath = os.path.join(save_dir, f'Z_embed_{train_name}.csv')
                if (not os.path.exists(Z_embed_train_savepath)):
                    if X_data_train is None:
                        X_data_train = pd.read_csv(f'{data_dir}/{X_filename}_{train_name}.csv', index_col=0)
                        y_data_train = pd.read_csv(f'{data_dir}/{y_filename}_{train_name}.csv', index_col=0)

                    Z = generate_latent_space(X_data_train, encoder)
                    Z.to_csv(os.path.join(save_dir, f'Z_{train_name}.csv'))

                    Z_pca = generate_pca_embedding(Z)
                    Z_pca.to_csv(os.path.join(save_dir, f'Z_pca_{train_name}.csv'))
                    Z_pca.columns = [f'PCA{i+1}' for i in range(Z_pca.shape[1])]

                    Z_umap = generate_umap_embedding(Z)
                    Z_umap.to_csv(os.path.join(save_dir, f'Z_umap_{train_name}.csv'))
                    Z_umap.columns = [f'UMAP{i+1}' for i in range(Z_umap.shape[1])]

                    Z_embed = pd.concat([Z_pca, Z_umap], axis=1)
                    Z_embed = Z_embed.join(y_data_train)
                    Z_embed.to_csv(Z_embed_train_savepath)
                    run[f'{setup_id}/Z_embed_{train_name}'].upload(Z_embed_train_savepath)

            run.wait()


        plot_latent_space = kwargs.get('plot_latent_space', '')
        plot_latent_space_cols = kwargs.get('plot_latent_space_cols', [y_head_cols, y_adv_cols])
        yes_plot_pca = kwargs.get('yes_plot_pca', False)
        print('plot_latent_space:', plot_latent_space)
        print('plot_latent_space_cols:', plot_latent_space_cols)
        if plot_latent_space:
            if os.path.exists(Z_embed_savepath):
                Z_embed = pd.read_csv(Z_embed_savepath, index_col=0)
            else:
                # check if the Z_embed file is in neptune
                if check_neptune_existance(run,f'{setup_id}/Z_embed_{eval_name}'):
                    raise ValueError(f'No Z_embed_{eval_name} file found in run {run_id}')

                # download the Z_embed file from neptune
                run[f'{setup_id}/Z_embed_{eval_name}'].download(Z_embed_savepath)
                Z_embed = pd.read_csv(Z_embed_savepath, index_col=0)

            missing_cols = [col for col in y_data_eval.columns if col not in Z_embed.columns]
            if len(missing_cols) > 0:
                print(f'Adding missing columns to Z_embed: {missing_cols}')
                Z_embed = Z_embed.join(y_data_eval[missing_cols])
                Z_embed.to_csv(Z_embed_savepath)
                run[f'{setup_id}/Z_embed_{eval_name}'].upload(Z_embed_savepath)



            if (plot_latent_space=='seaborn') or (plot_latent_space=='both') or (plot_latent_space=='sns'):

                for hue_col in plot_latent_space_cols:
                    if hue_col not in Z_embed.columns:
                        print(f'{hue_col} not in Z_embed columns')
                        continue


                    # palette = get_color_map(Z_embed[hue_col].nunique())
                    # Get the counts for each instance of the hue column, and the corresponding colormap
                    Z_count_sum = (~Z_embed[hue_col].isnull()).sum()
                    print(f'Number of samples in {eval_name}: {Z_count_sum}')
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


# Run an example

## Main

def main():

    from sklearn.linear_model import LogisticRegression


    data_dir = '/DATA'
    data_dir = get_latest_dataset(data_dir)

    kwargs = {
        'X_filename': 'X_finetune',
        'y_filename': 'y_finetune',
        'y_head_cols' : ['MSKCC BINARY','Benefit ORDINAL'],
        'y_adv_cols' : ['OS_Event'],
        # 'load_model_loc': 'pretrain',

        'encoder_kind': 'AE',
        'encoder_kwargs': {
            'latent_size': 8,
            'num_hidden_layers': 2,
            'hidden_size': 16,
            'activation': 'leakyrelu',
            'dropout_rate': 0.2,
            'use_batch_norm': False,
            # 'use_skip_connections': False,
            # 'use_residuals': False,
            # 'use_layer_norm': False,
            # 'use_weight_norm': False,
            # 'use_spectral_norm': False,
        },
        'head_kwargs_list': [
            {
                'kind': 'Binary',
                'name': 'MSKCC',
                'y_idx': 0,
                'weight': 1,
                'hidden_size': 4,
                'num_hidden_layers': 1,
                'dropout_rate': 0,
                'activation': 'leakyrelu',
                'use_batch_norm': False,
                'num_classes': 2,
            },
            {
                'kind': 'MultiClass',
                'name': 'Benefit',
                'y_idx': 1,
                'weight': 1,
                'hidden_size': 4,
                'num_hidden_layers': 1,
                'dropout_rate': 0,
                'activation': 'leakyrelu',
                'use_batch_norm': False,
                'num_classes': 3,
            },
        ],
        'adv_kwargs_list': [
            {
                'kind': 'Binary',
                'name': 'Adv Event',
                'y_idx': 0,
                'weight': 1, 
                'hidden_size': 4,
                'num_hidden_layers': 1,
                'dropout_rate': 0,
                'activation': 'leakyrelu',
                'use_batch_norm': False,
                'num_classes': 2,
            },
        ],
        'train_kwargs': {
            'num_epochs': 10,
            'learning_rate': 0.001,
            'optimizer_kind': 'Adam',
            # 'scheduler_kind': 'ReduceLROnPlateau',
            # 'scheduler_kwargs': {
            #     'factor': 0.1,
            #     'patience': 5,
            #     'threshold': 0.0001,
            #     'cooldown': 0,
            #     'min_lr': 0,
            # },
            'adversary_weight': 1,
            'head_weight': 1,
            'encoder_weight': 1,
        },
        'eval_kwargs': {
            'sklearn_models':  {
                                'Logistic Regression': LogisticRegression(max_iter=10000, C=1.0, solver='lbfgs')
                            },
        },
        'run_training': True,
        'run_evaluation': True,
        'save_latent_space': True,
        'plot_latent_space': 'both',
        'plot_latent_space_cols': ['Study ID','Cohort Label'],
    }

    run_id = setup_neptune_run(data_dir,setup_id='pretrain',**kwargs)


if __name__ == '__main__':
    main()