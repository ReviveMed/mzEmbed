# %%
import pandas as pd
import numpy as np
import neptune
import torch
from neptune.utils import stringify_unsupported
from misc import save_json, load_json, get_clean_batch_sz
from utils_neptune import get_latest_dataset,get_run_id_list, check_neptune_existance, check_if_path_in_struc, convert_neptune_kwargs
from models import initialize_model, get_encoder, get_head, MultiHead, create_model_wrapper, create_pytorch_model_from_info, CompoundModel
from train4 import train_compound_model, create_dataloaders_old, CompoundDataset, convert_y_data_by_codes
import os
from prep_run import create_selected_data, convert_kwargs_for_optuna, create_full_metadata, get_task_head_kwargs, make_kwargs_set
import shutil
import optuna
from collections import defaultdict
import re
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prep_study2 import get_default_study_kwargs
from prep_run import assign_sets, get_selection_df

from viz import generate_latent_space, generate_pca_embedding, generate_umap_embedding
from misc import assign_color_map
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

PROJECT_ID = 'revivemed/Survival-RCC'
#import torch_cpu_loader
# %%



def setup_wrapper(**kwargs):

    project_id = kwargs.get('project_id',PROJECT_ID)
    api_token = kwargs.get('api_token',NEPTUNE_API_TOKEN)
    setup_id = kwargs.get('setup_id','example')
    # prefix = kwargs.get('prefix','training_run')
    subdir_col = kwargs.get('subdir_col','Study ID')
    selections_df = kwargs.get('selections_df',None)
    output_dir = kwargs.get('output_dir',None)
    yes_plot_latent_space = kwargs.get('yes_plot_latent_space',False)
    fit_subset_col = kwargs.get('fit_subset_col',None)
    eval_subset_col_list = kwargs.get('eval_subset_col_list',[])
    eval_params_list = kwargs.get('eval_params_list',None)
    tags = kwargs.get('tags',[])
    
    resume_with_id = kwargs.get('resume_with_id',None) # resume run with id
    download_models_from_gcp = kwargs.get('download_models_from_gcp',False)
    
    pretrained_project_id = kwargs.get('pretrained_project_id',project_id)
    pretrained_model_id = kwargs.get('pretrained_model_id',None)
    pretrained_is_registered = kwargs.get('pretrained_is_registered',False) #is the encoder coming from a Neptune Model or Neptune Run object?
    pretrained_loc = kwargs.get('pretrained_loc',None) 
    use_pretrained_head= kwargs.get('use_pretrained_head',False)
    use_pretrained_adv= kwargs.get('use_pretrained_adv',False)
    
    head_name_list = kwargs.get('head_name_list',[])
    adv_name_list = kwargs.get('adv_name_list',[])
    num_iterations = kwargs.get('num_iterations',1)

    overwrite_default_params = kwargs.get('overwrite_existing_params',False)
    overwrite_params_fit_kwargs = kwargs.get('overwrite_params_fit_kwargs',{})
    overwrite_params_task_kwargs = kwargs.get('overwrite_params_task_kwargs',{})
    overwrite_params_other_kwargs = kwargs.get('overwrite_params_other_kwargs',{})
    
    if fit_subset_col is None:
        raise ValueError('fit_subset_col must be provided')

    optuna_study_info_dict = kwargs.get('optuna_study_info_dict',None)
    optuna_trial = kwargs.get('optuna_trial',None)
    if optuna_trial: use_optuna = True
    else: use_optuna = False

    assign_task_head_func = kwargs.get('assign_task_head_func',assign_task_head_default)

    if use_optuna:
        get_kwargs_func = kwargs.get('get_kwargs_func',get_default_study_kwargs)
    else:
        get_kwargs_func = kwargs.get('get_kwargs_func',make_kwargs_set)


    #################################

    input_data_dir = os.path.expanduser("~/INPUT_DATA")
    os.makedirs(input_data_dir, exist_ok=True)
    input_data_dir = get_latest_dataset(data_dir=input_data_dir,
                                        api_token=NEPTUNE_API_TOKEN,
                                        project=project_id)

    if os.path.exists(f'{input_data_dir}/metadata.csv'):
        all_metadata = pd.read_csv(f'{input_data_dir}/metadata.csv',index_col=0)
    else:
        all_metadata = create_full_metadata(input_data_dir, cleaning=True, save_file=True)

    if selections_df is None:
        if os.path.exists(f'{input_data_dir}/selection_df.csv'):
            selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv',index_col=0)
        else:
            print('Selections dataframe not found, creating a new one using default assignments')
            if 'Set' not in all_metadata.columns:
                selections_df = assign_sets(all_metadata,return_only_sets=True)
            else:
                selections_df = get_selection_df(all_metadata)



    if output_dir is None:
        output_dir = os.path.expanduser("~/TEMP_MODELS")

    run = neptune.init_run(project=project_id,
                    api_token=api_token,
                    with_id=resume_with_id,
                    tags=tags)

    run_id = run["sys/id"].fetch()
    if resume_with_id:
            # desc_str = run['desc_str'].fetch()
            original_params = run['params'].fetch()
            params = convert_neptune_kwargs(original_params)
            # original_sweep_kwargs = run['sweep_kwargs'].fetch()
            # original_sweep_kwargs = convert_neptune_kwargs(original_sweep_kwargs)
            if overwrite_default_params:
                print('WARNING: Overwriting existing params, are you sure you want to do this? this could lead to messy data')
                params['fit_kwargs'].update(overwrite_params_fit_kwargs)
                params['task_kwargs'].update(overwrite_params_task_kwargs)
                params['other_kwargs'].update(overwrite_params_other_kwargs)
                # params['encoder_desc']['encoder_kwargs'].update(encoder_kwargs)
                # params['encoder_desc']['encoder_load_dir'] = encoder_load_dir
                # params['encoder_desc']['encoder_project_id'] = encoder_project_id
                # params['encoder_desc']['encoder_model_id'] = encoder_model_id
                # params['encoder_desc']['encoder_is_a_run'] = encoder_is_a_run

    else:
        plot_latent_space_cols = []
        y_head_cols = []
        y_adv_cols = []
        head_kwargs_dict = {}
        adv_kwargs_dict = {}

        for head_name in head_name_list:
            head_kind,y_head_col,num_classes,default_weight = assign_task_head_func(head_name,all_metadata)

            head_kwargs_dict[head_name], y_head_cols = get_task_head_kwargs(head_kind=head_kind,
                                                y_head_col=y_head_col,
                                                y_cols=y_head_cols,
                                                head_name=head_name,
                                                num_classes=num_classes,
                                                default_weight=default_weight)
            
        for adv_name in adv_name_list:
            adv_kind,y_adv_col,num_classes,default_weight = assign_task_head_func(adv_name,all_metadata)
            adv_kwargs_dict[adv_name], y_adv_cols = get_task_head_kwargs(head_kind=adv_kind,
                                                y_head_col=y_adv_col,
                                                y_cols=y_adv_cols,
                                                head_name=adv_name,
                                                num_classes=num_classes,
                                                default_weight=default_weight)

        plot_latent_space_cols  = list(set(y_head_cols + y_adv_cols))


        run_kwargs = get_kwargs_func(head_kwargs_dict=head_kwargs_dict,adv_kwargs_dict=adv_kwargs_dict)
        other_kwargs = {}
        other_kwargs['plot_latent_space_cols'] = plot_latent_space_cols

        if use_optuna:
            run_kwargs = convert_kwargs_for_optuna(run_kwargs,optuna_trial)
            other_kwargs['optuna_study_info_dict'] = optuna_study_info_dict
            optuna_trial.set_user_attr('run_id',run_id)
            optuna_trial.set_user_attr('setup_id',setup_id)


        if pretrained_model_id:
            print(f'Attempt to load encoder from Neptune {pretrained_project_id}/{pretrained_model_id}/{pretrained_loc}')
            encoder_kwargs = {}
        else:
            encoder_kwargs = run_kwargs['encoder_kwargs']


        params = {
            'task_kwargs': {
                'y_head_cols': y_head_cols,
                'y_adv_cols': y_adv_cols,
                'head_kwargs_dict': run_kwargs['head_kwargs_dict'],
                'adv_kwargs_dict': run_kwargs['adv_kwargs_dict']
            },
            'encoder_desc': {
                'encoder_kwargs': encoder_kwargs,
                'model_id': pretrained_model_id,
                'project_id': pretrained_project_id,
                'model_is_registered': pretrained_is_registered,
                'model_loc': pretrained_loc,
                'download_models_from_gcp': download_models_from_gcp
            },
            'fit_kwargs': run_kwargs['fit_kwargs'],
            'other_kwargs': other_kwargs
        }

        if overwrite_default_params:
            print('WARNING: Overwriting existing params, are you sure you want to do this? this could lead to messy data')
            params['fit_kwargs'].update(overwrite_params_fit_kwargs)
            params['task_kwargs'].update(overwrite_params_task_kwargs)
            params['other_kwargs'].update(overwrite_params_other_kwargs)
        
        if (use_pretrained_head):
            raise NotImplementedError('use_pretrained_head not implemented in this script')

        if (use_pretrained_adv):
            raise NotImplementedError('use_pretrained_adv not implemented in this script')

        run['dataset'].track_files(input_data_dir)
        run['params'] = stringify_unsupported(params)


    if eval_params_list is None:
        # eval_params_list0 = [x for x in default_eval_params_list if x['y_cols'][0] in params['task_kwargs']['y_head_cols']]
        # eval_params_list1 = [x for x in default_eval_params_list if x['y_head'][0] in head_name_list]
        # eval_params_list = eval_params_list0 + eval_params_list1

        eval_params_list = [x for x in default_eval_params_list if x['y_head'] in head_name_list]

        # eval_params_list = [x for x in default_eval_params_list if x['y_head'][0].replace('-',' ').replace('_',' ') in desc_str_simplified]
        # eval_params_list = default_eval_params_list
    
    eval_params_list.append(default_eval_params_list[0]) # always add the reconstruction evaluation
    eval_params_list.append({})

    print('eval_params_list',eval_params_list)
    run_output_dir = f'{output_dir}/{run_id}'
    os.makedirs(run_output_dir,exist_ok=True)

    
    run, all_metrics = run_multiple_iterations(input_data_dir=input_data_dir,
                                params=params,
                                output_dir=run_output_dir,
                                prefix=setup_id,
                                fit_subset_col=fit_subset_col,
                                subdir_col=subdir_col,
                                selections_df=selections_df,
                                eval_subset_col_list=eval_subset_col_list,
                                eval_params_list=eval_params_list,
                                run=run,
                                yes_plot_latent_space=yes_plot_latent_space,
                                num_iterations=num_iterations)
    run['sys/failed'] = False
    run.stop()
    return run_id, all_metrics



def assign_task_head_default(head_name,all_metadata,default_weight=1.0):
    # raise NotImplementedError('assign_task_head not implemented in this script')

    print('Assignment of task head:',head_name)
    metadata_cols = all_metadata.columns
    standardized_cols = [col.replace(' ','_').replace('-','_').lower() for col in metadata_cols]
    head_name_std = head_name.replace(' ','_').replace('-','_').lower()


    if head_name == 'Both-OS':
        head_kind = 'Cox'
        y_head_col = 'OS'
        num_classes = 2
        default_weight = 1.0
    elif head_name == 'NIVO-OS':
        head_kind = 'Cox'
        y_head_col = 'NIVO OS'
        num_classes = 2
        default_weight = 1.0
    elif head_name == 'EVER-OS':
        head_kind = 'Cox'
        y_head_col = 'EVER OS'
        num_classes = 2
        default_weight = 1.0
    elif head_name == 'Study ID':
        head_kind = 'MultiClass'
        y_head_col = 'Study ID'
        num_classes = all_metadata['Study ID'].nunique() #22
        # print('num_classes:',num_classes)
        default_weight = 1.0
    elif head_name == 'Cohort-Label':
        head_kind = 'MultiClass'
        y_head_col = 'Cohort Label v0'
        num_classes = all_metadata['Cohort Label v0'].nunique() #4
        # print('num_classes:',num_classes)
        default_weight = 1.0
    elif head_name == 'is-Pediatric':
        head_kind = 'Binary'
        y_head_col = 'is Pediatric'
        num_classes = 2
        default_weight = 1.0
    elif head_name == 'Age':
        head_kind = 'Regression'
        y_head_col = 'Age'
        num_classes = 1
        default_weight = 1.0
    elif head_name == 'Sex':
        head_kind = 'Binary'
        y_head_col = 'Sex'
        num_classes = 2
        default_weight = 1.0
    elif head_name == 'BMI':
        head_kind = 'Regression'
        y_head_col = 'BMI'
        num_classes = 1
        default_weight = 1.0
    elif head_name == 'IMDC':
        head_kind = 'Binary'
        y_head_col = 'IMDC BINARY'
        num_classes = 2
        default_weight = 1.0
    else:
        print('Head name is not one of the predefined options, searching for a matching column in the metadata to make a best guess')
        possible_cols = []
        for col in standardized_cols:
            if head_name_std in col:
                possible_cols.append(col)

        if len(possible_cols) == 0:
            raise ValueError(f'No matching columns found for {head_name}')
        
        if len(possible_cols) > 1:
            raise ValueError(f'Multiple matching columns found for {head_name}')
        
        y_head_col = metadata_cols[standardized_cols.index(possible_cols[0])]
        y_uniq_vals = all_metadata[y_head_col].unique()
        y_dtype = all_metadata[y_head_col].dtype

        if y_dtype == 'object':
            if len(y_uniq_vals) == 2:
                head_kind = 'Binary'
                num_classes = 2
                default_weight = 1.0
            elif len(y_uniq_vals) > 2:
                head_kind = 'MultiClass'
                num_classes = len(y_uniq_vals)
                default_weight = 1.0
            else:
                raise ValueError(f'No matching head found for {head_name}')
        elif y_dtype == 'float':
            if len(y_uniq_vals) > 25:
                head_kind = 'Regression'
                num_classes = 1
                default_weight = 1.0
            elif len(y_uniq_vals) > 2:
                head_kind = 'MultiClass'
                num_classes = len(y_uniq_vals)
                default_weight = 1.0
            elif len(y_uniq_vals) == 2:
                head_kind = 'Binary'
                num_classes = 2
                default_weight = 1.0
            else:
                raise ValueError(f'No matching head found for {head_name}')

    print('Assigned head kind:',head_kind)
    print('Assigned y_head_col:',y_head_col)
    print('Assigned num_classes:',num_classes)
    print('Assigned default_weight:',default_weight)    
    return head_kind, y_head_col, num_classes, default_weight
            





def run_multiple_iterations(input_data_dir,
                                params,
                                output_dir,
                                prefix,
                                fit_subset_col,
                                subdir_col,
                                selections_df,
                                eval_subset_col_list,
                                eval_params_list,
                                run,
                                yes_plot_latent_space,
                                num_iterations=1):

    record_metrics = defaultdict(list)
    num_train_success = 0



    for iter in range(num_iterations):
        try:

            metrics = run_model_wrapper(data_dir=input_data_dir,
                            params=params,
                            output_dir=output_dir,
                            prefix=prefix,
                            fit_subset_col=fit_subset_col,
                            subdir_col=subdir_col,
                            selections_df=selections_df,
                            eval_params_list=eval_params_list,
                            eval_subset_col_list=eval_subset_col_list,
                            run_dict=run,
                            yes_plot_latent_space=(yes_plot_latent_space and iter==num_iterations-1))

            if metrics is None:
                continue
            
            for key,val in metrics.items():
                if isinstance(val,dict):
                    for k,v in val.items():
                        if isinstance(v,dict):
                            for kk,vv in v.items():
                                record_metrics[key+'_'+k+'_'+kk].append(vv)
                        else:
                            record_metrics[key+'_'+k].append(v)
                else:
                    record_metrics[key].append(val)
            num_train_success += 1
        except ValueError as e:
            print(e)
    
    
    # run[f'all_training_{prefix_name}/metrics/{key}'] = []
    for key,val in record_metrics.items():
        # run[f'all_training_{prefix_name}/metrics/{key}'].extend(val)
        run[f'avg_{prefix}/metrics/{key}'] = np.mean(val)
        if len(val) > 1:
            run[f'avg_{prefix}/metrics/std_{key}'] = np.std(val)
    run[f'avg_{prefix}/num_success'] = num_train_success



    all_metrics = {f'{prefix}__'+k:v for k,v in record_metrics.items()}

    return run, all_metrics


def run_model_wrapper(data_dir, params, output_dir=None, prefix='training_run',
                      fit_subset_col='train',subdir_col='subdir', selections_df=None,
                      eval_subset_col_list=['val'], eval_params_list=None, 
                      run_dict=None,
                      y_codes=None,
                      processed_data_dir=None,
                      yes_plot_latent_space=False,
                      neptune_api_token = NEPTUNE_API_TOKEN,
                      download_models_from_gcp=False,
                      upload_models_to_gcp=False):
    """
    Runs the model training and evaluation pipeline.

    Args:
        data_dir (str): The directory path where the input data is stored.
        params (dict): A dictionary containing the model parameters.
        output_dir (str, optional): The directory path where the output models will be saved. 
            If not provided, a default directory will be used.
        train_name (str, optional): The name of the training dataset. Default is 'train'.
        prefix (str, optional): The prefix to be used for tracking and saving the models. 
            Default is 'training_run'.
        eval_name_list (list, optional): A list of names of the evaluation datasets. 
            Default is ['val'].
        eval_params_list (list, optional): A list of dictionaries containing evaluation parameters. 
            Each dictionary can contain 'y_col_name', 'y_cols', and 'y_head' keys. 
            Default is None.
        run_dict (neptune.metadata_containers.run.Run or neptune.handler.Handler, optional): 
            An object representing the Neptune run. If provided, the models will be tracked and saved to Neptune. 
            Default is None.

    Returns:
        dict: A dictionary containing the evaluation metrics for each evaluation dataset and parameter combination.

    Raises:
        FileNotFoundError: If the encoder_info.json file is not found in the saved_model_dir.

    """

    if run_dict is None:
        run_dict = {}

    if isinstance(run_dict, neptune.metadata_containers.run.Run) or isinstance(run_dict, neptune.handler.Handler):
        print('Using Neptune')
        use_neptune= True
        #TODO: check if the models are already trained on neptune
        run_struc = run_dict.get_structure()
        if not download_models_from_gcp:
            download_models_from_neptune = check_if_path_in_struc(run_struc,f'{prefix}/models/encoder_info')
        # download_models_from_neptune = check_neptune_existance(run_dict,f'{prefix}/models/encoder_info')
        else:
            raise NotImplementedError('Downloading models from GCP not yet implemented')

    if use_neptune:
    #     run_dict[f'{prefix}/dataset'].track_files(data_dir)
    #     run_dict[f'{prefix}/model_name'] = 'Model2925'

        # default_fit_params = run_dict['params'].fetch()
        default_fit_params = run_dict['params/fit_kwargs'].fetch()
        default_fit_params = convert_neptune_kwargs(default_fit_params)
        # find the difference between the default params and the current params
        params_diff = {}
        # for k,v in params.items():
        for k,v in params['fit_kwargs'].items():
            if isinstance(v,dict):
                for kk,vv in v.items():
                    if default_fit_params.get(k) is None:
                        params_diff[k] = v
                    else:
                        if default_fit_params[k].get(kk) != vv:
                            params_diff[k] = v
            else:
                if default_fit_params.get(k) != v:
                    params_diff[k] = v

        # run_dict[f'{prefix}/params_diff'] = stringify_unsupported(params_diff)
        run_dict[f'{prefix}/params_diff/fit_kwargs'] = stringify_unsupported(params_diff)
        
        # run_dict[f'dataset'].track_files(data_dir)
        # run_dict[f'model_name'] = 'Model2925'
        # run_dict[f'params'] = stringify_unsupported(params)

    if eval_params_list is None:
        eval_params_list = [{}]

    if output_dir is None:
        output_dir = os.path.expanduser('~/TEMP_MODELS')

    if processed_data_dir is None:
        processed_data_dir = os.path.expanduser('~/PROCESSED_DATA')

    _, fit_file_id = create_selected_data(input_data_dir=data_dir,
                                                sample_selection_col=fit_subset_col,
                                                subdir_col=subdir_col,
                                                output_dir=processed_data_dir,
                                                selections_df=selections_df)

    X_fit_file = f'{processed_data_dir}/X_{fit_file_id}.csv'
    y_fit_file = f'{processed_data_dir}/y_{fit_file_id}.csv'
    if use_neptune:
        run_dict[f'{prefix}/datasets/X_{fit_file_id}'].track_files(X_fit_file)
        run_dict[f'{prefix}/datasets/y_{fit_file_id}'].track_files(y_fit_file)



    saved_model_dir = os.path.join(output_dir,prefix,'models')        
    os.makedirs(saved_model_dir,exist_ok=True)
    task_components_dict = params['task_kwargs']
    fit_kwargs = params['fit_kwargs']
    encoder_info_dict = params['encoder_desc']
    
    if (not os.path.exists(f'{saved_model_dir}/encoder_info.json')) and (use_neptune) and (download_models_from_gcp):
        raise NotImplementedError('Downloading models from GCP not yet implemented')

    elif (not os.path.exists(f'{saved_model_dir}/encoder_info.json')) and (use_neptune) and (download_models_from_neptune): 
        run_dict[f'{prefix}/models/encoder_state'].download(f'{saved_model_dir}/encoder_state.pt')
        run_dict[f'{prefix}/models/encoder_info'].download(f'{saved_model_dir}/encoder_info.json')
        run_dict[f'{prefix}/models/head_state'].download(f'{saved_model_dir}/head_state.pt')
        run_dict[f'{prefix}/models/head_info'].download(f'{saved_model_dir}/head_info.json')
        if os.path.exists(f'{saved_model_dir}/adv_info.json'):
            run_dict[f'{prefix}/models/adv_state'].download(f'{saved_model_dir}/adv_state.pt')
            run_dict[f'{prefix}/models/adv_info'].download(f'{saved_model_dir}/adv_info.json')
    
    if os.path.exists(f'{saved_model_dir}/encoder_info.json'):
        encoder = create_model_wrapper(f'{saved_model_dir}/encoder_info.json',f'{saved_model_dir}/encoder_state.pt')
        head = create_model_wrapper(f'{saved_model_dir}/head_info.json',f'{saved_model_dir}/head_state.pt',is_encoder=False)
        if os.path.exists(f'{saved_model_dir}/adv_info.json'):
            adv = create_model_wrapper(f'{saved_model_dir}/adv_info.json',f'{saved_model_dir}/adv_state.pt',is_encoder=False)
        else:
            adv = MultiHead([])

    else:
        X_data_fit = pd.read_csv(X_fit_file, index_col=0)
        y_data_fit = pd.read_csv(y_fit_file, index_col=0)

        try:
            _, encoder, head, adv = fit_model_wrapper(X=X_data_fit,
                                                    y=y_data_fit,
                                                    task_components_dict=task_components_dict,
                                                    encoder_info_dict=encoder_info_dict,
                                                    neptune_api_token=neptune_api_token,
                                                    run_dict=run_dict[prefix],
                                                    train_name = fit_file_id,
                                                    **fit_kwargs)

            save_model_wrapper(encoder, head, adv, 
                            save_dir=saved_model_dir,
                            run_dict=run_dict,
                            prefix=prefix,
                            upload_models_to_gcp=upload_models_to_gcp)
        except ValueError as e:
            print(f'Error: {e}')
            return None


    metrics = defaultdict(dict)
    if y_codes is None:
        y_codes = {}
        if use_neptune:
            if check_neptune_existance(run_dict,f'{prefix}/datasets/y_codes'):
                y_codes = run_dict[f'{prefix}/datasets/y_codes'].fetch()
                y_codes = convert_neptune_kwargs(y_codes)

    for eval_subset_col in eval_subset_col_list:
        _, eval_file_id = create_selected_data(input_data_dir=data_dir,
                                                        sample_selection_col=eval_subset_col,
                                                        subdir_col=subdir_col,
                                                        output_dir=processed_data_dir,
                                                        selections_df=selections_df)


        X_eval_file = f'{processed_data_dir}/X_{eval_file_id}.csv'
        y_eval_file = f'{processed_data_dir}/y_{eval_file_id}.csv'
        if use_neptune:
            run_dict[f'{prefix}/datasets/X_{eval_file_id}'].track_files(X_eval_file)
            run_dict[f'{prefix}/datasets/y_{eval_file_id}'].track_files(y_eval_file)


        X_data_eval = pd.read_csv(X_eval_file, index_col=0)
        y_data_eval = pd.read_csv(y_eval_file, index_col=0)

        for eval_params in eval_params_list:
            y_col_name = eval_params.get('y_col_name',None)
            y_cols = eval_params.get('y_cols',None)
            y_head = eval_params.get('y_head',None)
            if y_cols is None:
                y_cols = params['task_kwargs']['y_head_cols']

            try:
                if y_col_name is None:
                    metrics[f'{eval_file_id}' ].update(evaluate_model_wrapper(encoder, head, adv, X_data_eval, y_data_eval,
                                                                        y_cols=y_cols,
                                                                        y_head=y_head,
                                                                        y_codes=y_codes))
                else:
                    metrics[f'{eval_file_id}__head_{y_head}__on_{y_col_name}'].update(evaluate_model_wrapper(encoder, head, adv, X_data_eval, y_data_eval,
                                                                                        y_cols=y_cols,
                                                                                        y_head=y_head,
                                                                                        y_codes=y_codes))
                    # metrics[f'{eval_name}__{y_col_name}'].update(evaluate_model_wrapper(encoder, head, adv, X_data_eval, y_data_eval,
                    #                                                 y_cols=y_cols,
                    #                                                 y_head=y_head)

                if yes_plot_latent_space:
                    if isinstance(y_cols,str):
                        y_cols = [y_cols]
                    if y_cols[0] in y_data_eval.columns:
                        create_latentspace_plots(X_data_eval,y_data_eval, encoder, saved_model_dir, eval_file_id,
                                            run_dict, prefix, plot_latent_space='seaborn', 
                                            plot_latent_space_cols=y_cols)

            except ValueError as e:
                print(f'Error: {e}')
                # print(f'Error in {eval_name}__{y_col_name}')
                # metrics[f'{eval_name}__{y_col_name}'] = None
                
    if use_neptune:
        run_dict[f'{prefix}/metrics'] = metrics
        run_dict.wait()

    return metrics

############################################################

### Function to generate and plot the latent space
def create_latentspace_plots(X_data_eval,y_data_eval, encoder,save_dir,eval_name,
                             run,prefix,plot_latent_space='seaborn',
                             plot_latent_space_cols=None,yes_plot_pca=False):

    # plot_latent_space_cols = y_head_cols

    Z_embed_savepath = os.path.join(save_dir, f'Z_embed_{eval_name}.csv')
        
    if check_neptune_existance(run,f'{prefix}/Z_{eval_name}'):
        print(f'Z_{eval_name} already exists in {prefix} of run')
    
    else:
        Z = generate_latent_space(X_data_eval, encoder)
        Z.to_csv(os.path.join(save_dir, f'Z_{eval_name}.csv'))

        Z_pca = generate_pca_embedding(Z, n_components=4)
        Z_pca.to_csv(os.path.join(save_dir, f'Z_pca_{eval_name}.csv'))
        Z_pca.columns = [f'PCA{i+1}' for i in range(Z_pca.shape[1])]

        Z_umap = generate_umap_embedding(Z)
        Z_umap.to_csv(os.path.join(save_dir, f'Z_umap_{eval_name}.csv'))
        Z_umap.columns = [f'UMAP{i+1}' for i in range(Z_umap.shape[1])]

        Z_embed = pd.concat([Z_pca, Z_umap], axis=1)
        # Z_embed = Z_embed.join(y_data_eval)
        Z_embed.to_csv(Z_embed_savepath)
        run[f'{prefix}/Z_embed_{eval_name}'].upload(Z_embed_savepath)
    run.wait()



    if plot_latent_space_cols is None:
        plot_latent_space_cols = y_data_eval.columns
    
    print('plot_latent_space:', plot_latent_space)
    print('plot_latent_space_cols:', plot_latent_space_cols)

    if plot_latent_space:
        if os.path.exists(Z_embed_savepath):
            Z_embed = pd.read_csv(Z_embed_savepath, index_col=0)
        else:
            # check if the Z_embed file is in neptune
            if check_neptune_existance(run,f'{prefix}/Z_embed_{eval_name}'):
                raise ValueError(f'No Z_embed_{eval_name} file found in run')

            # download the Z_embed file from neptune
            run[f'{prefix}/Z_embed_{eval_name}'].download(Z_embed_savepath)
            Z_embed = pd.read_csv(Z_embed_savepath, index_col=0)

        missing_cols = [col for col in y_data_eval.columns if col not in Z_embed.columns]
        if len(missing_cols) > 0:
            print(f'Adding metadata columns to Z_embed: {missing_cols}')
            Z_embed = Z_embed.join(y_data_eval[missing_cols])
            # Z_embed.to_csv(Z_embed_savepath)
            # run[f'{prefix}/Z_embed_{eval_name}'].upload(Z_embed_savepath)



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
                    palette = assign_color_map(Z_embed[hue_col].unique().dropna())
                    Z_counts = Z_embed[hue_col].value_counts()

                plot_title = f'{prefix} Latent Space of {eval_name} (N={Z_count_sum})'
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
                    run[f'{prefix}/sns_Z_pca_{hue_col}_{eval_name}'].upload(os.path.join(save_dir, f'Z_pca_{hue_col}_{eval_name}.png'))
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
                run[f'{prefix}/sns_Z_umap_{hue_col}_{eval_name}'].upload(os.path.join(save_dir, f'Z_umap_{hue_col}_{eval_name}.png'))
                plt.close()

        if (plot_latent_space=='plotly') or (plot_latent_space=='both') or (plot_latent_space=='px'):
            for hue_col in plot_latent_space_cols:
                if yes_plot_pca:
                    plotly_fig = px.scatter(Z_embed, x='PCA1', y='PCA2', color=hue_col, title=f'PCA {hue_col}')
                    plotly_fig.update_traces(marker=dict(size=2*marker_sz))
                    run[f'{prefix}/px_Z_pca_{hue_col}_{eval_name}'].upload(plotly_fig)
                    plt.close()

                plotly_fig = px.scatter(Z_embed, x='UMAP1', y='UMAP2', color=hue_col)
                plotly_fig.update_traces(marker=dict(size=2*marker_sz))
                run[f'{prefix}/px_Z_umap_{hue_col}_{eval_name}'].upload(plotly_fig)
                plt.close()

        run.wait()

    return Z_embed


def get_model_neptune_object(model_id,project_id,neptune_api_token,model_is_registered):
    if model_is_registered:
        neptune_obj = neptune.init_model(project=project_id,
            api_token= neptune_api_token,
            with_id=model_id,
            mode="read-only")
    else:
        neptune_obj = neptune.init_run(project=project_id,
            api_token= neptune_api_token,
            with_id=model_id,
            mode = 'read-only')
    return neptune_obj


def get_specific_model(model_id,
                       model_loc='pretrain',
                       model_name='encoder',
                       local_dir=None,
                       project_id='revivemed/RCC',
                       neptune_api_token=NEPTUNE_API_TOKEN,
                       model_is_registered=False,
                       download_models_from_gcp=False,
                       use_rand_init=False,
                       overwrite_model_kwargs={}):
    
    if model_id is None:
        raise ValueError('model_id must be provided')
    
    if local_dir is None:
        local_dir = os.path.expanduser('~/PRETRAINED_MODELS')
    os.makedirs(local_dir,exist_ok=True)

    load_dir = os.path.join(local_dir,model_id,model_loc)
    os.makedirs(load_dir,exist_ok=True)
    model_info_path = os.path.join(load_dir,f'{model_name}_info.json')
    model_state_path =  os.path.join(load_dir,f'{model_name}_state.pt')
    neptune_was_used = False

    if os.path.exists(model_info_path):
        print(f'Loading model kwargs from {model_info_path}')
    else:
        neptune_was_used = True
        neptune_obj = get_model_neptune_object(model_id,project_id,neptune_api_token,model_is_registered)
        if model_loc:
            neptune_obj[f'{model_loc}/models/{model_name}_info'].download(model_info_path)
    model_kwargs = load_json(model_info_path)

    if overwrite_model_kwargs:
        for k,v in overwrite_model_kwargs.items():
            if k in model_kwargs:
                print(f'In {model_name}, Overwriting {k} with {v}')
                model_kwargs[k] = v
        # model_kwargs.update(overwrite_model_kwargs)

    model = initialize_model(**model_kwargs)

    if not use_rand_init:

        if os.path.exists(model_state_path):
            print(f'Loading model state from {model_state_path}')
        else:
            if not neptune_was_used:
                neptune_obj = get_model_neptune_object(model_id,project_id,neptune_api_token,model_is_registered)
                neptune_was_used = True

            if download_models_from_gcp:
                raise NotImplementedError('Downloading models from GCP not yet implemented')
            else:
                neptune_obj_struc = neptune_obj.get_structure()
                if model_loc:
                    neptune_obj_struc = neptune_obj_struc[model_loc]
                    if f'models' in neptune_obj_struc.keys():
                        if f'{model_name}_state' in neptune_obj_struc['models'].keys():
                            neptune_obj[f'{model_loc}/models/{model_name}_state'].download(model_state_path)
                        elif f'{model_name}_state_dict' in neptune_obj_struc['models'].keys():
                            neptune_obj[f'{model_loc}/models/{model_name}_state_dict'].download(model_state_path)
                        else:
                            raise ValueError(f'No {model_name}_state or {model_name}_state_dict found in Neptune')
                    elif 'model' in neptune_obj_struc.keys():
                        if f'{model_name}_state' in neptune_obj_struc['model'].keys():
                            neptune_obj[f'{model_loc}/model/{model_name}_state'].download(model_state_path)
                        elif f'{model_name}_state_dict' in neptune_obj_struc['model'].keys():
                            neptune_obj[f'{model_loc}/model/{model_name}_state_dict'].download(model_state_path)
                        else:
                            raise ValueError(f'No {model_name}_state or {model_name}_state_dict found in Neptune')
                    else:
                        raise ValueError(f'No model or models found in Neptune')


        model.load_state_dict(torch.load(model_state_path))
    else:
        print(f'Using random initialization for {model_name}')

    if neptune_was_used:
        neptune_obj.stop()

    return model




### Function to get the encoder
def get_encoder_model(dropout_rate=None, use_rand_init=False, load_dir=None, verbose=False,
                        project_id='revivemed/Survival-RCC',neptune_api_token=NEPTUNE_API_TOKEN,
                        model_id='SUR-MOD', model_is_a_run=False, encoder_model_loc='', 
                        download_models_from_gcp=False):
    """
    Creates an encoder model.

    Args:
        dropout_rate (float, optional): The dropout rate to be set in the encoder model. Defaults to None.
        use_rand_init (bool, optional): Whether to use random initialization for the encoder model. Defaults to False.
        load_dir (str, optional): The directory path to load the pretrained model from. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        encoder (object): The (optionally pretrained) encoder model.

    Raises:
        ValueError: If the 'hidden_size_mult' key is missing in the encoder_kwargs dictionary.

    """
    
    
    if (load_dir is None) and (model_id):
        load_dir = os.path.expanduser('~/PRETRAINED_MODELS/2925')
        os.makedirs(load_dir,exist_ok=True)

    encoder_kwargs_path = os.path.join(load_dir,'encoder_kwargs.json')
    encoder_state_path =  os.path.join(load_dir,'encoder_state.pt')

    if (not os.path.exists(encoder_kwargs_path)) or (not os.path.exists(encoder_state_path)):

        if model_is_a_run:
            neptune_obj = neptune.init_run(project=project_id,
                api_token= neptune_api_token,
                with_id=model_id,
                mode = 'read-only')
        else:
            neptune_obj = neptune.init_model(project=project_id,
                api_token= neptune_api_token,
                with_id=model_id,
                mode="read-only")
            
        neptune_obj_struc = neptune_obj.get_structure()

        if encoder_model_loc:
            neptune_model = neptune_obj[encoder_model_loc]
            neptune_obj_struc = neptune_obj_struc[encoder_model_loc]
        else:
            neptune_model = neptune_obj
    

        if not os.path.exists(encoder_state_path):
            #TODO add option to download from GCP
            if download_models_from_gcp:
                raise NotImplementedError('Downloading models from GCP not yet implemented')
            
            if 'models' in neptune_obj_struc.keys():
                if 'encoder_state' in neptune_obj_struc['models'].keys():
                    neptune_model['models/encoder_state'].download(encoder_state_path)
                elif 'encoder_state_dict' in neptune_obj_struc['models'].keys():
                    neptune_model['models/encoder_state_dict'].download(encoder_state_path)
                else:
                    raise ValueError('No encoder_state or encoder_state_dict found in Neptune')
            elif 'model' in neptune_obj_struc.keys():
                if 'encoder_state' in neptune_obj_struc['model'].keys():
                    neptune_model['model/encoder_state'].download(encoder_state_path)
                elif 'encoder_state_dict' in neptune_obj_struc['model'].keys():
                    neptune_model['model/encoder_state_dict'].download(encoder_state_path)
                else:
                    raise ValueError('No encoder_state or encoder_state_dict found in Neptune')
            else:
                raise ValueError('No model or models found in Neptune')

            # neptune_model['model/encoder_kwargs'].download(encoder_kwargs_path)
        if not os.path.exists(encoder_kwargs_path):
            print('WARNING: encoder_kwargs.json not found, attempt to get from Neptune')
            encoder_kwargs = neptune_model['original_kwargs/encoder_kwargs'].fetch()
            if 'input_size' not in encoder_kwargs:
                encoder_kwargs['input_size'] = 2736
            if 'kind' not in encoder_kwargs:
                encoder_kwargs['kind'] = neptune_model['original_kwargs/encoder_kind'].fetch()
            if 'hidden_size' not in encoder_kwargs:
                if 'hidden_size_mult' in encoder_kwargs:
                    latent_size = encoder_kwargs['latent_size']
                    encoder_kwargs['hidden_size'] = int(encoder_kwargs['hidden_size_mult']*latent_size)
                else:
                    raise ValueError()
                # remove the hidden_size_mult key
                encoder_kwargs.pop('hidden_size_mult')
            save_json(encoder_kwargs,encoder_kwargs_path)

        neptune_obj.stop()

    encoder_kwargs = load_json(encoder_kwargs_path)
    if (dropout_rate is not None):
        if verbose: print('Setting dropout rate to',dropout_rate)
        encoder_kwargs['dropout_rate'] = dropout_rate

    encoder = get_encoder(**encoder_kwargs)

    if not use_rand_init:
        encoder.load_state_dict(torch.load(encoder_state_path))

    return encoder



#### Function to get the model heads
def get_model_heads(head_kwargs_dict, backup_input_size=None, verbose=False):
    """
    Returns a MultiHead object based on the provided head_kwargs_dict.

    Args:
        head_kwargs_dict (dict or list): A dictionary or list of dictionaries containing the keyword arguments for each head.
        backup_input_size (int, optional): The backup input size to use if 'input_size' is not specified in the head_kwargs. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        MultiHead: A MultiHead object containing the heads created based on the provided head_kwargs_dict.

    Raises:
        ValueError: If the head_kwargs_dict is not a dictionary or a list.

    Example:
        head_kwargs_dict = {
            'head1': {'input_size': 100, 'output_size': 10},
            'head2': {'input_size': 200, 'output_size': 20}
        }
        heads = get_model_heads(head_kwargs_dict, backup_input_size=50, verbose=True)
    """
    
    # Function implementation goes here
    
    if head_kwargs_dict is None:
        return MultiHead([]) #get_head(kind='Dummy')
    
    if isinstance(head_kwargs_dict,dict):
        head_kwargs_list = [head_kwargs_dict[k] for k in head_kwargs_dict.keys()]
    elif isinstance(head_kwargs_dict,list):
        head_kwargs_list = head_kwargs_dict
    else:
        raise ValueError(f'Invalid head_kwargs_dict type: {type(head_kwargs_dict)}')

    head_list = []
    if len(head_kwargs_list) == 0:
       return MultiHead([])

    for h_kwargs in head_kwargs_list:
            
        if 'input_size' not in h_kwargs:
            if backup_input_size is None:
                raise ValueError('backup_input_size is None')
            if verbose: print('Setting input_size to',backup_input_size)
            h_kwargs['input_size'] = backup_input_size

        head = get_head(**h_kwargs)
        head_list.append(head)


    head = MultiHead(head_list)
    return head


### 
def build_components_of_model(head_desc_dict=None, 
                                adv_desc_dict=None,
                                encoder_desc_dict=None,
                                backup_input_size=2736,
                                local_dir=None,
                                neptune_api_token=NEPTUNE_API_TOKEN,
                                verbose = 0):
    """
    Builds the model components

    Args:


    Returns:
        tuple: A tuple containing the encoder model, head model, and adversary model.
    """
    

    encoder_kwargs = encoder_desc_dict.get('encoder_kwargs',encoder_desc_dict.get('kwargs',None))
    encoder_pretrained_info = encoder_desc_dict.get('encoder_pretrained_info',encoder_desc_dict.get('pretrained_info',None))

    head_kwargs_dict = head_desc_dict.get('head_kwargs_dict',head_desc_dict.get('head_kwargs',head_desc_dict.get('kwargs',None)))
    head_pretrained_info = head_desc_dict.get('head_pretrained_info',head_desc_dict.get('pretrained_info',None))

    adv_kwargs_dict = adv_desc_dict.get('adv_kwargs_dict',adv_desc_dict.get('adv_kwargs',adv_desc_dict.get('kwargs',None)))
    adv_pretrained_info = adv_desc_dict.get('adv_pretrained_info',adv_desc_dict.get('pretrained_info',None))

    if encoder_pretrained_info:
        if encoder_kwargs:
            print('WARNING: encoder_kwargs provided, this might cause errors when using a pretrained model')


        encoder = get_specific_model(model_id = encoder_pretrained_info.get('model_id',None),
                                    model_loc = encoder_pretrained_info.get('model_loc','pretrain'),
                                    model_name = encoder_pretrained_info.get('model_name','encoder'),
                                    local_dir=local_dir,
                                    project_id=encoder_pretrained_info.get('project_id','revivemed/RCC'),
                                    neptune_api_token=neptune_api_token,
                                    model_is_registered=encoder_pretrained_info.get('model_is_registered',False),
                                    download_models_from_gcp=encoder_pretrained_info.get('download_models_from_gcp',False),
                                    use_rand_init=encoder_pretrained_info.get('use_rand_init',False),
                                    overwrite_model_kwargs=encoder_kwargs)
    else:
        if 'input_size' not in encoder_kwargs:
            encoder_kwargs['input_size'] = backup_input_size
        encoder = initialize_model(**encoder_kwargs)

    if head_pretrained_info:
        if head_kwargs_dict:
            print('WARNING: head_kwargs_dict provided, this might cause errors when using a pretrained model')

        head = get_specific_model(model_id = head_pretrained_info.get('model_id',None),
                                    model_loc = head_pretrained_info.get('model_loc','pretrain'),
                                    model_name = head_pretrained_info.get('model_name','head'),
                                    local_dir=local_dir,
                                    project_id=head_pretrained_info.get('project_id','revivemed/RCC'),
                                    neptune_api_token=neptune_api_token,
                                    model_is_registered=head_pretrained_info.get('model_is_registered',False),
                                    download_models_from_gcp=head_pretrained_info.get('download_models_from_gcp',False),
                                    use_rand_init=head_pretrained_info.get('use_rand_init',False),
                                    overwrite_model_kwargs=head_kwargs_dict)
    else:
        # TODO: We need better handling of head input size to account for the size of the other_vars
        head = get_model_heads(head_kwargs_dict, backup_input_size=encoder.latent_size + 1)

    if adv_pretrained_info:
        if adv_kwargs_dict:
            print('WARNING: adv_kwargs_dict provided, this might cause errors when using a pretrained model')

        adv = get_specific_model(model_id = adv_pretrained_info.get('model_id',None),
                                    model_loc = adv_pretrained_info.get('model_loc','pretrain'),
                                    model_name = adv_pretrained_info.get('model_name','adv'),
                                    local_dir=local_dir,
                                    project_id=adv_pretrained_info.get('project_id','revivemed/RCC'),
                                    neptune_api_token=neptune_api_token,
                                    model_is_registered=adv_pretrained_info.get('model_is_registered',False),
                                    download_models_from_gcp=adv_pretrained_info.get('download_models_from_gcp',False),
                                    use_rand_init=adv_pretrained_info.get('use_rand_init',False),
                                    overwrite_model_kwargs=adv_kwargs_dict)
    else:
        adv = get_model_heads(adv_kwargs_dict, backup_input_size=encoder.latent_size)

    if head.kind == 'MultiHead':
        head.name = 'HEAD'

    if adv.kind == 'MultiHead':
        adv.name = 'ADVERSARY'

    return encoder, head, adv







### Function to build the model components for fine-tuning
def build_model_components(head_kwargs_dict, adv_kwargs_dict=None, dropout_rate=None, use_rand_init=False,
                           encoder=None,encoder_info_dict=None,verbose=0,neptune_api_token=NEPTUNE_API_TOKEN,
                           download_models_from_gcp=False,input_size=2736):
    """
    Builds the model components for fine-tuning.

    Args:
        head_kwargs_dict (dict): A dictionary containing the keyword arguments for the head model.
        adv_kwargs_dict (dict, optional): A dictionary containing the keyword arguments for the adversary model. Defaults to None.
        dropout_rate (float, optional): The dropout rate. Defaults to None.
        use_rand_init (bool, optional): Whether to use random initialization. Defaults to False.

    Returns:
        tuple: A tuple containing the encoder model, head model, and adversary model.
    """
    
    encoder_kwargs = encoder_info_dict.get('encoder_kwargs',None)
    encoder_load_dir = encoder_info_dict.get('encoder_load_dir',None)
    encoder_project_id = encoder_info_dict.get('encoder_project_id',None)
    encoder_model_id = encoder_info_dict.get('encoder_model_id',None)
    encoder_is_a_run = encoder_info_dict.get('encoder_is_a_run',False)
    encoder_model_loc = encoder_info_dict.get('encoder_model_loc','')
    download_models_from_gcp = encoder_info_dict.get('download_models_from_gcp',False)

    if (encoder is None) and (encoder_kwargs is None):
        encoder = get_encoder_model(dropout_rate=dropout_rate, 
                                        use_rand_init=use_rand_init,
                                        load_dir=encoder_load_dir, 
                                        verbose=False,
                                        project_id=encoder_project_id,
                                        neptune_api_token=neptune_api_token,
                                        model_id=encoder_model_id,
                                        model_is_a_run=encoder_is_a_run,
                                        encoder_model_loc=encoder_model_loc,
                                        download_models_from_gcp=download_models_from_gcp)
        
    elif encoder_kwargs is not None:
        if 'input_size' not in encoder_kwargs:
            encoder_kwargs['input_size'] = input_size
        assert encoder_kwargs['input_size'] == input_size, 'Encoder input size does not match the provided input size'
        encoder = get_encoder(**encoder_kwargs)
        if (dropout_rate is not None):
            if verbose: print('Setting dropout rate to',dropout_rate)
            encoder_kwargs['dropout_rate'] = dropout_rate


    # TODO: We need better handling of head input size to account for the size of the other_vars
    head = get_model_heads(head_kwargs_dict, backup_input_size=encoder.latent_size + 1)
    if head.kind == 'MultiHead':
        head.name = 'HEAD'

    adv = get_model_heads(adv_kwargs_dict, backup_input_size=encoder.latent_size)
    if adv.kind == 'MultiHead':
        adv.name = 'ADVERSARY'

    return encoder, head, adv






def fit_model_wrapper(X, y, task_components_dict={}, encoder_info_dict={}, run_dict={},
                      neptune_api_token=NEPTUNE_API_TOKEN, local_dir=None, **fit_kwargs):
    """
    Fits a compound model using the provided data and model components.

    Parameters:
    - X (numpy.ndarray): The input data for training the model.
    - y (pandas.DataFrame): The target variable(s) for training the model.
    - task_components_dict (dict): A dictionary containing the model component defaults.
    - run_dict (neptune.metadata_containers.run.Run or neptune.handler.Handler): 
        An optional Neptune run object for recording the model fitting.
    - **fit_kwargs: Additional keyword arguments for training the model.

    Returns:
    - run_dict (neptune.metadata_containers.run.Run or neptune.handler.Handler): 
        The updated Neptune run object.
    - encoder: The trained encoder component of the compound model.
    - head: The trained head component of the compound model.
    - adv: The trained adversary component of the compound model.
    """
    
    assert isinstance(task_components_dict,dict), 'task_components_dict should be a dictionary'

    if isinstance(run_dict, neptune.metadata_containers.run.Run) or isinstance(run_dict, neptune.handler.Handler):
        print('Record the model fitting to Neptune')
        use_neptune= True
    else:
        use_neptune = False
        
    ### Model Component Defaults
    y_head_cols = task_components_dict.get('y_head_cols',None)
    y_adv_cols = task_components_dict.get('y_adv_cols',None)
    head_kwargs_dict = task_components_dict.get('head_kwargs_dict',None)
    adv_kwargs_dict = task_components_dict.get('adv_kwargs_dict',None)
    use_pretrained_head = task_components_dict.get('use_pretrained_head',False)
    use_pretrained_adv = task_components_dict.get('use_pretrained_adv',False)
    input_size = X.shape[1]
    encoder_kwargs = encoder_info_dict.get('encoder_kwargs',None)
    encoder_model_id = encoder_info_dict.get('model_id',None)
    if encoder_model_id is None:
        use_pretrained_encoder = False
    else:
        use_pretrained_encoder= True

    if y_head_cols is None:
        # by default just select all of the numeric columns
        print('WARNING: y_head_cols not provided, selecting all numeric columns')
        y_head_cols = list(y.select_dtypes(include=[np.number]).columns)

    if y_adv_cols is None:
        y_adv_cols = []

    ### Train Defaults
    dropout_rate = fit_kwargs.get('dropout_rate', None)
    use_rand_init = fit_kwargs.get('use_rand_init', False)
    batch_size = fit_kwargs.get('batch_size', 64)
    holdout_frac = fit_kwargs.get('holdout_frac', 0)
    early_stopping_patience = fit_kwargs.get('early_stopping_patience', 0)
    scheduler_kind = fit_kwargs.get('scheduler_kind', None)
    train_name = fit_kwargs.get('train_name', 'fit')
    
    remove_nans = fit_kwargs.get('remove_nans', False)
    how_remove_nans = fit_kwargs.get('how_remove_nans', False)
    yes_clean_batches = fit_kwargs.get('yes_clean_batches', True)
    y_codes = fit_kwargs.get('y_codes', {})
    y_code_keys = fit_kwargs.get('y_code_keys', [])

    if use_pretrained_encoder:
        encoder_pretrained_info ={ 
            'model_id': encoder_info_dict.get('model_id',None),
            'model_loc': encoder_info_dict.get('model_loc','pretrain'),
            'model_name': 'encoder',
            'project_id': encoder_info_dict.get('project_id','revivemed/RCC'),
            'model_is_registered': encoder_info_dict.get('model_is_registered',False),
            'download_models_from_gcp': encoder_info_dict.get('download_models_from_gcp',False),
            'use_rand_init':use_rand_init
        }

        encoder_desc_dict = {'encoder_kwargs':None, 'encoder_pretrained_info':encoder_pretrained_info}
        if dropout_rate is not None:
            encoder_desc_dict['encoder_kwargs'] = {'dropout_rate':dropout_rate}
        else:
            encoder_desc_dict['encoder_kwargs'] = None

    else:
        encoder_desc_dict = {'encoder_kwargs':encoder_kwargs, 'encoder_pretrained_info':None}
        

    if use_pretrained_head and use_pretrained_encoder:
        head_pretrained_info ={ 
            'model_id': encoder_info_dict.get('model_id',None),
            'model_loc': encoder_info_dict.get('model_loc','pretrain'),
            'model_name': 'head',
            'project_id': encoder_info_dict.get('project_id','revivemed/RCC'),
            'model_is_registered': encoder_info_dict.get('model_is_registered',False),
            'download_models_from_gcp': encoder_info_dict.get('download_models_from_gcp',False),
            'use_rand_init':use_rand_init
        }
        head_desc_dict = {'head_kwargs_dict':None, 'head_pretrained_info':head_pretrained_info}
    else:
        head_desc_dict = {'head_kwargs_dict':head_kwargs_dict, 'head_pretrained_info':None}

    if use_pretrained_adv and use_pretrained_encoder:
        adv_pretrained_info ={ 
            'model_id': encoder_info_dict.get('model_id',None),
            'model_loc': encoder_info_dict.get('model_loc','pretrain'),
            'model_name': 'adv',
            'project_id': encoder_info_dict.get('project_id','revivemed/RCC'),
            'model_is_registered': encoder_info_dict.get('model_is_registered',False),
            'download_models_from_gcp': encoder_info_dict.get('download_models_from_gcp',False),
            'use_rand_init': use_rand_init
        }
        adv_desc_dict = {'adv_kwargs_dict':None, 'adv_pretrained_info':adv_pretrained_info}
    else:
        adv_desc_dict = {'adv_kwargs_dict':adv_kwargs_dict, 'adv_pretrained_info':None}

    ### Prepare the Data Loader
    X_size = X.shape[1]
    if (holdout_frac > 0) and (early_stopping_patience < 1) and (scheduler_kind is None):
        # raise ValueError('holdout_frac > 0 and early_stopping_patience < 1 is not recommended')
        print('holdout_frac > 0 and early_stopping_patience < 1 is not recommended, set hold out frac to 0')
        print('UNLESS you are using a scheduler, in which case the holdout_frac is used for the scheduler')
        holdout_frac = 0
        if yes_clean_batches:
            batch_size = get_clean_batch_sz(X_size, batch_size)
    else:
        if yes_clean_batches:
            batch_size = get_clean_batch_sz(X_size*(1-holdout_frac), batch_size)

    X, y = remove_rows_with_y_nans(X, y, y_head_cols+y_adv_cols,how=how_remove_nans)
    y_head_df = y[y_head_cols]
    y_adv_df = y[y_adv_cols]
    
    if remove_nans:
        print('Depreciated "remove_nans" argument, use "how_remove_nans" instead')
    # if remove_nans:
    #     y_head_nan_locs = y_head_df.isna().any(axis=1)
    #     if y_adv_df.shape[1] > 0:
    #         y_adv_nan_locs = y_adv_df.isna().any(axis=1)
    #         nan_locs = y_head_nan_locs | y_adv_nan_locs
    #     else:
    #         nan_locs = y_head_nan_locs

    #     X = X[~nan_locs]
    #     y_head_df = y_head_df[~nan_locs]
    #     y_adv_df = y_adv_df[~nan_locs]


    fit_dataset = CompoundDataset(X,y_head_df, y_adv_df,y_codes=y_codes)

    y_codes = fit_dataset.y_codes
    for key, val in y_codes.items():
        if key in y_code_keys:
            continue
        if len(val) < 2:
            continue
        # if use_neptune:
        run_dict[f'datasets/y_codes_keys'].append(key)
        run_dict[f'datasets/y_codes/{key}'].extend(val)
        y_code_keys.append(key)

    # stratify on the adversarial column (stratify=2)
    # this is probably not the most memory effecient method, would be better to do stratification before creating the dataset
    # train_loader_dct = create_dataloaders(fit_dataset, batch_size, holdout_frac, set_name=train_name, stratify=2)
    train_loader_dct = create_dataloaders_old(fit_dataset, batch_size, holdout_frac, set_name=train_name)


    ### Build the Model Components
    encoder, head, adv = build_components_of_model(encoder_desc_dict=encoder_desc_dict,
                                                    head_desc_dict=head_desc_dict,
                                                    adv_desc_dict=adv_desc_dict,
                                                    backup_input_size=input_size,
                                                    local_dir=local_dir,
                                                    neptune_api_token=neptune_api_token,
                                                    verbose=0)


    # encoder, head, adv = build_model_components(head_kwargs_dict=head_kwargs_dict,
    #                                             adv_kwargs_dict=adv_kwargs_dict,
    #                                             dropout_rate=dropout_rate,
    #                                             use_rand_init=use_rand_init,
    #                                             encoder_info_dict=encoder_info_dict,
    #                                             input_size=input_size)

    if use_pretrained_head:
        raise NotImplementedError('use_pretrained_head not implemented')
    
    if use_pretrained_adv:
        raise NotImplementedError('use_pretrained_adv not implemented')

    if fit_dataset is not None:
        #TODO load the class weights from the model info json
        head.update_class_weights(fit_dataset.y_head)
        adv.update_class_weights(fit_dataset.y_adv)

    if len(adv.heads)==0:
        fit_kwargs['adversary_weight'] = 0

    ### Train the Model
    encoder, head, adv = train_compound_model(train_loader_dct, 
                                            encoder, head, adv, 
                                            run=run_dict, **fit_kwargs)
    if encoder is None:
        raise ValueError('Encoder is None after training, training failed')

    
    return run_dict, encoder, head, adv



def save_model_wrapper(encoder, head, adv, save_dir=None, run_dict={}, prefix='training_run',upload_models_to_gcp = False):
    """
    Saves the encoder, head, and adversary models to the specified directory.

    Args:
        encoder (Encoder): The encoder model.
        head (Head): The head model.
        adv (Adversary): The adversary model.
        save_dir (str, optional): The directory to save the models. If not provided, a temporary directory will be used. Defaults to None.
        run_dict (dict, optional): A dictionary containing information about the Neptune run. Defaults to {}.
        prefix (str, optional): The prefix to use for the Neptune run. Defaults to 'training_run'.

    Returns:
        dict: A dictionary containing information about the Neptune run.

    Raises:
        NotImplementedError: If `upload_models_to_gcp` is set to True.

    Note:
        - If `save_dir` is not provided, a temporary directory will be created at '~/TEMP_MODELS'.
        - If `run_dict` is an instance of `neptune.metadata_containers.run.Run` or `neptune.handler.Handler`, the models will be saved to Neptune.
        - If `upload_models_to_gcp` is set to True, an exception will be raised as this functionality is not implemented.
    """
    

    if isinstance(run_dict, neptune.metadata_containers.run.Run) or isinstance(run_dict, neptune.handler.Handler):
        print('Save models to Neptune')
        use_neptune= True
    else:
        use_neptune = False

    delete_after_upload = False
    if save_dir is None:
        save_dir = os.path.expanduser('~/TEMP_MODELS')
        if use_neptune:
            delete_after_upload = True
            if os.path.exists(save_dir):
                # delete the directory
                shutil.rmtree(save_dir)
        os.makedirs(save_dir,exist_ok=True)



    encoder.save_state_to_path(save_dir,save_name='encoder_state.pt')
    encoder.save_info(save_dir,save_name='encoder_info.json')
    head.save_state_to_path(save_dir,save_name='head_state.pt')
    head.save_info(save_dir,save_name='head_info.json')
    adv.save_state_to_path(save_dir,save_name='adv_state.pt')
    adv.save_info(save_dir,save_name='adv_info.json')

    # torch.save(head.state_dict(), f'{save_dir}/{setup_id}_head_state_dict.pth')
    # torch.save(adv.state_dict(), f'{save_dir}/{setup_id}_adv_state_dict.pth')
    if use_neptune:
        if not upload_models_to_gcp:
            run_dict[f'{prefix}/models/encoder_state'].upload(f'{save_dir}/encoder_state.pt')
            run_dict[f'{prefix}/models/encoder_info'].upload(f'{save_dir}/encoder_info.json')
            run_dict[f'{prefix}/models/head_state'].upload(f'{save_dir}/head_state.pt')
            run_dict[f'{prefix}/models/head_info'].upload(f'{save_dir}/head_info.json')
            run_dict[f'{prefix}/models/adv_state'].upload(f'{save_dir}/adv_state.pt')
            run_dict[f'{prefix}/models/adv_info'].upload(f'{save_dir}/adv_info.json')
            run_dict.wait()

    if upload_models_to_gcp:
        raise NotImplementedError('upload_models_to_gcp not implemented')

    if use_neptune and delete_after_upload:
        run_dict.wait()
        shutil.rmtree(save_dir)

    return run_dict



def evaluate_model_wrapper(encoder, head, adv, X_data_eval, y_data_eval, y_cols, y_head=None, y_codes={}):
    """
    Wrapper function to evaluate a model.

    Parameters:
    - encoder: The encoder model.
    - head: The head model.
    - adv: The adversarial model.
    - X_data_eval: The input data for evaluation.
    - y_data_eval: The target data for evaluation.
    - y_cols: The columns to be used as target variables.
    - y_head: The name of the specific head to be used for evaluation.

    Returns:
    - The evaluation score of the model.

    Raises:
    - ValueError: If an invalid head name or y_cols length is provided.
    """

    if y_head is None:
        chosen_head = head
    
    elif y_head in ['Encoder','Autoencoder','Reconstruction','Recon','Decoder']:
        X = torch.tensor(X_data_eval.to_numpy(),dtype=torch.float32)
        # device = next(encoder.parameters()).device
        # X.to(device)
        encoder.eval()
        with torch.inference_mode():
            recon_score = encoder.score(X,X)
        return recon_score
    
    else:
        multihead_name_list = head.heads_names
        if y_head not in multihead_name_list:
            return {}
            # raise ValueError(f'Invalid head name: {y_head}')

        chosen_head_idx = multihead_name_list.index(y_head)
        chosen_head = head.heads[chosen_head_idx]

        if isinstance(chosen_head.y_idx,list):
            if len(y_cols) != len(chosen_head.y_idx):
                raise ValueError(f'Invalid y_cols length: {len(y_cols)} vs {len(chosen_head.y_idx)}')
            if len(y_cols) == 1:
                chosen_head.y_idx = [0]
            else:
                chosen_head.y_idx = list(range(len(y_cols)))
        else:
            if len(y_cols) != 1:
                raise ValueError(f'Invalid y_cols length: {len(y_cols)} vs {len(chosen_head.y_idx)}')
            chosen_head.y_idx = 0

        # if len(y_cols) != len(chosen_head.y_idx):
        #     raise ValueError(f'Invalid y_cols length: {len(y_cols)} vs {len(chosen_head.y_idx)}')
        # if len(y_cols) == 1:
        #     chosen_head.y_idx = 0
        # else:
        #     chosen_head.y_idx = list(range(len(y_cols)))
    encoder.to('cpu')
    chosen_head.to('cpu')

    if (chosen_head.kind == 'Dummy'):
        return {}
    if (chosen_head.kind == 'MultiHead') and (len(chosen_head.heads)==0):
        return {}
    
    model = CompoundModel(encoder, chosen_head)
    skmodel = create_pytorch_model_from_info(full_model=model)

    y_data, _ = convert_y_data_by_codes(y_data_eval[y_cols], y_codes)
    return skmodel.score(X_data_eval.to_numpy(),y_data.to_numpy().astype(np.float32))


############################################################
############################################################


def choose_subset_of_feats(X_data, feature_subset=0.25):
    if isinstance(feature_subset, float):
        # choose random subset of features
        feature_subset = np.random.choice(X_data.columns, int(feature_subset*X_data.shape[1]), replace=False)
        if X_data.shape[1] > 0:
            X_data = X_data[feature_subset]
    return X_data, feature_subset


def remove_rows_with_y_nans(X_data, y_data, subset_cols=None,how='any'):
    """
    Removes rows from the input and target data where the target data contains NaN values.

    Args:
        X_data (pandas.DataFrame): The input data.
        y_data (pandas.DataFrame): The target data.

    Returns:
        tuple: A tuple containing the filtered input and target data.
    """
    if subset_cols is None:
        subset_cols = y_data.columns


    # y_data.dropna(subset=subset_cols, how=how, axis=0, inplace=True)
    # X_data = X_data.loc[y_data.index].copy()

    if how == 'any':
        nan_locs = y_data[subset_cols].isna().any(axis=1)
    elif how == 'all':
        nan_locs = y_data[subset_cols].isna().all(axis=1)
    else:
        return X_data, y_data

    X_data = X_data.loc[~nan_locs,:]
    y_data = y_data.loc[~nan_locs,:]

    return X_data, y_data


###############################################################
###############################################################

default_eval_params_list = [
    # {},

    ###### Pretraining Tasks ######
    {
        'y_col_name':None, 
        'y_head': 'Encoder',
        'y_cols':['']},
    {
        'y_col_name':'Sex', 
        'y_head': 'Sex',
        'y_cols':['Sex']},
    {
        'y_col_name':'Age',
        'y_head':'Age',
        'y_cols':['Age']},
    {
        'y_col_name':'Study ID',
        'y_head':'Study ID',
        'y_cols':['Study ID']},
    {
        'y_col_name':'Cohort Label',
        'y_head':'Cohort Label',
        'y_cols':['Cohort Label v0']},
    {
        'y_col_name':'is Pediatric',
        'y_head':'is Pediatric',
        'y_cols':['is Pediatric']},
    {
        'y_col_name':'Smoking',
        'y_head':'Smoking',
        'y_cols':['Smoking Status']},


    ###### OS ######
    {
        'y_col_name':'NIVO OS',
        'y_head':'NIVO OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'NIVO OS',
        'y_head':'EVER OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'NIVO OS',
        'y_head':'Both OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col


    {
        'y_col_name':'EVER OS',
        'y_head':'NIVO OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'EVER OS',
        'y_head':'EVER OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'EVER OS',
        'y_head':'Both OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    

    {
        'y_col_name':'Both OS',
        'y_head':'EVER OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'Both OS',
        'y_head':'NIVO OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col   
    {
        'y_col_name':'Both OS',
        'y_head':'Both OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col             
    
        ###### OS Alt ######
    {
        'y_col_name':'NIVO-OS',
        'y_head':'NIVO-OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'NIVO-OS',
        'y_head':'EVER-OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'NIVO-OS',
        'y_head':'Both-OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col


    {
        'y_col_name':'EVER-OS',
        'y_head':'NIVO-OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'EVER-OS',
        'y_head':'EVER-OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'EVER-OS',
        'y_head':'Both-OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    

    {
        'y_col_name':'Both-OS',
        'y_head':'EVER-OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'Both-OS',
        'y_head':'NIVO-OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col   
    {
        'y_col_name':'Both-OS',
        'y_head':'Both-OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col     


    ###### PFS ######
    {
        'y_col_name':'NIVO PFS',
        'y_head':'NIVO PFS', # which head to apply to the y_col
        'y_cols': ['NIVO PFS','PFS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'NIVO PFS',
        'y_head':'EVER PFS', # which head to apply to the y_col
        'y_cols': ['NIVO PFS','PFS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'NIVO PFS',
        'y_head':'Both PFS', # which head to apply to the y_col
        'y_cols': ['NIVO PFS','PFS_Event']}, # which columns to use for the y_col


    {
        'y_col_name':'EVER PFS',
        'y_head':'NIVO PFS', # which head to apply to the y_col
        'y_cols': ['EVER PFS','PFS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'EVER PFS',
        'y_head':'EVER PFS', # which head to apply to the y_col
        'y_cols': ['EVER PFS','PFS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'EVER PFS',
        'y_head':'Both PFS', # which head to apply to the y_col
        'y_cols': ['EVER PFS','PFS_Event']}, # which columns to use for the y_col
    

    {
        'y_col_name':'Both PFS',
        'y_head':'EVER PFS', # which head to apply to the y_col
        'y_cols': ['PFS','PFS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'Both PFS',
        'y_head':'NIVO PFS', # which head to apply to the y_col
        'y_cols': ['PFS','PFS_Event']}, # which columns to use for the y_col   
    {
        'y_col_name':'Both PFS',
        'y_head':'Both PFS', # which head to apply to the y_col
        'y_cols': ['PFS','PFS_Event']}, # which columns to use for the y_col  

    ###### Prognostic Markers ######
    {
        'y_col_name':'MSKCC BINARY',
        'y_head':'MSKCC', # which head to apply to the y_col
        'y_cols': ['MSKCC BINARY']}, # which columns to use for the y_col
    {
        'y_col_name':'IMDC BINARY',
        'y_head':'IMDC', # which head to apply to the y_col
        'y_cols': ['IMDC BINARY']}, # which columns to use for the y_col

    {
        'y_col_name':'MSKCC ORDINAL',
        'y_head':'MSKCC_Ordinal', # which head to apply to the y_col
        'y_cols': ['MSKCC ORDINAL']}, # which columns to use for the y_col
    {
        'y_col_name':'IMDC ORDINAL',
        'y_head':'IMDC_Ordinal', # which head to apply to the y_col
        'y_cols': ['IMDC ORDINAL']}, # which columns to use for the y_col

    {
        'y_col_name':'MSKCC ORDINAL',
        'y_head':'MSKCC_MultiClass', # which head to apply to the y_col
        'y_cols': ['MSKCC ORDINAL']}, # which columns to use for the y_col
    {
        'y_col_name':'IMDC ORDINAL',
        'y_head':'IMDC_MultiClass', # which head to apply to the y_col
        'y_cols': ['IMDC ORDINAL']}, # which columns to use for the y_col        

    ###### Benefit ######

    {
        'y_col_name': 'Benefit BINARY',
        'y_head': 'Benefit',
        'y_cols': ['Benefit BINARY']},

    ###### LungCancer Binary ######
    {
        'y_col_name': 'LungCancer BINARY',
        'y_head': 'LungCancer',
        'y_cols': ['LungCancer BINARY']},

    {
        'y_col_name': 'Cancer',
        'y_head': 'Cancer',
        'y_cols': ['Cancer']},

    ###### Stanford BMI ######
    {
        'y_col_name': 'BMI',
        'y_head': 'BMI',
        'y_cols': ['BMI']},
]    