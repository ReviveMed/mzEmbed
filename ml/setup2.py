# setup a neptune run


import torch
import pandas as pd
import numpy as np
import os
import json
from models import get_model, Binary_Head, Dummy_Head, MultiClass_Head, MultiHead
from train3 import CompoundDataset, train_compound_model, get_end_state_eval_funcs, evaluate_compound_model, create_dataloaders
import neptune
from neptune.utils import stringify_unsupported
from utils_neptune import check_neptune_existance, start_neptune_run, convert_neptune_kwargs
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



def setup_neptune_run(data_dir,setup_id,with_run_id=None,**kwargs):
    print(setup_id)
    run, is_run_new = start_neptune_run(with_run_id=with_run_id)
    if not is_run_new:
        setup_is_new = not check_neptune_existance(run,f'{setup_id}/kwargs')
        if setup_is_new:
            setup_is_new = not check_neptune_existance(run,f'{setup_id}/original_kwargs')
    else:
        setup_is_new = False
    
    print(f'is_run_new: {is_run_new}, setup_is_new: {setup_is_new}')

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

        load_model_loc = kwargs.get('load_model_loc', False)
        load_encoder_loc = kwargs.get('load_encoder_loc', load_model_loc)
        load_head_loc = kwargs.get('load_head_loc', load_model_loc)
        load_adv_loc = kwargs.get('load_adv_loc', load_model_loc)
        y_head_cols = kwargs.get('y_head_cols', ['is Pediatric'])
        y_adv_cols = kwargs.get('y_adv_cols', ['Study ID ENC'])
        
        if not isinstance(y_head_cols, list):
            y_head_cols = [y_head_cols]

        if not isinstance(y_adv_cols, list):
            y_adv_cols = [y_adv_cols]
        
        if load_encoder_loc:
            print('loading pretrained encoders, overwriting encoder_kwargs')
            load_kwargs = run[f'{load_encoder_loc}/kwargs'].fetch()
            load_kwargs = convert_neptune_kwargs(load_kwargs)
            # kwargs['encoder_kwargs'].update(load_kwargs['encoder_kwargs'])
            if kwargs['encoder_kind'] != load_kwargs['encoder_kind']:
                raise ValueError(f'Encoder kind mismatch: {kwargs["encoder_kind"]} vs {load_kwargs["encoder_kind"]}')
            
            encoder_kwargs = load_kwargs.get('encoder_kwargs', {})
            encoder_kwargs.update(kwargs.get('encoder_kwargs', {}))
            kwargs['encoder_kwargs'] = encoder_kwargs
            print('encoder_kwargs:', kwargs['encoder_kwargs'])

        if load_head_loc:
            print('loading pretrained heads, overwriting head_kwargs_list')
            load_kwargs = run[f'{load_head_loc}/kwargs'].fetch()
            kwargs['head_kwargs_dict'] = load_kwargs.get('head_kwargs_dict', {})
            kwargs['head_kwargs_list'] = eval(load_kwargs.get('head_kwargs_list', '[]'))
            # assert len(kwargs['head_kwargs_list']) <= len(y_head_cols)

        if load_adv_loc:
            print('loading pretrained advs, overwriting adv_kwargs_list')
            load_kwargs = run[f'{load_adv_loc}/kwargs'].fetch()
            kwargs['adv_kwargs_dict'] = load_kwargs.get('adv_kwargs_dict', {})
            kwargs['adv_kwargs_list'] = eval(load_kwargs.get('adv_kwargs_list', '[]'))
            # assert len(kwargs['adv_kwargs_list']) <= len(y_adv_cols)

        if check_neptune_existance(run,f'{setup_id}/kwargs'):
            if not overwrite_existing_kwargs:
                raise ValueError(f'{setup_id} already exists in run {run_id} and overwrite_existing_kwargs is False')
            else:
                print(f'Overwriting existing {setup_id} in run {run_id}')
                del run[f'{setup_id}/kwargs']
                run[f'{setup_id}/kwargs'] = stringify_unsupported(kwargs)
        else:
            run[f'{setup_id}/kwargs'] = stringify_unsupported(kwargs)
        
        if is_run_new:
            run[f'{setup_id}/original_kwargs'] = stringify_unsupported(kwargs)

        if setup_is_new:
            run[f'{setup_id}/original_kwargs'] = stringify_unsupported(kwargs)

        local_dir = kwargs.get('local_dir', f'~/output')
        local_dir = os.path.expanduser(local_dir)
        save_dir = f'{local_dir}/{run_id}'
        os.makedirs(save_dir, exist_ok=True)

    except Exception as e:
        run['sys/tag'].add('init failed')
        run.stop()
        raise e

    ####################################
    ##### Load the Data ######
    run_training = kwargs.get('run_training', True)
    run_evaluation = kwargs.get('run_evaluation', True)
    save_latent_space = kwargs.get('save_latent_space', True)
    
    try:
        print('loading data')
        X_filename = kwargs.get('X_filename', 'X_pretrain')
        y_filename = kwargs.get('y_filename', 'y_pretrain')
        # nan_filename = kwargs.get('nan_filename', 'nans')
        train_name = kwargs.get('train_name', 'train')
        eval_name = kwargs.get('eval_name', 'val')
        X_size = None


        if run_training or run_evaluation or True:
            X_data_train = pd.read_csv(f'{data_dir}/{X_filename}_{train_name}.csv', index_col=0)
            y_data_train = pd.read_csv(f'{data_dir}/{y_filename}_{train_name}.csv', index_col=0)
            X_size = X_data_train.shape[1]
        # nan_data = pd.read_csv(f'{data_dir}/{nan_filename}_{train_name}.csv', index_col=0)

        if run_evaluation or save_latent_space:
            X_data_eval = pd.read_csv(f'{data_dir}/{X_filename}_{eval_name}.csv', index_col=0)
            y_data_eval = pd.read_csv(f'{data_dir}/{y_filename}_{eval_name}.csv', index_col=0)
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

        if run_training or run_evaluation or True:
            train_dataset = CompoundDataset(X_data_train,y_data_train[y_head_cols], y_data_train[y_adv_cols])
            eval_dataset = CompoundDataset(X_data_eval,y_data_eval[y_head_cols], y_data_eval[y_adv_cols])

            train_loader_dct = create_dataloaders(train_dataset, batch_size, holdout_frac, set_name=train_name)
            eval_loader_dct = create_dataloaders(eval_dataset, batch_size, set_name = eval_name)
            eval_loader_dct.update(train_loader_dct)

    except Exception as e:
        run['sys/tag'].add('data-load failed')
        run.stop()
        raise e

    ####################################
    ###### Create the Encoder Models ######
    try:
        print('creating models')
        encoder_kind = kwargs.get('encoder_kind', 'AE')
        encoder_kwargs = kwargs.get('encoder_kwargs', {})
        other_input_size = kwargs.get('other_input_size', 1)
        latent_size = encoder_kwargs.get('latent_size', 8)
        input_size = kwargs.get('input_size', None)
        load_model_weights = kwargs.get('load_model_weights', True)
        
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

        if 'hidden_size_mult' in encoder_kwargs:
            encoder_kwargs['hidden_size'] = int(encoder_kwargs['hidden_size_mult']*latent_size)
            # remove the hidden_size_mult key
            encoder_kwargs.pop('hidden_size_mult')
    
        encoder = get_model(encoder_kind, input_size, **encoder_kwargs)


        if (load_encoder_loc) and (load_model_weights):
            os.makedirs(os.path.join(save_dir,load_encoder_loc), exist_ok=True)

            run[f'{load_encoder_loc}/models/encoder_state_dict'].download(f'{save_dir}/{load_encoder_loc}/encoder_state_dict.pth')
            encoder_state_dict = torch.load(f'{save_dir}/{load_encoder_loc}/encoder_state_dict.pth')
            encoder.load_state_dict(encoder_state_dict)

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
            # elif head_kind == 'Regression':
                # head_list.append(Regression_Head(**head_kwargs))
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
        head.update_class_weights(train_dataset.y_head)


        if load_head_loc and load_model_weights:
            head_file_ids = head.get_file_ids()
            for head_file_id in head_file_ids:
                run[f'{load_head_loc}/models/{head_file_id}_state'].download(f'{save_dir}/{load_head_loc}/{head_file_id}_state.pt')
                run[f'{load_head_loc}/models/{head_file_id}_info'].download(f'{save_dir}/{load_head_loc}/{head_file_id}_info.json')
                # head_state_dict = torch.load(f'{save_dir}/{load_head_loc}/{head_file_id}_state.pt')
            head.load_state_from_path(f'{save_dir}/{load_head_loc}')

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
            # elif adv_kind == 'Regression':
                # adv_list.append(Regression_Head(**adv_kwargs))
            elif adv_kind == 'NA':
                adv_list.append(Dummy_Head())
            else:
                raise ValueError(f'Invalid adv_kind: {adv_kind}')

        adv = MultiHead(adv_list)

        y_adv_col_array = np.array(y_adv_cols)
        for a in adv_list:
            cols = y_adv_col_array[a.y_idx]
            print(f'{a.kind} {a.name} uses columns: {cols}')


        adv.update_class_weights(train_dataset.y_adv)

        if load_adv_loc and load_model_weights:
            adv_file_ids = adv.get_file_ids()
            for adv_file_id in adv_file_ids:
                run[f'{load_adv_loc}/models/{adv_file_id}_state'].download(f'{save_dir}/{load_adv_loc}/{adv_file_id}_state.pt')
                run[f'{load_adv_loc}/models/{adv_file_id}_info'].download(f'{save_dir}/{load_adv_loc}/{adv_file_id}_info.json')
                # adv_state_dict = torch.load(f'{save_dir}/{load_adv_loc}/{adv_file_id}_state.pt')
            adv.load_state_from_path(f'{save_dir}/{load_adv_loc}')
            
    except Exception as e:
        run['sys/tag'].add('model-creation failed')
        run.stop()
        raise e

    ####################################
    ###### Train the Models ######
    run_training = kwargs.get('run_training', True)
    run_random_init = kwargs.get('run_random_init', False) # this should be redundant with load_model_weights

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
                encoder.reset_params()
                head.reset_params()
                adv.reset_params()


            train_kwargs = kwargs.get('train_kwargs', {})
            train_kwargs['prefix'] = f'{setup_id}/train'
            encoder, head, adv = train_compound_model(train_loader_dct, 
                                                    encoder, head, adv, 
                                                    run=run, **train_kwargs)

            # log the models
            # run[f'{setup_id}/encoder'] = npt_logger.log_model('encoder')
            # run[f'{setup_id}/head'] = npt_logger.log_model(head)
            # run[f'{setup_id}/adv'] = npt_logger.log_model(adv)

            # alternative way to log models
            torch.save(encoder.state_dict(), f'{save_dir}/{setup_id}_encoder_state_dict.pth')
            # torch.save(head.state_dict(), f'{save_dir}/{setup_id}_head_state_dict.pth')
            # torch.save(adv.state_dict(), f'{save_dir}/{setup_id}_adv_state_dict.pth')

            run[f'{setup_id}/models/encoder_state_dict'].upload(f'{save_dir}/{setup_id}_encoder_state_dict.pth')
            # run[f'{setup_id}/models/head_state_dict'].upload(f'{save_dir}/{setup_id}_head_state_dict.pth')
            # run[f'{setup_id}/models/adv_state_dict'].upload(f'{save_dir}/{setup_id}_adv_state_dict.pth')

            os.makedirs(os.path.join(save_dir,setup_id), exist_ok=True)
            head.save_state_to_path(f'{save_dir}/{setup_id}')
            adv.save_state_to_path(f'{save_dir}/{setup_id}')
            head.save_info(f'{save_dir}/{setup_id}')
            adv.save_info(f'{save_dir}/{setup_id}')

            head_file_ids = head.get_file_ids()
            for head_file_id in head_file_ids:
                run[f'{setup_id}/models/{head_file_id}_state'].upload(f'{save_dir}/{setup_id}/{head_file_id}_state.pt')
                run[f'{setup_id}/models/{head_file_id}_info'].upload(f'{save_dir}/{setup_id}/{head_file_id}_info.json')
            adv_file_ids = adv.get_file_ids()
            for adv_file_id in adv_file_ids:
                run[f'{setup_id}/models/{adv_file_id}_state'].upload(f'{save_dir}/{setup_id}/{adv_file_id}_state.pt')
                run[f'{setup_id}/models/{adv_file_id}_info'].upload(f'{save_dir}/{setup_id}/{adv_file_id}_info.json')

        except Exception as e:
            run['sys/tag'].add('training failed')
            run.stop()
            raise e


    ####################################
    ###### Evaluate the Models ######
    run_evaluation = kwargs.get('run_evaluation', True)

    if run_evaluation:
        try:
            print('evaluating models')
            eval_kwargs = kwargs.get('eval_kwargs', {})
            eval_kwargs['prefix'] = f'{setup_id}/eval'
            evaluate_compound_model(eval_loader_dct, 
                                    encoder, head, adv, 
                                    run=run, **eval_kwargs)
        except Exception as e:
            run['sys/tag'].add('evaluation failed')
            run.stop()
            raise e


    
    ####################################
    ###### Generate the Latent Space ######
    print('generating latent space plots')
    try:
        save_latent_space = kwargs.get('save_latent_space', True)
        Z_embed_savepath = os.path.join(save_dir, f'Z_embed_{eval_name}.csv')

        if save_latent_space:
            
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



        plot_latent_space = kwargs.get('plot_latent_space', '')
        plot_latent_space_cols = kwargs.get('plot_latent_space_cols', [y_head_cols, y_adv_cols])
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

            marker_sz = 5/(1+np.log10(Z_embed.shape[0]))
            if (plot_latent_space=='seaborn') or (plot_latent_space=='both') or (plot_latent_space=='sns'):

                for hue_col in plot_latent_space_cols:
                    palette = get_color_map(Z_embed[hue_col].nunique())

                    ## PCA ##
                    fig = sns.scatterplot(data=Z_embed, x='PCA1', y='PCA2', hue=hue_col, palette=palette,s=marker_sz)
                    # place the legend outside the plot
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    
                    # edit the legend to include the number of samples in each cohort
                    handles, labels = fig.get_legend_handles_labels()
                    labels = [f'{label} ({Z_embed[Z_embed[hue_col]==label].shape[0]})' for label in labels]
                    # make the size of the markers in the handles larger
                    for handle in handles:
                        # print(dir(handle))
                        handle.set_markersize(10)
                        # handle._sizes = [100]
                    
                    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                    plt.savefig(os.path.join(save_dir, f'Z_pca_{hue_col}_{eval_name}.png'), bbox_inches='tight')
                    run[f'{setup_id}/sns_Z_pca_{hue_col}_{eval_name}'].upload(os.path.join(save_dir, f'Z_pca_{hue_col}_{eval_name}.png'))
                    plt.close()

                    ## UMAP ##
                    fig = sns.scatterplot(data=Z_embed, x='UMAP1', y='UMAP2', hue=hue_col, palette=palette,s=marker_sz)
                    # place the legend outside the plot
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                    # edit the legend to include the number of samples in each cohort
                    handles, labels = fig.get_legend_handles_labels()
                    labels = [f'{label} ({Z_embed[Z_embed[hue_col]==label].shape[0]})' for label in labels]
                    # make the size of the markers in the handles larger
                    for handle in handles:
                        # print(dir(handle))
                        handle.set_markersize(10)
                        # handle._sizes = [100]
                    
                    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


                    plt.savefig(os.path.join(save_dir, f'Z_umap_{hue_col}_{eval_name}.png'), bbox_inches='tight')
                    run[f'{setup_id}/sns_Z_umap_{hue_col}_{eval_name}'].upload(os.path.join(save_dir, f'Z_umap_{hue_col}_{eval_name}.png'))
                    plt.close()

            if (plot_latent_space=='plotly') or (plot_latent_space=='both') or (plot_latent_space=='px'):
                for hue_col in plot_latent_space_cols:
                    plotly_fig = px.scatter(Z_embed, x='PCA1', y='PCA2', color=hue_col, title=f'PCA {hue_col}')
                    plotly_fig.update_traces(marker=dict(size=2*marker_sz))
                    run[f'{setup_id}/px_Z_pca_{hue_col}_{eval_name}'].upload(plotly_fig)
                    plt.close()

                    plotly_fig = px.scatter(Z_embed, x='UMAP1', y='UMAP2', color=hue_col)
                    plotly_fig.update_traces(marker=dict(size=2*marker_sz))
                    run[f'{setup_id}/px_Z_umap_{hue_col}_{eval_name}'].upload(plotly_fig)
                    plt.close()

    except Exception as e:
        run['sys/tag'].add('plotting failed')
        run.stop()
        raise e


    run.stop()
    return run_id


# Run an example

## Main

def main():

    from sklearn.linear_model import LogisticRegression


    data_dir = '/DATA2'
    # data_dir = os.path.expanduser(data_dir)

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