# setup a neptune run


import torch
import pandas as pd
import numpy as np
import os
import json
from models import get_model
from train2 import CompoundDataset, train_compound_model, get_end_state_eval_funcs, evaluate_compound_model, create_dataloaders
import neptune
from neptune.utils import stringify_unsupported
from neptune_pytorch import NeptuneLogger
from viz import generate_latent_space, generate_umap_embedding, generate_pca_embedding

import uuid
import matplotlib.pyplot as plt
import seaborn as sns



# set up neptune
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='



def filter_feats(X_data, y_data, nan_data, **kwargs):
    if kwargs is None:
        return X_data.columns.to_list()

    if 'chosen_feats' in kwargs:
        return kwargs['chosen_feats']
    
    if 'chosen_feats_file' in kwargs:
        chosen_feats = pd.read_csv(kwargs['chosen_feats_file'], index_col=0).index.to_list()
        return chosen_feats

    set_col = kwargs.get('assignment_col', 'Set')
    if set_col not in y_data.columns:
        raise ValueError(f'{set_col} not in y_data.columns')
    pretrain_files = y_data[y_data[set_col] == 'Pretrain'].index.to_list() 
    finetune_files = y_data[y_data[set_col] == 'Finetune'].index.to_list()

    finetune_peak_freq_th = kwargs.get('finetune_peak_freq_th', 0)
    pretrain_peak_freq_th = kwargs.get('pretrain_peak_freq_th', 0)
    overall_peak_freq_th = kwargs.get('overall_peak_freq_th', 0)
    finetune_var_q_th = kwargs.get('finetune_var_q_th', 0)
    finetune_var_th = kwargs.get('finetune_var_th', None)

    finetune_peak_freq = 1- nan_data.loc[finetune_files].sum(axis=0)/len(finetune_files)
    pretrain_peak_freq = 1- nan_data.loc[pretrain_files].sum(axis=0)/len(pretrain_files)
    overall_peak_freq = 1- nan_data.sum(axis=0)/nan_data.shape[0]
    finetune_var = X_data.loc[finetune_files].var(axis=0)

    if (finetune_var_th is None) and (finetune_var_q_th > 0):
        finetune_var_th = finetune_var.quantile(finetune_var_q_th)
        print('finetune_var_th:', finetune_var_th)
    elif finetune_var_th is None:
        finetune_var_th = 0   

    peak_filt_df = pd.DataFrame({
            'finetune_peak_freq': finetune_peak_freq,
            'pretrain_peak_freq': pretrain_peak_freq,
            'overall_peak_freq': overall_peak_freq,
            'finetune_var': finetune_var,
        }, index=X_data.columns)
    
    
    chosen_feats = peak_filt_df[  (peak_filt_df['finetune_peak_freq'] >= finetune_peak_freq_th)
                                & (peak_filt_df['pretrain_peak_freq'] >= pretrain_peak_freq_th)
                                & (peak_filt_df['overall_peak_freq'] >= overall_peak_freq_th)
                                & (peak_filt_df['finetune_var'] >= finetune_var_th)
                            ].index.to_list() 

    return chosen_feats
        




def setup_neptune_run(data_dir,setup_id,with_id=None,from_id=None,**kwargs):

    if with_id is None:
        run = neptune.init_run(project='revivemed/RCC',
            api_token=NEPTUNE_API_TOKEN)
    else:
        run = neptune.init_run(project='revivemed/RCC',
            api_token=NEPTUNE_API_TOKEN,
            run=with_id)
        print('Continuing run:', with_id)

    
    run[f'{setup_id}/kwargs'] = stringify_unsupported(kwargs)
    
    run_id = run.id
    local_dir = kwargs.get('local_dir', f'~/output')
    local_dir = os.path.expanduser(local_dir)
    save_dir = f'{local_dir}/{run_id}'
    os.makedirs(save_dir, exist_ok=True)

    ##### Load the Data
    X_filename = kwargs.get('X_filename', 'X_pretrain')
    y_filename = kwargs.get('y_filename', 'y_pretrain')
    # nan_filename = kwargs.get('nan_filename', 'nans')
    train_name = kwargs.get('train_name', 'train')
    eval_name = kwargs.get('eval_name', 'val')

    y_head_col = kwargs.get('y_head_col', 'is Pediatric')
    y_adv_col = kwargs.get('y_adv_col', 'Study ID ENC')

    # X_data = pd.read_csv(f'{data_dir}/{X_filename}.csv', index_col=0)
    # y_data = pd.read_csv(f'{data_dir}/{y_filename}.csv', index_col=0)
    # nan_data = pd.read_csv(f'{data_dir}/{nan_filename}.csv', index_col=0)

    # filter_peaks_kwargs = kwargs.get('filter_peaks_kwargs', {})
    # chosen_feats = filter_feats(X_data, y_data, nan_data, **filter_peaks_kwargs)
    # X_data = X_data[chosen_feats].copy()
    # nan_data = nan_data[chosen_feats].copy()

    # set_col = kwargs.get('set_col', 'Pretrain')
    # train_ids = y_data[y_data[set_col]==train_name].index.to_list()
    # eval_ids = y_data[y_data[set_col]==eval_name].index.to_list()

    # X_data_train = X_data.loc[train_ids]
    # X_data_eval = X_data.loc[eval_ids]
    # y_data_train = y_data.loc[train_ids]
    # y_data_eval = y_data.loc[eval_ids]

    X_data_train = pd.read_csv(f'{data_dir}/{X_filename}_{train_name}.csv', index_col=0)
    y_data_train = pd.read_csv(f'{data_dir}/{y_filename}_{train_name}.csv', index_col=0)
    # nan_data = pd.read_csv(f'{data_dir}/{nan_filename}_{train_name}.csv', index_col=0)

    X_data_eval = pd.read_csv(f'{data_dir}/{X_filename}_{eval_name}.csv', index_col=0)
    y_data_eval = pd.read_csv(f'{data_dir}/{y_filename}_{eval_name}.csv', index_col=0)
    # nan_data = pd.read_csv(f'{data_dir}/{nan_filename}_{eval_name}.csv', index_col=0)



    ##### Create the DataLoaders
    batch_size = kwargs.get('batch_size', 32)
    holdout_frac = kwargs.get('holdout_frac', 0)

    train_dataset = CompoundDataset(X_data_train,y_data_train[y_head_col], y_data_train[y_adv_col])
    eval_dataset = CompoundDataset(X_data_eval,y_data_eval[y_head_col], y_data_eval[y_adv_col])

    train_loader_dct = create_dataloaders(train_dataset, batch_size, holdout_frac, set_name=train_name)
    eval_loader_dct = create_dataloaders(eval_dataset, batch_size, set_name = eval_name)
    eval_loader_dct.update(train_loader_dct)

    ###### Create the Models
    encoder_kind = kwargs.get('encoder_kind', 'AE')
    encoder_kwargs = kwargs.get('encoder_kwargs', {})
    other_input_size = kwargs.get('other_input_size', 1)
    latent_size = encoder_kwargs.get('latent_size', 8)
    input_size = kwargs.get('input_size', X_data_train.shape[1])
    assert input_size == X_data_train.shape[1]

    head_kind = kwargs.get('head_kind', 'NA')
    adv_kind = kwargs.get('adv_kind', 'NA')
    head_kwargs = kwargs.get('head_kwargs', {})
    adv_kwargs = kwargs.get('adv_kwargs', {})

    encoder = get_model(encoder_kind, input_size, **encoder_kwargs)
    head = get_model(head_kind, latent_size+other_input_size, **head_kwargs)
    adv = get_model(adv_kind, latent_size, **adv_kwargs)


    ###### Train the Models
    npt_logger = NeptuneLogger(
        run=run,
        model=encoder,
        log_model_diagram=True,
        log_gradients=True,
        log_parameters=True,
        log_freq=5,
    )


    train_kwargs = kwargs.get('train_kwargs', {})
    encoder, head, adv = train_compound_model(train_loader_dct, 
                                              encoder, head, adv, 
                                              run=run, **train_kwargs)

    # log the models
    run[f'{setup_id}/encoder'] = npt_logger.log_model(encoder)
    run[f'{setup_id}/head'] = npt_logger.log_model(head)
    run[f'{setup_id}/adv'] = npt_logger.log_model(adv)

    # alternative way to log models
    torch.save(encoder.state_dict(), f'{save_dir}/{setup_id}_encoder_state_dict.pth')
    torch.save(head.state_dict(), f'{save_dir}/{setup_id}_head_state_dict.pth')
    torch.save(adv.state_dict(), f'{save_dir}/{setup_id}_adv_state_dict.pth')

    run[f'{setup_id}/encoder_state_dict'].upload(f'{save_dir}/{setup_id}_encoder_state_dict.pth')
    run[f'{setup_id}/head_state_dict'].upload(f'{save_dir}/{setup_id}_head_state_dict.pth')
    run[f'{setup_id}/adv_state_dict'].upload(f'{save_dir}/{setup_id}_adv_state_dict.pth')

    encoder.save(f'{setup_id}_encoder')

    ###### Evaluate the Models
    eval_funcs = get_end_state_eval_funcs()
    eval_kwargs = kwargs.get('eval_kwargs', {})
    evaluate_compound_model(eval_loader_dct, 
                            encoder, head, adv, 
                            run=run, **eval_kwargs)
    



    Z = generate_latent_space(X_data_eval, encoder)
    Z.to_csv(os.path.join(save_dir, f'Z_{eval_name}.csv'))
    Z_umap = generate_umap_embedding(Z)
    Z_umap.to_csv(os.path.join(save_dir, f'Z_umap_{eval_name}.csv'))

    Z_umap.columns = [f'UMAP{i+1}' for i in range(Z_umap.shape[1])]
    Z_umap = Z_umap.join(y_data_eval)
    hue_col = y_head_col
    palette = 'tab10'
    fig = sns.scatterplot(data=Z_umap, x='UMAP1', y='UMAP2', hue=hue_col, palette=palette)
    plt.savefig(os.path.join(save_dir, f'Z_umap_{eval_name}.png'))
    run[f'{setup_id}/Z_umap_{eval_name}'].upload(os.path.join(save_dir, f'Z_umap_{eval_name}.png'))
    run[f'{setup_id}/Z_umap_{eval_name} fig'] = fig
    plt.close()

    run.stop()
    return