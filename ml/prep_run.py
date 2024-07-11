
import neptune
import numpy as np
import optuna
import json
from misc import round_to_sig
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
# from optuna.distributions import json_to_distribution, check_distribution_compatibility, distribution_to_json
from neptune.utils import stringify_unsupported

import os
import pandas as pd
import anndata as ad

# import neptune exceptions
from neptune.exceptions import NeptuneException #, NeptuneServerError



def create_selected_data(input_data_dir, sample_selection_col, 
        metadata_df=None,subdir_col='Study ID',
        output_dir=None, metadata_cols=[], 
        save_nan=False, use_anndata=False):
    """
    Creates selected data based on the given parameters.

    Parameters:
    - input_data_dir (str): The directory path where the input data is located.
    - sample_selection_col (str): The column name in the metadata dataframe used for sample selection.
    - metadata_df (pandas.DataFrame, optional): The metadata dataframe. If not provided, it will be read from 'metadata.csv' in the input_data_dir.
    - subdir_col (str, optional): The column name in the metadata dataframe used for subdirectory identification. Default is 'Study ID'.
    - output_dir (str, optional): The directory path where the output data will be saved. If not provided, it will be created as 'formatted_data' in the input_data_dir.
    - metadata_cols (list, optional): The list of column names to include in the output metadata. If not provided, all columns will be included.
    - save_nan (bool, optional): Whether to save NaN values in the output files. Default is False.
    - use_anndata (bool, optional): Whether to use AnnData format for saving the output data. Default is False.

    Returns:
    - output_dir (str): The directory path where the output data is saved.
    - save_file_id (str): The identifier used in the output file names.

    Raises:
    - ValueError: If the number of lost samples is too many compared to the metadata.

    """

    if output_dir is None:
        output_dir = os.path.join(input_data_dir, 'formatted_data')
        os.makedirs(output_dir, exist_ok=True)

    if metadata_df is None:
        metadata_df = pd.read_csv(f'{input_data_dir}/metadata.csv', index_col=0)
    subdir_list = metadata_df[subdir_col].unique()

    select_ids = metadata_df[metadata_df[sample_selection_col]].index.to_list()
    print(f'Number of samples selected: {len(select_ids)}')

    if len(metadata_cols) == 0:
        metadata_cols = metadata_df.columns.to_list()

    save_file_id = sample_selection_col.replace(' ', '_')
    y_file = f'{output_dir}/y_{save_file_id}.csv'
    if save_nan:
        X_file = f'{output_dir}/nan_{save_file_id}.csv'
    else:
        X_file = f'{output_dir}/X_{save_file_id}.csv'
    h5ad_file = f'{output_dir}/{save_file_id}.h5ad'

    if use_anndata and os.path.exists(h5ad_file):
        print(f'Files already exist at {output_dir}')
        return output_dir
    if not use_anndata and os.path.exists(y_file) and os.path.exists(X_file):
        print(f'Files already exist at {output_dir}')
        return output_dir


    X_list = []
    obs_list = []

    for subdir in subdir_list:
        if save_nan:
            intensity_file = f'{input_data_dir}/{subdir}/nan_matrix.csv'
        else:
            intensity_file = f'{input_data_dir}/{subdir}/scaled_intensity_matrix.csv'

        if os.path.exists(intensity_file):
            select_ids = metadata_df[metadata_df['Study ID'] == subdir].index.to_list()
            subset_select_ids = list(set(select_ids).intersection(subdir))
            if len(subset_select_ids) > 0:
                intensity_df = pd.read_csv(intensity_file, index_col=0)
                intensity_df = intensity_df.loc[subset_select_ids].copy()
                X_list.append(intensity_df)
                obs_list.extend(subset_select_ids)
        else:
            print(f'{subdir} is missing')
            continue

    X = pd.concat(X_list, axis=0)
    obs = metadata_df.loc[obs_list, metadata_cols]

    if len(obs) != X.shape[0]:
        print('Warning, the number of samples in the metadata and intensity matrix do not match')
        print(f'Number of samples in metadata: {len(obs)}')
        print(f'Number of samples in intensity matrix: {X.shape[0]}')
        common_samples = list(set(X.index).intersection(obs.index))
        print(f'Number of common samples: {len(common_samples)}')
        if len(common_samples) < 0.9 * len(obs):
            raise ValueError('The number of lost samples is too many, check the data')
        X = X.loc[common_samples, :].copy()
        obs = obs.loc[common_samples, :].copy()

    if use_anndata:
        adata = ad.AnnData(X=X, obs=obs)
        adata.write_h5ad(h5ad_file)
        print(f'Anndata file saved at {h5ad_file}')
    else:
        obs.to_csv(f'{output_dir}/y_{save_file_id}.csv')
        X.to_csv(f'{output_dir}/X_{save_file_id}.csv')
        print('CSV files saved')

    return output_dir, save_file_id

########################################################################################

def get_task_head_kwargs(head_kind, y_head_col, y_cols=None, head_name=None, num_classes=0):
    """
    Returns the keyword arguments for a task head based on the given parameters.

    Parameters:
        head_kind (str): The kind of task head. Possible values are 'Cox', 'Binary', 'MultiClass', or 'Ordinal'.
        y_head_col (str): The name of the target column for the task head.
        y_cols (list, optional): A list of target columns. Defaults to None.
        head_name (str, optional): The name of the task head. Defaults to None.
        num_classes (int, optional): The number of classes for the task head. Defaults to 0.

    Returns:
        tuple: A tuple containing the task head keyword arguments and the updated list of target columns.

    Raises:
        ValueError: If the given y_head_col is not recognized for the Cox model.
        ValueError: If num_classes is not specified for MultiClass and Ordinal heads.
    """

    if head_name is None:
        head_name = y_head_col

    if y_cols is None:
        y_cols = []

    y_head_cols = [y_head_col]
    if head_kind == 'Cox':
        if 'OS' in y_head_col:
           y_head_cols.append('OS_Event')
        elif 'PFS' in y_head_col:
            y_head_cols.append('PFS_Event')
        else:
            raise ValueError(f"y_head_col {y_head_col} not recognized for Cox model")

    if head_kind == 'Binary':
        num_classes = 2
    elif (head_kind == 'MultiClass') or (head_kind == 'Ordinal'):
        if num_classes == 0:
            raise ValueError("num_classes must be specified for MultiClass and Ordinal heads")


    for col in y_head_cols:
        if col not in y_cols:
            y_cols.append(col)

    if len(y_head_cols) == 1:
        y_idx = y_cols.index(y_head_cols[0])
    else:
        y_idx = [y_cols.index(col) for col in y_head_cols]

    head_kwargs = {
            'kind': head_kind,
            'name': head_name,
            'weight': 1.0,
            'y_idx': y_idx,
            'hidden_size': 4,
            'num_hidden_layers': 1,
            'dropout_rate': 0,
            'activation': 'leakyrelu',
            'use_batch_norm': False,
            'num_classes': num_classes,
            }

    return head_kwargs, y_cols

########################################################################################
# if the variable is single value, then use the single value, if it is None, then use the min,max,step with optuna, if it is a list, then pick from the list

def make_kwargs_set(sig_figs=2,
                    encoder_kind='VAE',
                    activation_func= 'leakyrelu',
                    use_batch_norm= False,
                    head_kwargs_dict={},
                    adv_kwargs_dict={},

                    latent_size=64, latent_size_min=4, latent_size_max=128, latent_size_step=4,
                    hidden_size=96, hidden_size_min=16, hidden_size_max=64, hidden_size_step=16,
                    hidden_size_mult=1.5, hidden_size_mult_min=0, hidden_size_mult_max=2, hidden_size_mult_step=0.5,
                    num_hidden_layers=2, num_hidden_layers_min=1, num_hidden_layers_max=10, num_hidden_layers_step=1,
                    
                    num_attention_heads=2, num_attention_heads_min=1, num_attention_heads_max=5, num_attention_heads_step=1,
                    num_decoder_layers=2, num_decoder_layers_min=1, num_decoder_layers_max=5, num_decoder_layers_step=1,
                    embed_dim=32, embed_dim_min=4, embed_dim_max=64, embed_dim_step=4,
                    decoder_embed_dim=8, decoder_embed_dim_min=4, decoder_embed_dim_max=64, decoder_embed_dim_step=4,
                    default_hidden_fraction=0.2, default_hidden_fraction_min=0, default_hidden_fraction_max=0.5, default_hidden_fraction_step=0.1,

                    dropout_rate=0, dropout_rate_min=0, dropout_rate_max=0.5, dropout_rate_step=0.1,
                    encoder_weight=1.0, encoder_weight_min=0, encoder_weight_max=2, encoder_weight_step=0.5,
                    head_weight=1.0, head_weight_min=0, head_weight_max=2, head_weight_step=0.5,
                    adv_weight=1.0, adv_weight_min=0, adv_weight_max=2, adv_weight_step=0.5,

                    task_head_weight=-1, task_head_weight_min=0, task_head_weight_max=10, task_head_weight_step=0.1,
                    task_adv_weight=-1, task_adv_weight_min=0, task_adv_weight_max=10, task_adv_weight_step=0.1,
                    task_hidden_size=4, task_hidden_size_min=4, task_hidden_size_max=64, task_hidden_size_step=4,
                    task_num_hidden_layers=1, task_num_hidden_layers_min=1, task_num_hidden_layers_max=3, task_num_hidden_layers_step=1,

                    weight_decay=0.0001, weight_decay_min=0, weight_decay_max=0.0005, weight_decay_step=0.00001, weight_decay_log=False,
                    l1_reg_weight=0, l1_reg_weight_min=0, l1_reg_weight_max=0.01, l1_reg_weight_step=0.0001, l1_reg_weight_log=False,
                    l2_reg_weight=0, l2_reg_weight_min=0, l2_reg_weight_max=0.01, l2_reg_weight_step=0.0001, l2_reg_weight_log=False,
                    
                    batch_size=32, batch_size_min=16,batch_size_max=64,batch_size_step=16,
                    noise_factor=0.1, noise_factor_min=0.01, noise_factor_max=0.1, noise_factor_step=0.01,
                    num_epochs=100, num_epochs_min=50, num_epochs_max=300, num_epochs_step=10, num_epochs_log=False,
                    learning_rate=0.0005, learning_rate_min=0.0001, learning_rate_max=0.05, learning_rate_step=None,learning_rate_log=True,
                    early_stopping_patience=0, early_stopping_patience_min=0, early_stopping_patience_max=50, early_stopping_patience_step=5,
                    adversarial_start_epoch=0, adversarial_start_epoch_min=-1, adversarial_start_epoch_max=10, adversarial_start_epoch_step=2,
                    ):




    ##### Encoder Architecture #####
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
        hidden_size = -1
        hidden_size_mult = -1

        if num_attention_heads is None:
            num_attention_heads = IntDistribution(num_attention_heads_min, num_attention_heads_max, step=num_attention_heads_step)
        elif isinstance(num_attention_heads, list):
            num_attention_heads = CategoricalDistribution(num_attention_heads)

        if num_decoder_layers is None:
            num_decoder_layers = IntDistribution(num_decoder_layers_min, num_decoder_layers_max, step=num_decoder_layers_step)
        elif isinstance(num_decoder_layers, list):
            num_decoder_layers = CategoricalDistribution(num_decoder_layers)
        elif num_decoder_layers == -1:
            num_decoder_layers = num_hidden_layers

        if embed_dim is None:
            embed_dim = IntDistribution(embed_dim_min, embed_dim_max, step=embed_dim_step)
        elif isinstance(embed_dim, list):
            embed_dim = CategoricalDistribution(embed_dim)
        
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


    ##### Task Architcture #####
    if (len(head_kwargs_dict) > 0) or (len(adv_kwargs_dict) > 0):
        if task_hidden_size is None:
            task_hidden_size = IntDistribution(task_hidden_size_min, task_hidden_size_max, step=task_hidden_size_step)
        elif isinstance(task_hidden_size, list):
            task_hidden_size = CategoricalDistribution(task_hidden_size)
        
        if task_num_hidden_layers is None:
            task_num_hidden_layers = IntDistribution(task_num_hidden_layers_min, task_num_hidden_layers_max, step=task_num_hidden_layers_step)
        elif isinstance(task_num_hidden_layers, list):
            task_num_hidden_layers = CategoricalDistribution(task_num_hidden_layers)


    ##### Head Tasks #####
    if len(head_kwargs_dict) > 0:
        
        if head_weight is None:
            head_weight = FloatDistribution(head_weight_min, head_weight_max, step=head_weight_step)
        elif isinstance(head_weight, list):
            head_weight = CategoricalDistribution(head_weight)

        for head_kwargs_key in head_kwargs_dict.keys():
            
            head_kwargs_dict[head_kwargs_key]['hidden_size'] = task_hidden_size
            head_kwargs_dict[head_kwargs_key]['num_hidden_layers'] = task_num_hidden_layers

            if head_weight is None:
                head_kwargs_dict[head_kwargs_key].update({'weight': FloatDistribution(task_head_weight_min, task_head_weight_max, step=task_head_weight_step)})
            elif isinstance(head_weight, list):
                head_kwargs_dict[head_kwargs_key].update({'weight': CategoricalDistribution(task_head_weight)})
            elif task_head_weight > 0:
                head_kwargs_dict[head_kwargs_key].update({'weight': task_head_weight})
            # if task_head_weight =0, then do NOT update the task weight
    else:
        head_weight = 0
    
    ##### Adversarial Tasks #####
    if len(adv_kwargs_dict) >0:
        if adv_weight is None:
            adv_weight = FloatDistribution(adv_weight_min, adv_weight_max, step=adv_weight_step)
        elif isinstance(adv_weight, list):
            adv_weight = CategoricalDistribution(adv_weight)

        for adv_kwargs_key in adv_kwargs_dict.keys():
            adv_kwargs_dict[adv_kwargs_key]['hidden_size'] = task_hidden_size
            adv_kwargs_dict[adv_kwargs_key]['num_hidden_layers'] = task_num_hidden_layers

            if adv_weight is None:
                adv_kwargs_dict[adv_kwargs_key].update({'weight': FloatDistribution(task_adv_weight_min, task_adv_weight_max, step=task_adv_weight_step)})
            elif isinstance(adv_weight, list):
                adv_kwargs_dict[adv_kwargs_key].update({'weight': CategoricalDistribution(task_adv_weight)})
            elif task_adv_weight > 0:
                adv_kwargs_dict[adv_kwargs_key].update({'weight': task_adv_weight})
            # if task_adv_weight =0, then do NOT update the task weight
    else:
        adv_weight = 0
            

    ##### Regularization Hyperparams #####
    if weight_decay is None:
        if weight_decay_log:
            weight_decay = FloatDistribution(weight_decay_min, weight_decay_max, log=weight_decay_log)
        else:
            weight_decay = FloatDistribution(weight_decay_min, weight_decay_max, step=weight_decay_step)
    elif isinstance(weight_decay, list):
        weight_decay = CategoricalDistribution(weight_decay)

    if l1_reg_weight is None:
        if l1_reg_weight_log:
            l1_reg_weight = FloatDistribution(l1_reg_weight_min, l1_reg_weight_max, log=l1_reg_weight_log)
        else:
            l1_reg_weight = FloatDistribution(l1_reg_weight_min, l1_reg_weight_max, step=l1_reg_weight_step)
    elif isinstance(l1_reg_weight, list):
        l1_reg_weight = CategoricalDistribution(l1_reg_weight)

    if l2_reg_weight is None:
        if l2_reg_weight_log:
            l2_reg_weight = FloatDistribution(l2_reg_weight_min, l2_reg_weight_max, log=l2_reg_weight_log)
        else:
            l2_reg_weight = FloatDistribution(l2_reg_weight_min, l2_reg_weight_max, step=l2_reg_weight_step)
    elif isinstance(l2_reg_weight, list):
        l2_reg_weight = CategoricalDistribution(l2_reg_weight)

    pass

    ##### Fitting/Training Hyperparams #####

    if batch_size is None:
        batch_size = IntDistribution(batch_size_min, batch_size_max, step=batch_size_step)
    elif isinstance(batch_size, list):
        batch_size = CategoricalDistribution(batch_size)

    if noise_factor is None:
        noise_factor = FloatDistribution(noise_factor_min, noise_factor_max, step=noise_factor_step)
    elif isinstance(noise_factor, list):
        noise_factor = CategoricalDistribution(noise_factor)

    if num_epochs is None:
        if num_epochs_log:
            num_epochs = IntDistribution(num_epochs_min, num_epochs_max, log=num_epochs_log)
        else:
            num_epochs = IntDistribution(num_epochs_min, num_epochs_max, step=num_epochs_step)
    elif isinstance(num_epochs, list):
        num_epochs = CategoricalDistribution(num_epochs)

    if learning_rate is None:
        if learning_rate_log:
            learning_rate = FloatDistribution(learning_rate_min, learning_rate_max, log=learning_rate_log)
        else:
            learning_rate = FloatDistribution(learning_rate_min, learning_rate_max, step=learning_rate_step)
    elif isinstance(learning_rate, list):
        learning_rate = CategoricalDistribution(learning_rate)

    if early_stopping_patience is None:
        early_stopping_patience = IntDistribution(early_stopping_patience_min, early_stopping_patience_max, step=early_stopping_patience_step)
    elif isinstance(early_stopping_patience, list):
        early_stopping_patience = CategoricalDistribution(early_stopping_patience)

    if adversarial_start_epoch is None:
        adversarial_start_epoch = IntDistribution(adversarial_start_epoch_min, adversarial_start_epoch_max, step=adversarial_start_epoch_step)
    elif isinstance(adversarial_start_epoch, list):
        adversarial_start_epoch = CategoricalDistribution(adversarial_start_epoch)


    ##### Format the kwargs #####

    encoder_kwargs = {
            'activation': activation_func,
            'latent_size': latent_size,
            'num_hidden_layers': num_hidden_layers,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm,
            'hidden_size': hidden_size,
            'hidden_size_mult' : hidden_size_mult,
            'num_attention_heads': num_attention_heads,
            'num_decoder_layers': num_decoder_layers,
            'embed_dim': embed_dim,
            'decoder_embed_dim': decoder_embed_dim,
            'default_hidden_fraction': default_hidden_fraction
            }

    fit_kwargs = {
                    'optimizer_name': 'adamw',
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'l1_reg_weight': l1_reg_weight,
                    'l2_reg_weight': l2_reg_weight,
                    'encoder_weight': encoder_weight,
                    'head_weight': head_weight,
                    'adversary_weight': adv_weight,
                    'noise_factor': noise_factor,
                    'early_stopping_patience': early_stopping_patience,
                    'adversarial_start_epoch': adversarial_start_epoch
                }

    kwargs = {
                ################
                ## General ##
                'encoder_kind': encoder_kind,
                'encoder_kwargs': encoder_kwargs,
                'holdout_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
                'batch_size': batch_size,
                'head_kwargs_dict': head_kwargs_dict,
                'adv_kwargs_dict': adv_kwargs_dict,
                'fit_kwargs': fit_kwargs
        }
    
    kwargs = round_kwargs_to_sig(kwargs,sig_figs=sig_figs)
    kwargs = convert_model_kwargs_list_to_dict(kwargs,style=2)

    return kwargs

########################################################################################
########################################################################################
########################################################################################



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