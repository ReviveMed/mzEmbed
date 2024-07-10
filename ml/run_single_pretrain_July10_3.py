# useful for debug purposes and general testing, makes kwargs with a VAE encoder using some default values
import os
from prep_study import make_kwargs
from setup3 import setup_neptune_run
from utils_neptune import get_latest_dataset
from sklearn.linear_model import LogisticRegression
import pandas as pd
import anndata as ad

import os
import pandas as pd
import anndata as ad

def get_format_for_model(input_data_dir, sample_selection_col, output_dir=None, metadata_cols=[], save_nan=False, use_anndata=False):
    if output_dir is None:
        output_dir = os.path.join(input_data_dir, 'formatted_data')
        os.makedirs(output_dir, exist_ok=True)

    all_metadata = pd.read_csv(f'{input_data_dir}/metadata.csv', index_col=0)
    study_id_list = all_metadata['Study ID'].unique()

    select_ids = all_metadata[all_metadata[sample_selection_col]].index.to_list()
    print(f'Number of samples selected: {len(select_ids)}')

    if len(metadata_cols) == 0:
        metadata_cols = all_metadata.columns.to_list()

    save_file_id = sample_selection_col.replace(' ', '_')
    h5ad_file = f'{output_dir}/{save_file_id}.h5ad'

    if use_anndata and os.path.exists(h5ad_file):
        print(f'Files already exist at {output_dir}')
        return output_dir
    if not use_anndata and os.path.exists(f'{output_dir}/y_{save_file_id}.csv') and os.path.exists(f'{output_dir}/X_{save_file_id}.csv'):
        print(f'Files already exist at {output_dir}')
        return output_dir


    X_list = []
    obs_list = []

    for study_id in study_id_list:
        if save_nan:
            intensity_file = f'{input_data_dir}/{study_id}/nan_matrix.csv'
        else:
            intensity_file = f'{input_data_dir}/{study_id}/scaled_intensity_matrix.csv'

        if os.path.exists(intensity_file):
            study_ids = all_metadata[all_metadata['Study ID'] == study_id].index.to_list()
            subset_select_ids = list(set(select_ids).intersection(study_ids))
            if len(subset_select_ids) > 0:
                intensity_df = pd.read_csv(intensity_file, index_col=0)
                intensity_df = intensity_df.loc[subset_select_ids].copy()
                X_list.append(intensity_df)
                obs_list.extend(subset_select_ids)
        else:
            print(f'{study_id} is missing')
            continue

    X = pd.concat(X_list, axis=0)
    obs = all_metadata.loc[obs_list, metadata_cols]

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

    return output_dir

# %% Set up the Data

homedir = os.path.expanduser("~")
input_data_dir = f'{homedir}/INPUT_PRETRAIN_DATA'
os.makedirs(input_data_dir, exist_ok=True)
input_data_dir = get_latest_dataset(data_dir=input_data_dir)


data_dir = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain Discovery Train')
_ = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain Discovery Val')
_ = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain Discovery')
_ = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain Test')
_ = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain All')


data_dir = get_format_for_model(input_data_dir,sample_selection_col= 'Finetune Discovery Train')
_ = get_format_for_model(input_data_dir,sample_selection_col= 'Finetune Discovery Val')
# _ = get_format_for_model(input_data_dir,sample_selection_col= 'Finetune Discovery')
_ = get_format_for_model(input_data_dir,sample_selection_col= 'Finetune Test')
# _ = get_format_for_model(input_data_dir,sample_selection_col= 'Finetune All')


# %% Set up the model

# encoder_kind = 'MA_Encoder_to_FF_Decoder'
encoder_kind = 'VAE'
# encoder_kind = 'metabFoundation'
# kwargs = make_kwargs(encoder_kind=encoder_kind,choose_from_distribution=False)
kwargs = make_kwargs(encoder_kind=encoder_kind,choose_from_distribution=False)

kwargs['run_evaluation'] = True



kwargs['y_head_cols'] = ['Sex','Cohort Label v0', 'Age','BMI','IMDC','OS','OS_Event']
kwargs['y_adv_cols'] = ['Study ID']
kwargs['X_filename_prefix']  = 'X_Pretrain'
kwargs['y_filename_prefix']  = 'y_Pretrain'
kwargs['train_name'] = 'Discovery_Train'
kwargs['eval_name']  = 'Discovery_Val'
kwargs['head_kwargs_list'] = []
kwargs['head_kwargs_dict'] = {}
kwargs['adv_kwargs_list'] = []
kwargs['adv_kwargs_dict'] = {}
kwargs['fit_kwargs']['train_name'] = 'Discovery_Train'
kwargs['fit_kwargs']['encoder_weight'] = 1
kwargs['fit_kwargs']['head_weight'] = 1
kwargs['fit_kwargs']['adversary_weight'] = 0.0
kwargs['fit_kwargs']['num_epochs'] = 200
kwargs['encoder_kwargs']['dropout_rate'] = 0.0
kwargs['encoder_kwargs']['latent_size'] = 64


kwargs['head_kwargs_dict'] = {
                'MultiClass_Cohort Label': 
                    {
                        'kind': 'MultiClass',
                        'name': 'Cohort Label',
                        'y_idx': 1,
                        'weight': 1.0,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
                        'num_classes': 4,
                    },
                    'Binary_Sex':
                    {
                        'kind': 'Binary',
                        'name': 'Sex',
                        'y_idx': 0,
                        'weight': 3.0,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
                        'num_classes': 2,
                    },
                    'Regression_Age':
                    {
                        'kind': 'Regression',
                        'name': 'Age',
                        'y_idx': 2,
                        'weight': 1.0,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
                    },
                    'Regression_BMI':
                    {
                        'kind': 'Regression',
                        'name': 'BMI',
                        'y_idx': 3,
                        'weight': 1.0,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
                    }
}


print(kwargs)

# run_id = setup_neptune_run(data_dir,
#                            setup_id='pretrain',
#                            neptune_mode='async',
#                         #    neptune_mode='debug',
#                            yes_logging = True,
#                            tags=['debug'],
#                            # load_model_from_run_id=with_id,
#                            **kwargs)


# with_id = 'RCC-3168'
# run_id = setup_neptune_run(data_dir,
#                            setup_id='pretrain',
#                            neptune_mode='async',
#                         #    neptune_mode='debug',
#                            yes_logging = True,
#                            tags=['debug'],
#                            with_run_id = with_id,
#                         #    load_model_from_run_id=with_id,
#                            run_training=False,
#                            overwrite_existing_kwargs=True,
#                            plot_latent_space='sns',
#                            load_model_loc='pretrain',
#                            eval_name='All'
#                            )

with_id = 'RCC-3168'
finetune_kwargs = {}
finetune_kwargs['X_filename_prefix']  = 'X_Finetune'
finetune_kwargs['y_filename_prefix']  = 'y_Finetune'
finetune_kwargs['train_name'] = 'Discovery_Train'
finetune_kwargs['eval_name']  = 'Discovery_Val'
finetune_kwargs['num_repeats'] = 5
finetune_kwargs['batch_size'] = 64
finetune_kwargs['yes_clean_batches'] = False
finetune_kwargs['fit_kwargs'] = {}
finetune_kwargs['fit_kwargs']['head_weight'] = 1
finetune_kwargs['fit_kwargs']['encoder_weight'] = 0.0
finetune_kwargs['fit_kwargs']['adversary_weight'] = 0.0
finetune_kwargs['fit_kwargs']['num_epochs'] = 30
finetune_kwargs['fit_kwargs']['learning_rate'] = 0.0001
finetune_kwargs['fit_kwargs']['weight_decay'] = 0.00001
finetune_kwargs['fit_kwargs']['clip_grads_with_norm'] = True
finetune_kwargs['fit_kwargs']['noise_factor'] = 0.1
# finetune_kwargs['encoder_kwargs'] = {'dropout_rate': 0.1}
finetune_kwargs['fit_kwargs']['train_name'] = 'Discovery_Train'
finetune_kwargs['y_head_cols'] = ['Sex','Cohort Label v0', 'Age','BMI','IMDC','OS','OS_Event']
finetune_kwargs['y_adv_cols'] = ['Treatment']
finetune_kwargs['adv_kwargs_list'] = []
finetune_kwargs['adv_kwargs_dict'] = {}
finetune_kwargs['head_kwargs_list'] = []

finetune_kwargs['head_kwargs_dict'] = {
                'Cox_both-OS': 
                    {
                        'kind': 'Cox',
                        'name': 'both-OS',
                        'y_idx': [5,6],
                        'weight': 1.0,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
                    }}


run_id = setup_neptune_run(data_dir,
                           setup_id='finetune both-OS2',
                           neptune_mode='async',
                        #    neptune_mode='debug',
                           yes_logging = True,
                           tags=['debug'],
                           with_run_id = with_id,
                        #    load_model_from_run_id=with_id,
                           overwrite_existing_kwargs=False,
                           restart_run = True,
                           plot_latent_space='sns',
                        #    load_model_loc='pretrain',
                            load_encoder_loc='pretrain',
                           **finetune_kwargs
                           )
