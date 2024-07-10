# useful for debug purposes and general testing, makes kwargs with a VAE encoder using some default values
import os
from prep_study import make_kwargs
from setup3 import setup_neptune_run
from utils_neptune import get_latest_dataset
from sklearn.linear_model import LogisticRegression
import pandas as pd

def get_format_for_model(input_data_dir,sample_selection_col,output_dir=None,metadata_cols=[],save_nan=False):

    if output_dir is None:
        output_dir = os.path.join(input_data_dir,'formatted_data')

    all_metadata = pd.read_csv(f'{input_data_dir}/metadata.csv', index_col=0)
    # all_metadata = pd.read_csv(f'{input_data_dir}/metadata.csv', index_col=0, low_memory=False)
    study_id_list = all_metadata['Study ID'].unique()

    select_ids = all_metadata[all_metadata[sample_selection_col]].index.to_list()
    
    print(f'Number of samples selected: {len(select_ids)}')
    if len(metadata_cols)> 0:
        y = all_metadata.loc[select_ids,metadata_cols].copy()
    else:
        y = all_metadata.loc[select_ids,:].copy()

    save_file_id = sample_selection_col.replace(' ','_')
    y_file = f'{output_dir}/y_{save_file_id}.csv'
    if save_nan:
        X_file = f'{output_dir}/nan_{save_file_id}.csv'
    else:
        X_file = f'{output_dir}/X_{save_file_id}.csv'

    if os.path.exists(X_file):
        print(f'Files already exist at {output_dir}')
        return output_dir
    
    X_list = []

    for study_id in study_id_list:
        # print(study_id)
        if save_nan:
            intensity_file = f'{input_data_dir}/{study_id}/nan_matrix.csv'
        else:
            intensity_file = f'{input_data_dir}/{study_id}/scaled_intensity_matrix.csv'

        if os.path.exists(intensity_file):
            study_ids = all_metadata[all_metadata['Study ID']==study_id].index.to_list()
            subset_select_ids = list(set(select_ids).intersection(study_ids))
            if len(subset_select_ids) >0:
                intensity_df = pd.read_csv(intensity_file, index_col=0)
                intensity_df = intensity_df.loc[subset_select_ids].copy()
                X_list.append(intensity_df)
        else:
            print(f'{study_id} is missing')
            continue

    X = pd.concat(X_list, axis=1)

    if len(y) != X.shape[0]:
        print('Warning, the number of samples in the metadata and intensity matrix do not match')
        common_samples = list(set(X.index).intersection(y.index))
        X = X.loc[common_samples].copy()
        y = y.loc[common_samples].copy()
    else:
        y = y.loc[X.index].copy()


    y.to_csv(y_file)
    X.to_csv(X_file)

    return output_dir

# %% Set up the Data

homedir = os.path.expanduser("~")
input_data_dir = f'{homedir}/INPUT_PRETRAIN_DATA'
os.makedirs(input_data_dir, exist_ok=True)
input_data_dir = get_latest_dataset(data_dir=input_data_dir)


data_dir = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain Discovery Train')
_ = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain Discovery Val')
# _ = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain Discovery')
_ = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain Test')
# _ = get_format_for_model(input_data_dir,sample_selection_col= 'Pretrain All')


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



kwargs['y_head_cols'] = ['Sex','Cohort Label v0', 'Age']
kwargs['y_adv_cols'] = ['Study ID']
kwargs['X_filename']  = 'X_Finetune'
kwargs['y_filename']  = 'y_Finetune'
kwargs['train_name'] = 'Discovery_Train'
kwargs['eval_name']  = 'Discovery_Val'
kwargs['head_kwargs_list'] = []
kwargs['head_kwargs_dict'] = {}
kwargs['adv_kwargs_list'] = []
kwargs['adv_kwargs_dict'] = {}
kwargs['train_kwargs']['train_name'] = 'Discovery_Train'
kwargs['train_kwargs']['encoder_weight'] = 1
kwargs['train_kwargs']['head_weight'] = 1


kwargs['head_kwargs_dict'] = {
                # 'MultiClass_Cohort Label': 
                    #         {
                    #     'kind': 'MultiClass',
                    #     'name': 'Cohort Label',
                    #     'y_idx': 1,
                    #     'weight': 1.0,
                    #     'hidden_size': 4,
                    #     'num_hidden_layers': 1,
                    #     'dropout_rate': 0,
                    #     'activation': 'leakyrelu',
                    #     'use_batch_norm': False,
                    #     'num_classes': 4,
                    # },
                    'Binary_Sex':
                    {
                        'kind': 'Binary',
                        'name': 'Sex',
                        'y_idx': 0,
                        'weight': 2.0,
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
                    }
}


print(kwargs)

run_id = setup_neptune_run(data_dir,
                           setup_id='pretrain',
                           neptune_mode='async',
                        #    neptune_mode='debug',
                           yes_logging = True,
                           tags=['debug'],
                           # load_model_from_run_id=with_id,
                           **kwargs)

