from setup2 import setup_neptune_run
from utils_neptune import  get_run_id_list, check_neptune_existance, get_latest_dataset
import pandas as pd
import numpy as np
import shutil
import os
data_dir = get_latest_dataset()
setup_id = 'surv_finetune_5'
kwargs = {}
# run_id = 'RCC-2027'
# run_id = 'RCC-2390'
run_id = 'RCC-2502'


def update_finetune_data(file_suffix,redo=False):
    vnum = 2
    y_savefile = f'{data_dir}/y_{file_suffix}{vnum:.0f}.csv'
    if redo and os.path.exists(y_savefile):
        os.remove(y_savefile)

    if not os.path.exists(y_savefile):
        y_val_data = pd.read_csv(f'{data_dir}/y_{file_suffix}.csv')
        y_val_data['NIVO OS'] = np.nan
        y_val_data.loc[y_val_data['Treatment']=='NIVOLUMAB', 'NIVO OS'] = y_val_data.loc[y_val_data['Treatment']=='NIVOLUMAB', 'OS']
        y_val_data['EVER OS'] = np.nan
        y_val_data.loc[y_val_data['Treatment']=='EVEROLIMUS', 'EVER OS'] = y_val_data.loc[y_val_data['Treatment']=='EVEROLIMUS', 'OS']
        
        y_val_data['NIVO PFS'] = np.nan
        y_val_data.loc[y_val_data['Treatment']=='NIVOLUMAB', 'NIVO PFS'] = y_val_data.loc[y_val_data['Treatment']=='NIVOLUMAB', 'PFS']
        y_val_data['EVER PFS'] = np.nan
        y_val_data.loc[y_val_data['Treatment']=='EVEROLIMUS', 'EVER PFS'] = y_val_data.loc[y_val_data['Treatment']=='EVEROLIMUS', 'PFS']
        
        y_val_data['Treatment is NIVO'] = 0
        y_val_data.loc[y_val_data['Treatment']=='NIVOLUMAB', 'Treatment is NIVO'] = 1
        y_val_data.to_csv(y_savefile,index=False)
        shutil.copy(f'{data_dir}/X_finetune_val.csv',f'{data_dir}/X_{file_suffix}{vnum:.0f}.csv')

    return

redo = False
update_finetune_data('finetune_val',redo=redo)
update_finetune_data('finetune_train',redo=redo)
update_finetune_data('finetune_test',redo=redo)
update_finetune_data('finetune_trainval',redo=redo)



encoder_kwargs = {
            'activation': 'leaky_relu',
            'latent_size': 16,
            'num_hidden_layers': 2,
            'dropout_rate': 0.2,
            'use_batch_norm': False,
            # 'hidden_size': int(1.5*latent_size),
            'hidden_size_mult' : 1.5
            }

kwargs['encoder_kind'] = 'AE'
kwargs['encoder_kwargs'] = encoder_kwargs
kwargs['overwrite_existing_kwargs'] = True
kwargs['load_encoder_loc'] = False
# kwargs['load_encoder_loc'] = 'pretrain'
kwargs['load_model_loc'] = False
kwargs['X_filename'] = 'X_finetune'
kwargs['y_filename'] = 'y_finetune'
kwargs['eval_name'] = 'val2'
kwargs['train_name'] = 'train2'
kwargs['run_training'] = True
kwargs['run_evaluation'] = True
kwargs['save_latent_space'] = False
kwargs['plot_latent_space'] = ''
# kwargs['plot_latent_space_cols'] = ['OS']
# kwargs['y_head_cols'] = ['OS','OS_Event']
kwargs['plot_latent_space_cols'] = ['OS']
# kwargs['y_head_cols'] = ['OS','OS_Event']
# kwargs['y_head_cols'] = ['PFS','PFS_Event']
# kwargs['y_adv_cols'] = []
kwargs['y_head_cols'] = ['NIVO OS','OS_Event']
kwargs['y_adv_cols'] = ['EVER OS','OS_Event']
kwargs['upload_models_to_neptune'] = True


kwargs['head_kwargs_dict'] = {}
kwargs['adv_kwargs_dict'] = {}
kwargs['head_kwargs_list'] = [{
    'kind': 'Cox',
    'name': 'OS',
    'weight': 1,
    'y_idx': [0,1],
    'hidden_size': 4,
    'num_hidden_layers': 1,
    'dropout_rate': 0,
    'activation': 'leakyrelu',
    'use_batch_norm': False
    }]


# kwargs['adv_kwargs_list'] = []
kwargs['adv_kwargs_list'] = [{
    'kind': 'Cox',
    'name': 'OS',
    'weight': 1,
    'y_idx': [0,1],
    'hidden_size': 4,
    'num_hidden_layers': 1,
    'dropout_rate': 0,
    'activation': 'leakyrelu',
    'use_batch_norm': False
    }]

kwargs['train_kwargs'] = {}
kwargs['train_kwargs']['num_epochs'] = 80
kwargs['train_kwargs']['early_stopping_patience'] = 0
kwargs['holdout_frac'] = 0
kwargs['train_kwargs']['head_weight'] = 1
kwargs['train_kwargs']['encoder_weight'] = 0
kwargs['train_kwargs']['adversary_weight'] = 1
kwargs['train_kwargs']['learning_rate'] = 0.001
# kwargs['train_kwargs']['learning_rate'] = 0.0001
kwargs['train_kwargs']['l2_reg_weight'] = 0.0005
kwargs['train_kwargs']['l1_reg_weight'] = 0.0005
kwargs['train_kwargs']['noise_factor'] = 0.05
kwargs['train_kwargs']['weight_decay'] = 0
kwargs['run_evaluation'] = True
kwargs['eval_kwargs'] = {}
kwargs['eval_kwargs']['sklearn_models'] = {}
kwargs['num_repeats'] = 4

setup_neptune_run(data_dir,with_run_id=run_id,setup_id=setup_id,**kwargs) #,neptune_mode='debug'
# run_id = setup_neptune_run(data_dir,setup_id=setup_id,**kwargs)
# print(run_id)