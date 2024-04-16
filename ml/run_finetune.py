


NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune
import os
import pandas as pd
import numpy as np
import optuna
import json
from prep_study import add_runs_to_study, reuse_run, convert_neptune_kwargs, \
    make_kwargs, convert_distributions_to_suggestion, convert_model_kwargs_list_to_dict,\
    round_kwargs_to_sig
from setup2 import setup_neptune_run
from misc import download_data_dir
from utils_neptune import  get_run_id_list, check_neptune_existance, get_latest_dataset, get_run_id_list_from_query
from sklearn.linear_model import LogisticRegression
import time
from neptune.exceptions import NeptuneException
from collections import defaultdict
import shutil
import re

data_dir = get_latest_dataset()


default_sweep_kwargs = {
    'holdout_frac': 0,
    'encoder_kwargs__dropout_rate': 0.2,
    'train_kwargs__num_epochs': 30,
    'train_kwargs__early_stopping_patience': 0,
    'train_kwargs__learning_rate': 0.0005,
    'train_kwargs__l2_reg_weight': 0.0005,
    'train_kwargs__l1_reg_weight': 0.005,
    'train_kwargs__noise_factor': 0.1,
    'train_kwargs__weight_decay': 0,
    'train_kwargs__adversarial_mini_epochs': 5,
}


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
        shutil.copy(f'{data_dir}/X_{file_suffix}.csv',f'{data_dir}/X_{file_suffix}{vnum:.0f}.csv')

    return

def get_head_kwargs_by_desc(desc_str):
    if (desc_str is None) or (desc_str == ''):
        return {}, [], []
    

    if 'mskcc' in desc_str.lower():
        y_head_cols = ['MSKCC BINARY']
        head_name = 'MSKCC'
        head_kind = 'Binary'
        num_classes = 2
        y_idx = 0
        plot_latent_space_cols = ['MSKCC']

    elif 'imdc' in desc_str.lower():
        y_head_cols = ['IMDC BINARY']
        head_name = 'IMDC'
        head_kind = 'Binary'
        num_classes = 2
        y_idx = 0
        plot_latent_space_cols = ['IMDC']

    elif 'both-os' in desc_str.lower():
        y_head_cols = ['OS','OS_Event']
        head_name = 'OS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['OS']   

    elif 'both-pfs' in desc_str.lower():
        y_head_cols = ['PFS','PFS_Event']
        head_name = 'PFS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['PFS']        

    elif 'nivo-os' in desc_str.lower():
        y_head_cols = ['NIVO OS','OS_Event']
        head_name = 'OS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['NIVO OS']   

    elif 'nivo-pfs' in desc_str.lower():
        y_head_cols = ['NIVO PFS','PFS_Event']
        head_name = 'PFS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['NIVO PFS']         

    elif 'ever-os' in desc_str.lower():
        y_head_cols = ['EVER OS','OS_Event']
        head_name = 'OS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['EVER OS']   

    elif 'ever-pfs' in desc_str.lower():
        y_head_cols = ['EVER PFS','PFS_Event']
        head_name = 'PFS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['EVER OS']            
    else:
        raise ValueError('Unknown desc_str:',desc_str)



    head_kwargs = {
            'kind': head_kind,
            'name': head_name,
            'weight': 1,
            'y_idx': y_idx,
            'hidden_size': 4,
            'num_hidden_layers': 0,
            'dropout_rate': 0,
            'activation': 'leakyrelu',
            'use_batch_norm': False,
            'num_classes': num_classes,
            }

    return head_kwargs, y_head_cols, plot_latent_space_cols


############################################################
# Compute Finetune
############################################################

def compute_finetune(run_id,plot_latent_space=False,
                           n_trials=5,desc_str=None,
                           sweep_kwargs=None,eval_name='val2',
                           recompute_plot=False):

    run = neptune.init_run(project='revivemed/RCC',
            api_token=NEPTUNE_API_TOKEN,
            with_id=run_id,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
            mode='read-only')

    run_struc= run.get_structure()
    if 'pretrain' not in run_struc:
        print('No pretrain in run:',run_id)
        run.stop()
        return run_id
    if 'models' not in run_struc['pretrain']:
        print('No saved models in run:',run_id)
        run.stop()
        return run_id
    if 'encoder_state_dict' not in run_struc['pretrain']['models']:
        print('No saved encoder_state_dict in run:',run_id)
        run.stop()
        return run_id

    #############################################################
    ## Plot the latent space
    ############################################################
    kwargs = {}
    if plot_latent_space:
        
        ### ### ###
        ### Need to check if the encoder_kind is TGEM_Encoder
        run = neptune.init_run(project='revivemed/RCC',
                api_token=NEPTUNE_API_TOKEN,
                with_id=run_id,
                capture_stdout=False,
                capture_stderr=False,
                capture_hardware_metrics=False)
                # mode='read-only')

        original_kwargs = run['pretrain/original_kwargs'].fetch()
        # run.stop()
        original_kwargs = convert_neptune_kwargs(original_kwargs)
        encoder_kind = original_kwargs['encoder_kind']
        print('encoder_kind:',encoder_kind)
        ### ### ###

        kwargs['overwrite_existing_kwargs'] = True
        # kwargs['encoder_kind'] = 'AE'
        kwargs['load_model_loc'] = 'pretrain'
        kwargs['run_evaluation'] = False
        kwargs['run_training'] = False
        kwargs['save_latent_space'] = True
        # kwargs['save_latent_space'] = False

        kwargs['plot_latent_space'] = 'sns' #'both'
        kwargs['plot_latent_space_cols'] = ['Study ID','Cohort Label','is Pediatric','is Female', 'Age']

        # run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

        if not (encoder_kind == 'TGEM_Encoder'):
            
            if (not ('sns_Z_umap_is Female_train' in run_struc.keys())) or recompute_plot:
                kwargs['eval_name'] = 'train'
                # run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)
                setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,run=run,**kwargs)
            else:
                print('Already plotted train')

            if (not ('sns_Z_umap_is Female_test' in run_struc.keys())) or recompute_plot:
                kwargs['eval_name'] = 'test'
                # run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)
                setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,run=run,**kwargs)
            else:
                print('Already plotted test')

        if (not ('sns_Z_umap_is Female_val' in run_struc.keys())) or recompute_plot:
            kwargs['eval_name'] = 'val'
            # run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)
            setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,run=run,**kwargs)
        else:
            print('Already plotted val')

        run['sys/failed'] = False
        run.stop()

    ############################################################
    ## Finetune
    ############################################################
    # if n_trials==0:
    #     print('skip finetune')
    #     return run_id
    if desc_str is None:
        print('No desc_str')
        return

    if 'ADV' in desc_str:
        adv_desc_str = desc_str.split('ADV')[1]
        head_desc_str = desc_str.split('ADV')[0]
        adversary_weight = 1
    else:
        adv_desc_str = ''
        head_desc_str = desc_str
        adversary_weight = 0

    y_head_cols = []
    head_kwargs_list = []
    plot_latent_space_cols = []
    
    if 'AND' in head_desc_str:
        head_desc_str_list = head_desc_str.split('AND')
    else:
        head_desc_str_list = [head_desc_str]


    for h_desc in head_desc_str_list:
        head_kwargs, head_cols, plot_latent_space_head_cols = get_head_kwargs_by_desc(h_desc)
        head_kwargs_list.append(head_kwargs)
        y_head_cols += head_cols
        plot_latent_space_cols += plot_latent_space_head_cols


    y_adv_cols = []
    adv_kwargs_list = []

    if 'AND' in adv_desc_str:
        adv_desc_str_list = adv_desc_str.split('AND')
    else:
        adv_desc_str_list = [adv_desc_str]

    for a_desc in adv_desc_str_list:
        adv_kwargs, adv_cols, plot_latent_space_adv_cols = get_head_kwargs_by_desc(a_desc)
        adv_kwargs_list.append(adv_kwargs)
        y_adv_cols += adv_cols
        plot_latent_space_cols += plot_latent_space_adv_cols


    # head_kwargs, head_cols, plot_latent_space_head_cols = get_head_kwargs_by_desc(head_desc_str)
    # y_head_cols = head_cols
    # head_kwargs_list = [head_kwargs]

    # adv_kwargs, adv_cols, plot_latent_space_adv_cols = get_head_kwargs_by_desc(adv_desc_str)
    # adv_kwargs_list = [adv_kwargs]
    # y_adv_cols = adv_cols

    # plot_latent_space_cols = plot_latent_space_head_cols + plot_latent_space_adv_cols


    if eval_name.lower() == 'val':
        train_name = 'train'
    elif eval_name.lower() == 'val2':
        train_name = 'train2'
    elif eval_name.lower() == 'test':
        train_name = 'trainval'


    if sweep_kwargs is None:
        sweep_kwargs = default_sweep_kwargs
    
    for key in default_sweep_kwargs.keys():
        if key not in sweep_kwargs.keys():
            sweep_kwargs[key] = default_sweep_kwargs[key]

    num_epochs = None
    if 'epoch_' in desc_str.lower():
        match = re.search(r'epoch_(\d+)', desc_str.lower())
        if match:
            num_epochs = int(match.group(1))
        

    


    if n_trials>0:    
        kwargs = {}
        kwargs['train_kwargs'] = {}
        kwargs['encoder_kwargs'] = {}

        kwargs['overwrite_existing_kwargs'] = True
        kwargs['load_encoder_loc'] = 'pretrain'
        kwargs['load_model_loc'] = False
        kwargs['X_filename'] = 'X_finetune'
        kwargs['y_filename'] = 'y_finetune'
        kwargs['eval_name'] = eval_name
        kwargs['train_name'] = train_name
        kwargs['run_training'] = True
        kwargs['run_evaluation'] = True
        kwargs['save_latent_space'] = True
        kwargs['plot_latent_space'] = ''
        kwargs['plot_latent_space_cols'] = plot_latent_space_cols
        kwargs['y_head_cols'] = y_head_cols
        kwargs['y_adv_cols'] = y_adv_cols
        kwargs['upload_models_to_neptune'] = False
        kwargs['num_repeats'] = n_trials


        kwargs['head_kwargs_dict'] = {}
        kwargs['adv_kwargs_dict'] = {}
        kwargs['head_kwargs_list'] = head_kwargs_list
        kwargs['adv_kwargs_list'] = adv_kwargs_list

        kwargs['encoder_kwargs']['dropout_rate'] = sweep_kwargs.get('encoder_kwargs__dropout_rate')


        # kwargs['train_kwargs']['num_epochs'] = 20
        kwargs['train_kwargs']['early_stopping_patience'] = sweep_kwargs.get('train_kwargs__early_stopping_patience')
        kwargs['holdout_frac'] = sweep_kwargs.get('holdout_frac')
        kwargs['train_kwargs']['head_weight'] = 1
        kwargs['train_kwargs']['clip_grads_with_norm'] = False
        kwargs['train_kwargs']['encoder_weight'] = 0
        kwargs['train_kwargs']['adversary_weight'] = adversary_weight
        kwargs['train_kwargs']['learning_rate'] = sweep_kwargs.get('train_kwargs__learning_rate')
        # kwargs['train_kwargs']['learning_rate'] = 0.0001
        kwargs['train_kwargs']['l2_reg_weight'] = sweep_kwargs.get('train_kwargs__l2_reg_weight')
        kwargs['train_kwargs']['l1_reg_weight'] = sweep_kwargs.get('train_kwargs__l1_reg_weight')
        kwargs['train_kwargs']['noise_factor'] = sweep_kwargs.get('train_kwargs__noise_factor')
        kwargs['train_kwargs']['weight_decay'] = sweep_kwargs.get('train_kwargs__weight_decay')
        kwargs['train_kwargs']['adversarial_mini_epochs'] = sweep_kwargs.get('train_kwargs__adversarial_mini_epochs')
        kwargs['run_evaluation'] = True
        kwargs['eval_kwargs'] = {}
        kwargs['eval_kwargs']['sklearn_models'] = {}

        if num_epochs is None:
            # if encoder_kind == 'TGEM_Encoder':
            #     kwargs['train_kwargs']['num_epochs'] = 3
            # else:
            kwargs['train_kwargs']['num_epochs'] = sweep_kwargs.get('train_kwargs__num_epochs')
        else:
            kwargs['train_kwargs']['num_epochs'] = num_epochs
        print('num_epochs:',num_epochs)

        kwargs = convert_model_kwargs_list_to_dict(kwargs)

        setup_id = f'{desc_str}_finetune'
        _ = setup_neptune_run(data_dir,setup_id=setup_id,with_run_id=run_id,**kwargs)



        # either one of these settings should run a model with random initialized weights, but do both just in case
        kwargs['run_random_init'] = True
        kwargs['load_model_weights'] = False
        kwargs['save_latent_space'] = False
        
        setup_id = f'{desc_str}_randinit'
        _ = setup_neptune_run(data_dir,setup_id=setup_id,with_run_id=run_id,**kwargs)


    # ############################################################
    # ## Get the Average AUROC for the finetune models
    # ############################################################
    
    # # Save the Average AUC for the finetune models
    # run = neptune.init_run(project='revivemed/RCC',
    #                 api_token=NEPTUNE_API_TOKEN,
    #                 with_id=run_id,
    #                 capture_stdout=False,
    #                 capture_stderr=False,
    #                 capture_hardware_metrics=False)
    


    # for setup_id in ['finetune_'+desc_str,'randinit_'+desc_str]:
    #     print('setup_id:',setup_id)
    #     for key, key_loc in metric_string_dct.items():
    #         key_loc = key_loc.replace('/','_')
    #         print('key:',key_loc)
    #         vals_table = run[f'{setup_id}/history/'+key_loc].fetch_values()
    #         vals = vals_table['value']
    #         # print(vals)
    #         run[f'summary/{setup_id}/{key} avg'] = np.mean(vals)
    #         run[f'summary/{setup_id}/{key} std'] = np.std(vals)
    #         run[f'summary/{setup_id}/{key} count'] = len(vals)
    #         run.wait()



    return run_id

# def main():
###########################################################################
# MAIN
###########################################################################

if __name__ == '__main__':


    # get user from the command line
    import sys
    if len(sys.argv)>1:
        plot_latent_space = int(sys.argv[1])
    else:
        plot_latent_space = 0

    # number of trials/repeats for finetuning
    if len(sys.argv)>2:
        n_trials = int(sys.argv[2])
    else:
        n_trials = 1

    # run_id to finetune, or list of run_ids, or a tag
    if len(sys.argv)>3:
        chosen_id = sys.argv[3]
    else:
        chosen_id = None

    # description string of the finetuning task
    if len(sys.argv)>4:
        chosen_finetune_desc = sys.argv[4]
    else:
        chosen_finetune_desc = None

    # which dataset is used for evaluation
    if len(sys.argv)>5:
        eval_name = sys.argv[5]
    else:
        eval_name = 'val2'


    if chosen_id is None:
        chosen_id = input('Enter Run id: ')
    try:
        chosen_id = int(chosen_id)
        chosen_id = 'RCC-'+str(chosen_id)
    except:
        pass

    if 'RCC' not in chosen_id:
        # tags = [chosen_id]
        # run_id_list = get_run_id_list(tags=tags,encoder_kind='AE')

        query = chosen_id
        run_id_list = get_run_id_list_from_query(query=query,limit=1000)

        # run_id_list = get_run_id_list(tags=tags)
    else:
        if ',' in chosen_id:
            run_id_list = chosen_id.split(',')
        else:
            run_id_list =  [chosen_id]


    if chosen_finetune_desc is not None:
        if ',' in chosen_finetune_desc:
            desc_str_list = chosen_finetune_desc.split(',')
        else:
            desc_str_list = [chosen_finetune_desc]
    else:
        desc_str_list = [None]


    if plot_latent_space<2:
        recompute_plot = False
    else:
        recompute_plot = True
        print('will recompute the plots')
    plot_latent_space =bool(plot_latent_space)

    if len(desc_str_list) > 1 and recompute_plot:
        print('more than one desc_str, do not waste time recomputing plots')
        recompute_plot = False


    redo = False
    update_finetune_data('finetune_val',redo=redo)
    update_finetune_data('finetune_train',redo=redo)
    update_finetune_data('finetune_test',redo=redo)
    update_finetune_data('finetune_trainval',redo=redo)

    already_run = []
    print('Number of runs to finetune:',len(run_id_list))

    # run_id_list = ['RCC-1735']
    for run_id in run_id_list:
        if run_id in already_run:
            continue
        print('run_id:',run_id)
        plot_latent_space0 = plot_latent_space
        for desc_str in desc_str_list:
            try:
                run_id = compute_finetune(run_id,n_trials=n_trials,
                                        plot_latent_space=plot_latent_space0,
                                        desc_str=desc_str,
                                        eval_name=eval_name,
                                        recompute_plot=recompute_plot)
                # do not bother attempting to plot the latent space again
                plot_latent_space0 = False
            except NeptuneException as e:
                print('NeptuneException:',e)
                continue
