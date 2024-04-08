


NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune
import os
import numpy as np
import optuna
import json
from prep_study import add_runs_to_study, reuse_run, convert_neptune_kwargs, \
    objective_func1, make_kwargs, convert_distributions_to_suggestion, convert_model_kwargs_list_to_dict,\
    round_kwargs_to_sig
from setup2 import setup_neptune_run
from misc import download_data_dir
from utils_neptune import  get_run_id_list, check_neptune_existance, get_latest_dataset
from sklearn.linear_model import LogisticRegression
import time
from neptune.exceptions import NeptuneException
from collections import defaultdict
import shutil
import re

data_dir = get_latest_dataset()


def compute_finetune(run_id,plot_latent_space=False,
                           n_trials=3,desc_str='finetune'):



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
                capture_hardware_metrics=False,
                mode='read-only')

        run_struc= run.get_structure()
        original_kwargs = run['pretrain/original_kwargs'].fetch()
        run.stop()
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
        kwargs['plot_latent_space_cols'] = ['Study ID','Cohort Label','is Pediatric','is Female']

        # run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

        if not (encoder_kind == 'TGEM_Encoder'):
            
            if not ('sns_Z_umap_is Female_train' in run_struc.keys()):
                kwargs['eval_name'] = 'train'
                run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)
            else:
                print('Already plotted train')

            if not ('sns_Z_umap_is Female_test' in run_struc.keys()):
                kwargs['eval_name'] = 'test'
                run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)
            else:
                print('Already plotted test')

        if not ('sns_Z_umap_is Female_val' in run_struc.keys()):
            kwargs['eval_name'] = 'val'
            run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)
        else:
            print('Already plotted val')

    ############################################################
    ## Finetune
    ############################################################
    # if n_trials==0:
    #     print('skip finetune')
    #     return run_id

    if 'mskcc' in desc_str.lower():
        metric_string_dct = {}
        count_check_name = 'MSKCC Train AUROC count'
        y_head_cols = ['MSKCC BINARY']
        head_name = 'MSKCC'
        head_kind = 'Binary'
        num_classes = 2
        plot_latent_space_cols = ['MSKCC']
        metric_string_dct['MSKCC Train AUROC'] = 'eval/train/Binary_MSKCC/AUROC (micro)'
        metric_string_dct['MSKCC Val AUROC'] = 'eval/val/Binary_MSKCC/AUROC (micro)'

    elif 'imdc' in desc_str.lower():
        metric_string_dct = {}
        count_check_name = 'IMDC Train AUROC count'
        y_head_cols = ['IMDC BINARY']
        head_name = 'IMDC'
        head_kind = 'Binary'
        num_classes = 2
        plot_latent_space_cols = ['IMDC']
        metric_string_dct['IMDC Train AUROC'] = 'eval/train/Binary_IMDC/AUROC (micro)'
        metric_string_dct['IMDC Val AUROC'] = 'eval/val/Binary_IMDC/AUROC (micro)'


    num_epochs = None
    if 'epoch_' in desc_str.lower():
        match = re.search(r'epoch_(\d+)', desc_str.lower())
        if match:
            num_epochs = int(match.group(1))


            

    else:
        raise ValueError('Unknown desc_str:',desc_str)

    if n_trials>0:
        run = neptune.init_run(project='revivemed/RCC',
                        api_token=NEPTUNE_API_TOKEN,
                        with_id=run_id,
                        capture_stdout=False,
                        capture_stderr=False,
                        capture_hardware_metrics=False,
                        mode='read-only')


        run_struc= run.get_structure()
        # if 'summary/randinit_Apr04.1_MSKCC/MSKCC Train AUROC avg':
        if 'summary' in run_struc.keys():
            if f'randinit_{desc_str}' in run_struc['summary'].keys():
                if count_check_name in run_struc['summary'][f'randinit_{desc_str}'].keys():

                    run_counts = run[f'summary/randinit_{desc_str}/{count_check_name}'].fetch()
                    if run_counts > n_trials:
                        print('Already computed over {} trials'.format(run_counts))
                        run.stop()
                        return run_id
                    
                    n_trials = n_trials - run_counts
                    print('computing {} more trials'.format(n_trials))


        kwargs = run['pretrain/original_kwargs'].fetch()
        kwargs = convert_neptune_kwargs(kwargs)
        encoder_kind = kwargs['encoder_kind']
        run.stop()
        
    if n_trials>0:    
        kwargs['overwrite_existing_kwargs'] = True
        kwargs['load_encoder_loc'] = 'pretrain'
        kwargs['load_model_loc'] = False
        kwargs['X_filename'] = 'X_finetune'
        kwargs['y_filename'] = 'y_finetune'
        kwargs['eval_name'] = 'val'
        kwargs['run_training'] = True
        kwargs['run_evaluation'] = True
        kwargs['save_latent_space'] = False
        kwargs['plot_latent_space'] = ''
        kwargs['plot_latent_space_cols'] = plot_latent_space_cols
        kwargs['y_head_cols'] = y_head_cols
        kwargs['y_adv_cols'] = []
        kwargs['upload_models_to_neptune'] = False


        kwargs['head_kwargs_dict'] = {}
        kwargs['adv_kwargs_dict'] = {}
        kwargs['head_kwargs_list'] = [{
            'kind': head_kind,
            'name': head_name,
            'weight': 1,
            'y_idx': 0,
            'hidden_size': 4,
            'num_hidden_layers': 0,
            'dropout_rate': 0,
            'activation': 'leakyrelu',
            'use_batch_norm': False,
            'num_classes': num_classes,
            }]
        
        kwargs['encoder_kwargs']['dropout_rate'] = 0.2
        kwargs['adv_kwargs_list'] = []

        # kwargs['train_kwargs']['num_epochs'] = 20
        kwargs['train_kwargs']['early_stopping_patience'] = 0
        kwargs['holdout_frac'] = 0
        kwargs['train_kwargs']['head_weight'] = 1
        kwargs['train_kwargs']['encoder_weight'] = 0
        kwargs['train_kwargs']['adversary_weight'] = 0
        kwargs['train_kwargs']['learning_rate'] = 0.0005
        # kwargs['train_kwargs']['learning_rate'] = 0.0001
        kwargs['train_kwargs']['l2_reg_weight'] = 0.0005
        kwargs['train_kwargs']['l1_reg_weight'] = 0.005
        kwargs['train_kwargs']['noise_factor'] = 0.1
        kwargs['train_kwargs']['weight_decay'] = 0
        kwargs['run_evaluation'] = True
        kwargs['eval_kwargs'] = {}
        kwargs['eval_kwargs']['sklearn_models'] = {}

        if num_epochs is None:
            if encoder_kind == 'TGEM_Encoder':
                kwargs['train_kwargs']['num_epochs'] = 3
            else:
                kwargs['train_kwargs']['num_epochs'] = 30
        else:
            kwargs['train_kwargs']['num_epochs'] = num_epochs
            print('num_epochs:',num_epochs)

        kwargs = convert_model_kwargs_list_to_dict(kwargs)

        setup_id = f'finetune_{desc_str}'
        print('ID:',setup_id)
        for ii in range(n_trials):
            print('trial #:',ii)
            _ = setup_neptune_run(data_dir,setup_id=setup_id,with_run_id=run_id,**kwargs)



        # either one of these settings should run a model with random initialized weights, but do both just in case
        kwargs['run_random_init'] = True
        kwargs['load_model_weights'] = False
        
        setup_id = f'randinit_{desc_str}'
        print('ID:',setup_id)
        for ii in range(n_trials):
            print('trial #:',ii)
            _ = setup_neptune_run(data_dir,setup_id=setup_id,with_run_id=run_id,**kwargs)


    ############################################################
    ## Get the Average AUROC for the finetune models
    ############################################################
    
    # Save the Average AUC for the finetune models
    run = neptune.init_run(project='revivemed/RCC',
                    api_token=NEPTUNE_API_TOKEN,
                    with_id=run_id,
                    capture_stdout=False,
                    capture_stderr=False,
                    capture_hardware_metrics=False)
    


    for setup_id in ['finetune_'+desc_str,'randinit_'+desc_str]:
        print('setup_id:',setup_id)
        for key, key_loc in metric_string_dct.items():
            key_loc = key_loc.replace('/','_')
            print('key:',key_loc)
            vals_table = run[f'{setup_id}/history/'+key_loc].fetch_values()
            vals = vals_table['value']
            # print(vals)
            run[f'summary/{setup_id}/{key} avg'] = np.mean(vals)
            run[f'summary/{setup_id}/{key} std'] = np.std(vals)
            run[f'summary/{setup_id}/{key} count'] = len(vals)
            run.wait()


    run['sys/failed'] = False
    run.stop()

    return run_id

# def main():

if __name__ == '__main__':


    # get user from the command line
    import sys
    if len(sys.argv)>1:
        plot_latent_space = bool(int(sys.argv[1]))
    else:
        plot_latent_space = False

    if len(sys.argv)>2:
        n_trials = int(sys.argv[2])
    else:
        n_trials = 1


    if len(sys.argv)>3:
        chosen_id = sys.argv[3]
    else:
        chosen_id = None

    if len(sys.argv)>4:
        desc_str = sys.argv[4]
    else:
        desc_str = 'Apr04.1_MSKCC'

    if chosen_id is None:
        chosen_id = input('Enter Run id: ')

    try:
        chosen_id = int(chosen_id)
        chosen_id = 'RCC-'+str(chosen_id)
    except:
        pass

    if 'RCC' not in chosen_id:
        tags = [chosen_id]
        run_id_list = get_run_id_list(tags=tags,encoder_kind='AE')
        # run_id_list = get_run_id_list(tags=tags)
    else:
        if ',' in chosen_id:
            run_id_list = chosen_id.split(',')
        else:
            run_id_list =  [chosen_id]


    already_run = []
    print('Number of runs to finetune:',len(run_id_list))

    # run_id_list = ['RCC-1735']
    for run_id in run_id_list:
        if run_id in already_run:
            continue
        print('run_id:',run_id)
        try:
            run_id = compute_finetune(run_id,n_trials=n_trials,plot_latent_space=plot_latent_space,desc_str=desc_str)
        except NeptuneException as e:
            print('NeptuneException:',e)
            continue
