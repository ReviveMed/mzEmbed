


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
from setup3 import setup_neptune_run
from misc import download_data_dir
from utils_neptune import  get_run_id_list, check_neptune_existance, get_latest_dataset, get_run_id_list_from_query
from sklearn.linear_model import LogisticRegression
import time
from neptune.exceptions import NeptuneException
from collections import defaultdict
import shutil
import re
from misc import update_finetune_data


default_sweep_kwargs = {
    'holdout_frac': 0,
    'head_hidden_layers': 0,
    'encoder_kwargs__dropout_rate': 0.2,
    'train_kwargs__num_epochs': 30,
    'train_kwargs__early_stopping_patience': 0,
    'train_kwargs__learning_rate': 0.0005,
    'train_kwargs__l2_reg_weight': 0.0005,
    'train_kwargs__l1_reg_weight': 0.005,
    'train_kwargs__noise_factor': 0.1,
    'train_kwargs__weight_decay': 0,
    'train_kwargs__adversarial_mini_epochs': 5,
    'train_kwargs__adversary_weight': 1,
    'train_kwargs__adversarial_start_epoch': 10,
    'train_kwargs__encoder_weight': 0,
    'train_kwargs__clip_grads_with_norm': False,
}



def get_head_kwargs_by_desc(desc_str,num_hidden_layers=0,weight=1,y_cols=None):
    if (desc_str is None) or (desc_str == ''):
        return None, [], []
    
    if y_cols is None:
        y_cols = []

    if 'weight-' in desc_str:
        match = re.search(r'weight-(\d+)', desc_str)
        if match:
            weight = int(match.group(1))
            desc_str = desc_str.replace(match.group(0),'')

    if 'mskcc' in desc_str.lower():
        # if 'mskcc-ord' in desc_str.lower:
            # raise NotImplementedError('not implemented yet')
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

    elif 'nivo-benefit' in desc_str.lower():
        raise NotImplementedError()

    elif 'benefit' in desc_str.lower():
        y_head_cols = ['Benefit BINARY']
        head_name = 'Benefit'
        head_kind = 'Binary'
        num_classes = 2
        y_idx = 0
        plot_latent_space_cols = ['Benefit']

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
        head_name = 'NIVO OS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['NIVO OS']   

    elif 'nivo-pfs' in desc_str.lower():
        y_head_cols = ['NIVO PFS','PFS_Event']
        head_name = 'NIVO PFS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['NIVO PFS']         

    elif 'ever-os' in desc_str.lower():
        y_head_cols = ['EVER OS','OS_Event']
        head_name = 'EVER OS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['EVER OS']   

    elif 'ever-pfs' in desc_str.lower():
        y_head_cols = ['EVER PFS','PFS_Event']
        head_name = 'EVER PFS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['EVER OS']            
    else:
        raise ValueError('Unknown desc_str:',desc_str)

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
            'weight': weight,
            'y_idx': y_idx,
            'hidden_size': 4,
            'num_hidden_layers': num_hidden_layers,
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
                           n_trials=5,
                           desc_str='skip',
                           data_dir=None,
                           sweep_kwargs=None,
                           eval_name='val2',
                           train_name = None,
                           recompute_plot=False,
                           pretrain_eval_on_test=False,
                           finetune_eval_on_test=False,
                           skip_random_init=False,
                           other_kwargs=None):

    if data_dir is None:
        data_dir = get_latest_dataset()

    run = neptune.init_run(project='revivemed/RCC',
            api_token=NEPTUNE_API_TOKEN,
            with_id=run_id,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
            mode='read-only')

    
    run_struc= run.get_structure()
    if 'info/state' in run_struc.keys():
        if run['info/state'].fetch() == 'Active':
            print('Run is already active')
            return
        
        # # temporary
        # print('temporarily skip runs that have info/state')
        # return
    

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
        run["info/state"] = 'Inactive'
        run.stop()

    if pretrain_eval_on_test:

        run = neptune.init_run(project='revivemed/RCC',
                api_token=NEPTUNE_API_TOKEN,
                with_id=run_id,
                capture_stdout=False,
                capture_stderr=False,
                capture_hardware_metrics=False)

        kwargs['overwrite_existing_kwargs'] = True
        # kwargs['encoder_kind'] = 'AE'
        kwargs['load_model_loc'] = 'pretrain'
        kwargs['run_evaluation'] = True
        kwargs['run_training'] = False
        kwargs['save_latent_space'] = False
        # kwargs['save_latent_space'] = False
        kwargs['eval_name'] = 'test'
        # run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)
        setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,run=run,**kwargs)

        run['sys/failed'] = False
        run["info/state"] = 'Inactive'
        run.stop()

    ############################################################
    ## Finetune
    ############################################################
    # if n_trials==0:
    #     print('skip finetune')
    #     return run_id
    
    if sweep_kwargs is None:
        sweep_kwargs = default_sweep_kwargs
    
    for key in default_sweep_kwargs.keys():
        if key not in sweep_kwargs.keys():
            sweep_kwargs[key] = default_sweep_kwargs[key]
    
    if desc_str == 'skip':
        print('skip finetune')
        return

    if desc_str is None:
        print('No desc_str')
        return


    clean_desc_str = desc_str
    if 'optuna_' in desc_str:
        clean_desc_str = clean_desc_str.replace('optuna_','')
    if 'Optimized_' in clean_desc_str:
        clean_desc_str = clean_desc_str.replace('Optimized_','')
    if '__' in clean_desc_str:
        clean_desc_str = clean_desc_str.split('__')[0]

    if 'ADV' in clean_desc_str:
        adv_desc_str = clean_desc_str.split('ADV')[1]
        head_desc_str = clean_desc_str.split('ADV')[0]
        adversary_weight = sweep_kwargs.get('train_kwargs__adversary_weight')
    else:
        adv_desc_str = ''
        head_desc_str = clean_desc_str
        adversary_weight = 0

    y_head_cols = []
    head_kwargs_list = []
    plot_latent_space_cols = []
    
    if 'AND' in head_desc_str:
        head_desc_str_list = head_desc_str.split('AND')
    else:
        head_desc_str_list = [head_desc_str]


    head_hidden_layers = sweep_kwargs.get('head_hidden_layers',0)

    for h_desc in head_desc_str_list:
        head_weight = sweep_kwargs.get(f'{h_desc}__weight',1)
        head_kwargs, head_cols, plot_latent_space_head_cols = get_head_kwargs_by_desc(h_desc,
                                                                                    num_hidden_layers=head_hidden_layers,
                                                                                    weight=head_weight,y_cols=y_head_cols)
        if head_kwargs is None:
            continue
        head_kwargs_list.append(head_kwargs)
        for col in head_cols:
            if col not in y_head_cols:
                y_head_cols.append(col)

        for col in plot_latent_space_head_cols:
            if col not in plot_latent_space_cols:
                plot_latent_space_cols.append(col)



    y_adv_cols = []
    adv_kwargs_list = []

    if 'AND' in adv_desc_str:
        adv_desc_str_list = adv_desc_str.split('AND')
    else:
        adv_desc_str_list = [adv_desc_str]

    for a_desc in adv_desc_str_list:
        adv_weight = sweep_kwargs.get(f'{a_desc}__weight',1)
        adv_kwargs, adv_cols, plot_latent_space_adv_cols = get_head_kwargs_by_desc(a_desc,
                                                                                   num_hidden_layers=head_hidden_layers,
                                                                                    weight=adv_weight,y_cols=y_adv_cols)
        if adv_kwargs is None:
            continue
        adv_kwargs_list.append(adv_kwargs)
        for col in adv_cols:
            if col not in y_adv_cols:
                y_adv_cols.append(col)
        for col in plot_latent_space_adv_cols:
            if col not in plot_latent_space_cols:
                plot_latent_space_cols.append(col)


    # head_kwargs, head_cols, plot_latent_space_head_cols = get_head_kwargs_by_desc(head_desc_str)
    # y_head_cols = head_cols
    # head_kwargs_list = [head_kwargs]

    # adv_kwargs, adv_cols, plot_latent_space_adv_cols = get_head_kwargs_by_desc(adv_desc_str)
    # adv_kwargs_list = [adv_kwargs]
    # y_adv_cols = adv_cols

    # plot_latent_space_cols = plot_latent_space_head_cols + plot_latent_space_adv_cols


    if eval_name.lower() == 'val':
        if train_name is None:
            train_name = 'train'
    elif eval_name.lower() == 'val2':
        if train_name is None:
            train_name = 'train2'
    elif eval_name.lower() == 'test':
        if train_name is None:
            train_name = 'trainval'
    elif eval_name.lower() == 'test2':
        if train_name is None:
            train_name = 'trainval2'


    num_epochs = None
    # if 'epoch_' in desc_str.lower():
    #     match = re.search(r'epoch_(\d+)', desc_str.lower())
    #     if match:
    #         num_epochs = int(match.group(1))
        


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
        kwargs['encoder_kwargs']['default_hidden_fraction'] = 0.0


        # kwargs['train_kwargs']['num_epochs'] = 20
        kwargs['train_kwargs']['early_stopping_patience'] = sweep_kwargs.get('train_kwargs__early_stopping_patience')
        kwargs['holdout_frac'] = sweep_kwargs.get('holdout_frac')
        kwargs['train_kwargs']['head_weight'] = 1
        kwargs['train_kwargs']['clip_grads_with_norm'] = sweep_kwargs.get('train_kwargs__clip_grads_with_norm')
        kwargs['train_kwargs']['encoder_weight'] = sweep_kwargs.get('train_kwargs__encoder_weight')
        kwargs['train_kwargs']['adversary_weight'] = adversary_weight
        kwargs['train_kwargs']['learning_rate'] = sweep_kwargs.get('train_kwargs__learning_rate')
        # kwargs['train_kwargs']['learning_rate'] = 0.0001
        kwargs['train_kwargs']['l2_reg_weight'] = sweep_kwargs.get('train_kwargs__l2_reg_weight')
        kwargs['train_kwargs']['l1_reg_weight'] = sweep_kwargs.get('train_kwargs__l1_reg_weight')
        kwargs['train_kwargs']['noise_factor'] = sweep_kwargs.get('train_kwargs__noise_factor')
        kwargs['train_kwargs']['weight_decay'] = sweep_kwargs.get('train_kwargs__weight_decay')
        # kwargs['train_kwargs']['adversarial_mini_epochs'] = sweep_kwargs.get('train_kwargs__adversarial_mini_epochs')
        kwargs['train_kwargs']['adversarial_start_epoch'] = sweep_kwargs.get('train_kwargs__adversarial_start_epoch')
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

        if (other_kwargs is not None) and (len(other_kwargs)>0):
            if isinstance(other_kwargs,dict):
                existing_kwargs = kwargs.keys()
                print('add other_kwargs to kwargs')
                for key in other_kwargs.keys():
                    if key in existing_kwargs:
                        print(f'WARNING: key {key} already exists in kwargs')
                        # continue
                    kwargs[key] = other_kwargs[key]

        setup_id = f'{desc_str}_finetune'
        _ = setup_neptune_run(data_dir,setup_id=setup_id,with_run_id=run_id,**kwargs)


        if not skip_random_init:
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
        chosen_finetune_desc = ''

    # which dataset is used for finetune evaluation
    if len(sys.argv)>5:
        eval_name = sys.argv[5]
    else:
        eval_name = 'val'


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
        run_id_list = get_run_id_list_from_query(query=query,limit=3000)

        # run_id_list = get_run_id_list(tags=tags)
    else:
        if ',' in chosen_id:
            run_id_list = chosen_id.split(',')
        else:
            run_id_list =  [chosen_id]


    if len(chosen_finetune_desc)>0:
        if ',' in chosen_finetune_desc:
            desc_str_list = chosen_finetune_desc.split(',')
        else:
            desc_str_list = [chosen_finetune_desc]
    else:
        desc_str_list = ['skip']


    if plot_latent_space<2:
        recompute_plot = False
    else:
        recompute_plot = True
        print('will recompute the plots')
    plot_latent_space =bool(plot_latent_space)

    if len(desc_str_list) > 1 and recompute_plot:
        print('more than one desc_str, do not waste time recomputing plots')
        recompute_plot = False

    homedir = os.path.expanduser("~")
    data_dir = f'{homedir}/PRETRAIN_DATA'
    os.makedirs(data_dir, exist_ok=True)
    data_dir = get_latest_dataset(data_dir=data_dir)

    redo = False
    if '2' in eval_name:
        update_finetune_data('finetune_val',data_dir,redo=redo)
        update_finetune_data('finetune_train',data_dir,redo=redo)
        update_finetune_data('finetune_test',data_dir,redo=redo)
        update_finetune_data('finetune_trainval',data_dir,redo=redo)

    already_run = []
    print('Number of runs to finetune:',len(run_id_list))

    sweep_kwargs = None
    pretrain_eval_on_test = True


    # run_id_list = ['RCC-1735']
    for run_id in run_id_list:
        if run_id in already_run:
            continue
        print('run_id:',run_id)
        plot_latent_space0 = plot_latent_space
        for desc_str in desc_str_list:
            sweep_kwargs = {}

            # customize the sweep kwargs based on the desc_str
            if 'hhl' in desc_str:
                sweep_kwargs['head_hidden_layers'] = 1

            if 'epc_' in desc_str:
                match = re.search(r'epc_(\d+)', desc_str.lower())
                if match:
                    sweep_kwargs['train_kwargs__num_epochs'] = int(match.group(1))  

            if 'regl1_' in desc_str:
                match = re.search(r'regl1_(\d+)', desc_str.lower())
                if match:
                    sweep_kwargs['train_kwargs__l1_reg_weight'] = float(match.group(1))

            if 'regl2_' in desc_str:
                match = re.search(r'regl2_(\d+)', desc_str.lower())
                if match:
                    sweep_kwargs['train_kwargs__l2_reg_weight'] = float(match.group(1))

            if 'lr_' in desc_str:
                match = re.search(r'lr_(\d+)', desc_str.lower())
                if match:
                    lr = float(match.group(1))/1000
                    print('lr:',lr)
                    sweep_kwargs['train_kwargs__learning_rate'] = lr

            if 'noi_' in desc_str:
                match = re.search(r'noi_(\d+)', desc_str.lower())
                if match:
                    noi_factor = float(match.group(1))/10
                    print('noi_factor:',noi_factor)
                    sweep_kwargs['train_kwargs__noise_factor'] = noi_factor

            try:
                run_id = compute_finetune(run_id,n_trials=n_trials,
                                        plot_latent_space=plot_latent_space0,
                                        data_dir=data_dir,
                                        desc_str=desc_str,
                                        eval_name=eval_name,
                                        recompute_plot=recompute_plot,
                                        sweep_kwargs=sweep_kwargs,
                                        pretrain_eval_on_test=pretrain_eval_on_test)
                # do not bother attempting to plot the latent space again
                plot_latent_space0 = False
            except NeptuneException as e:
                print('NeptuneException:',e)
                continue
            except ValueError as e:
                print('ValueError:',e)
                continue
