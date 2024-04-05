


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
from utils_neptune import  get_run_id_list, check_neptune_existance
from sklearn.linear_model import LogisticRegression
import time
from neptune.exceptions import NeptuneException
from collections import defaultdict

data_dir = '/DATA2'
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(data_dir+'/X_pretrain_train.csv'):
    # data_url = 'https://www.dropbox.com/scl/fo/iy2emxpwa4rkr3ad7vhc2/h?rlkey=hvhfa3ny9dlavnooka3kwvu5v&dl=1' #march 22
    data_url = 'https://www.dropbox.com/scl/fo/2xr104jnz9qda7oemrwob/h?rlkey=wy7q95pj81qpgcn7zida2xjps&dl=1' #march 29
    download_data_dir(data_url, save_dir=data_dir)


def compute_mskcc_finetune(run_id,plot_latent_space=False,
                           n_trials=3,desc_str='MSKCC'):



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

    metric_string_dct = {}
    metric_string_dct['MSKCC Train AUROC'] = 'eval/train/Binary_MSKCC/AUROC (micro)'
    metric_string_dct['MSKCC Val AUROC'] = 'eval/val/Binary_MSKCC/AUROC (micro)'

    if n_trials>0:
        run = neptune.init_run(project='revivemed/RCC',
                        api_token=NEPTUNE_API_TOKEN,
                        with_id=run_id,
                        capture_stdout=False,
                        capture_stderr=False,
                        capture_hardware_metrics=False,
                        mode='read-only')

        kwargs = run['pretrain/original_kwargs'].fetch()
        kwargs = convert_neptune_kwargs(kwargs)
        

        run.stop()
        
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
        kwargs['plot_latent_space_cols'] = ['MSKCC']
        kwargs['y_head_cols'] = ['MSKCC BINARY']
        kwargs['y_adv_cols'] = []
        kwargs['upload_models_to_neptune'] = False


        kwargs['head_kwargs_dict'] = {}
        kwargs['adv_kwargs_dict'] = {}
        kwargs['head_kwargs_list'] = [{
            'kind': 'Binary',
            'name': 'MSKCC',
            'weight': 1,
            'y_idx': 0,
            'hidden_size': 4,
            'num_hidden_layers': 0,
            'dropout_rate': 0,
            'activation': 'leakyrelu',
            'use_batch_norm': False,
            'num_classes': 2,
            }]
        
        kwargs['encoder_kwargs']['dropout_rate'] = 0.2
        kwargs['adv_kwargs_list'] = []
        kwargs['train_kwargs']['num_epochs'] = 30
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
        n_trials = 0


    if len(sys.argv)>3:
        chosen_run_id = sys.argv[3]
    else:
        chosen_run_id = None

    # if chosen_run_id is None:
        # chosen_run_id = input('Enter Run id: ')
    try:
        chosen_run_id = int(chosen_run_id)
        chosen_run_id = 'RCC-'+str(chosen_run_id)
    except:
        pass


    already_run = []
    # run_id_list = ['RCC-1296']
    # run_id_list = ['RCC-924','RCC-973','RCC-938','RCC-931','RCC-984','RCC-933','RCC-1416','RCC-1364','RCC-1129']
    if chosen_run_id is not None:
        run_id_list = [chosen_run_id]
    else:
        run_id_list = get_run_id_list(tags=['april04_top200'],encoder_kind='AE')
        # already_run = get_run_id_list(tags=['top'],encoder_kind='AE')
    run_id_list.append('RCC-1620')
    print(len(run_id_list))

    # run_id_list = ['RCC-1735']
    for run_id in run_id_list:
        if run_id in already_run:
            continue
        print('run_id:',run_id)
        try:
            # run_id = compute_mskcc_finetune(run_id,n_trials=n_trials,plot_latent_space=plot_latent_space,desc_str='Apr04_MSKCC')
            run_id = compute_mskcc_finetune(run_id,n_trials=n_trials,plot_latent_space=plot_latent_space,desc_str='Apr04.1_MSKCC')
        except NeptuneException as e:
            print('NeptuneException:',e)
            continue


    # clean up the previously failed runs #RCC-1132, RCC-1126
    # run_id_list = ['RCC-926', 'RCC-927', 'RCC-928','RCC-929']
    # for run_id in run_id_list:
    #     print('run_id:',run_id)
    #     run_id = compute_mskcc_finetune(run_id)

    #     run = neptune.init_run(project='revivemed/RCC',
    #                     api_token=NEPTUNE_API_TOKEN,
    #                     with_id=run_id,
    #                     capture_stdout=False,
    #                     capture_stderr=False,
    #                     capture_hardware_metrics=False)
        
    #     run['sys/failed'] = False
    #     run.stop()