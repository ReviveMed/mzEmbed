


NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

import neptune
import os
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

data_dir = '/DATA2'
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(data_dir+'/X_pretrain_train.csv'):
    # data_url = 'https://www.dropbox.com/scl/fo/iy2emxpwa4rkr3ad7vhc2/h?rlkey=hvhfa3ny9dlavnooka3kwvu5v&dl=1' #march 22
    data_url = 'https://www.dropbox.com/scl/fo/2xr104jnz9qda7oemrwob/h?rlkey=wy7q95pj81qpgcn7zida2xjps&dl=1' #march 29
    download_data_dir(data_url, save_dir=data_dir)


def compute_mskcc_finetune(run_id):


        ###############################
        ### Plot the latent space
        kwargs = {}
        kwargs['overwrite_existing_kwargs'] = True
        kwargs['encoder_kind'] = 'AE'
        kwargs['load_model_loc'] = 'pretrain'
        kwargs['run_evaluation'] = False
        kwargs['run_training'] = False
        kwargs['save_latent_space'] = True
        # kwargs['save_latent_space'] = False

        kwargs['plot_latent_space'] = 'sns' #'both'
        kwargs['plot_latent_space_cols'] = ['Study ID','Cohort Label','is Pediatric']

        # run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

        kwargs['eval_name'] = 'train'
        run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

        kwargs['eval_name'] = 'test'
        run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)

        kwargs['eval_name'] = 'val'
        run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)


        ############################### Finetune

        # kwargs = {}
        # kwargs['load_model_loc'] = 'finetune_mkscc'
        # kwargs['run_train'] = False
        # kwargs['run_evaluation'] = True
        # kwargs['eval_kwargs'] = {}
        # kwargs['eval_kwargs']['sklearn_models'] = {}
        # kwargs['overwrite_existing_kwargs'] = True

        run = neptune.init_run(project='revivemed/RCC',
                        api_token=NEPTUNE_API_TOKEN,
                        with_id=run_id,
                        capture_stdout=False,
                        capture_stderr=False,
                        capture_hardware_metrics=False)

        kwargs = run['pretrain/kwargs'].fetch()
        kwargs = convert_neptune_kwargs(kwargs)
        
        if check_neptune_existance(run,'finetune_mkscc_1'):
            run.stop()
            return run_id
            # score = run['finetune_mkscc_1/eval/val/Binary_MSKCC/AUROC (micro)'].fetch()
            # if score > 0.95:
            #     del run['finetune_mkscc_1/kwargs']
            #     del run['finetune_mkscc_1']
            #     run.stop()
            # else:
            #     run.stop()
            #     return run_id
        run.stop()

        # pause for 2 seconds
        # time.sleep(2)

        kwargs['load_encoder_loc'] = 'pretrain'
        kwargs['load_model_loc'] = False
        kwargs['X_filename'] = 'X_finetune'
        kwargs['y_filename'] = 'y_finetune'
        kwargs['eval_name'] = 'val'
        kwargs['run_training'] = True
        kwargs['run_evaluation'] = True
        kwargs['save_latent_space'] = True
        kwargs['plot_latent_space'] = 'sns'
        kwargs['plot_latent_space_cols'] = ['MSKCC']
        kwargs['y_head_cols'] = ['MSKCC BINARY']
        kwargs['y_adv_cols'] = []


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
        kwargs['train_kwargs']['num_epochs'] = 20
        kwargs['train_kwargs']['early_stopping_patience'] = 0
        kwargs['holdout_frac'] = 0
        kwargs['train_kwargs']['head_weight'] = 1
        kwargs['train_kwargs']['encoder_weight'] = 0
        kwargs['train_kwargs']['adversary_weight'] = 0
        kwargs['train_kwargs']['learning_rate'] = 0.001
        kwargs['train_kwargs']['l2_reg_weight'] = 0.0005
        kwargs['train_kwargs']['l1_reg_weight'] = 0.005
        kwargs['train_kwargs']['noise_factor'] = 0.1
        kwargs['train_kwargs']['weight_decay'] = 0
        kwargs['run_evaluation'] = True
        kwargs['eval_kwargs'] = {}
        kwargs['eval_kwargs']['sklearn_models'] = {}


        kwargs = convert_model_kwargs_list_to_dict(kwargs)

        # kwargs = convert_model_kwargs_list_to_dict(kwargs)
        # run_id = setup_neptune_run(data_dir,setup_id='finetune_mkscc',with_run_id=run_id,**kwargs)
        _ = setup_neptune_run(data_dir,setup_id='finetune_mkscc_0',with_run_id=run_id,**kwargs)
        _ = setup_neptune_run(data_dir,setup_id='finetune_mkscc_1',with_run_id=run_id,**kwargs)
        _ = setup_neptune_run(data_dir,setup_id='finetune_mkscc_2',with_run_id=run_id,**kwargs)

        kwargs['run_random_init'] = True
        _ = setup_neptune_run(data_dir,setup_id='randinit_mkscc_0',with_run_id=run_id,**kwargs)

        kwargs['load_model_weights'] = False
        _ = setup_neptune_run(data_dir,setup_id='randinit_mkscc_1',with_run_id=run_id,**kwargs)

        kwargs['load_model_weights'] = False
        kwargs['run_random_init'] = False
        _ = setup_neptune_run(data_dir,setup_id='randinit_mkscc_2',with_run_id=run_id,**kwargs)

        return run_id

# def main():

if __name__ == '__main__':

    already_run = []
    # run_id_list = ['RCC-1296']
    # run_id_list = ['RCC-924','RCC-973','RCC-938','RCC-931','RCC-984','RCC-933','RCC-1416','RCC-1364','RCC-1129']
    run_id_list = get_run_id_list(tags=['april2'],encoder_kind='AE')
    for run_id in run_id_list:
        if run_id in already_run:
            continue
        print('run_id:',run_id)
        try:
            run_id = compute_mskcc_finetune(run_id)
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