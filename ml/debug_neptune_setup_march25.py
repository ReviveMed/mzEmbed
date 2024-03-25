import neptune
from neptune.utils import stringify_unsupported
import os
from train2 import get_end_state_eval_funcs

from setup import setup_neptune_run
from misc import download_data_dir
from sklearn.linear_model import LogisticRegression

# data_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination/alignment_RCC_2024_Feb_27/March_22_Data'


data_dir = '/DATA2'
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(data_dir+'/X_pretrain_train.csv'):
    data_url = 'https://www.dropbox.com/scl/fo/iy2emxpwa4rkr3ad7vhc2/h?rlkey=hvhfa3ny9dlavnooka3kwvu5v&dl=1'
    download_data_dir(data_url, save_dir=data_dir)

activation = 'leakyrelu'
encoder_kind = 'AE'
# encoder_kind = trial.suggest_categorical('encoder_kind', ['AE', 'VAE'])
# latent_size = 8

# run_id = 'RCC-29'

for latent_size in [4,8,16,32]:
    for num_hidden_layers in [1,2,3]:
        for adv_weight in [0.1,0.5,1.0,2.0,5.0]:
            for head_weight in [0.1,0.5,1.0,2.0,5.0]:
                run_id = 'NEW'

                encoder_kwargs = {
                    'activation': activation,
                    'latent_size': latent_size,
                    'num_hidden_layers': num_hidden_layers,
                    'dropout_rate': 0.2,
                    'use_batch_norm': False,
                    'hidden_size': 1.5*latent_size,
                    }



                kwargs = {
                        ################
                        ## General ##
                        'encoder_kind': 'AE',
                        'encoder_kwargs': encoder_kwargs,
                        'other_size': 1,

                        ################
                        ## Pretrain ##

                        'holdout_frac': 0.2, # each trial the train/val samples will be different, also not stratified?
                        'batch_size': 64,
                        'head_kind': 'NA',
                        'head_kwargs' : {},
                        
                        'adv_kind': 'NA',
                        'adv_kwargs' : {},

                        'train_kwargs': {
                            # 'num_epochs': trial.suggest_int('pretrain_epochs', 10, 100,log=True),
                            'num_epochs': 100,
                            'lr': 0.01,
                            'weight_decay': 0,
                            'l1_reg_weight': 0,
                            'l2_reg_weight': 0.001,
                            'encoder_weight': 1,
                            'head_weight': 0,
                            'adversary_weight': 0,
                            'noise_factor': 0.1,
                            'early_stopping_patience': 20,
                            'eval_funcs': get_end_state_eval_funcs(),
                            'adversarial_mini_epochs': 5,
                        },

                        'eval_kwargs' :{
                            'eval_funcs': get_end_state_eval_funcs(),
                            'sklearn_models': {
                                'Adversary Logistic Regression': LogisticRegression(max_iter=10000, C=1.0, solver='lbfgs')
                            }

                        }
                }


                kwargs['adv_kind'] = 'MultiClassClassifier'
                kwargs['adv_kwargs'] = {
                    'hidden_size': 4,
                    'num_hidden_layers': 1,
                    'dropout_rate': 0,
                    'activation': 'leakyrelu',
                    'use_batch_norm': False,
                    'num_classes': 19,
                    }
                kwargs['train_kwargs']['adversary_weight'] = adv_weight


                kwargs['head_kind'] = 'BinaryClassifier'
                kwargs['head_kwargs'] = {
                    'hidden_size': 4,
                    'num_hidden_layers': 1,
                    'dropout_rate': 0,
                    'activation': 'leakyrelu',
                    'use_batch_norm': False,
                    'num_classes': 2,
                    }
                kwargs['train_kwargs']['head_weight'] = head_weight

                kwargs['run_training'] = True
                kwargs['run_evaluation'] = True
                kwargs['save_latent_space'] = True
                kwargs['plot_latent_space'] = 'both'
                kwargs['plot_latent_space_cols'] = ['Study ID','Cohort Label']



                run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)




                ### Finetune Test

                kwargs['load_model_loc'] = 'pretrain'
                # kwargs['load_model_loc'] = 'finetune'
                kwargs['run_training'] = True
                kwargs['run_evaluation'] = True
                kwargs['save_latent_space'] = True
                kwargs['plot_latent_space'] = 'sns'
                kwargs['plot_latent_space_cols'] = ['MSKCC']
                kwargs['y_head_col'] = 'MSKCC BINARY'
                kwargs['y_adv_col'] = 'MSKCC BINARY'


                kwargs['adv_kind'] = 'NA'
                kwargs['adv_kwargs'] = {}
                kwargs['train_kwargs']['adversary_weight'] = 0


                kwargs['head_kind'] = 'BinaryClassifier'
                kwargs['head_kwargs'] = {
                    'hidden_size': 4,
                    'num_hidden_layers': 0,
                    'dropout_rate': 0,
                    'activation': 'leakyrelu',
                    'use_batch_norm': False,
                    'num_classes': 2,
                    }
                kwargs['train_kwargs']['head_weight'] = 1
                kwargs['train_kwargs']['encoder_weight'] = 1
                kwargs['eval_kwargs']['sklearn_models'] = {}

                run_id = setup_neptune_run(data_dir,setup_id='finetune',with_run_id=run_id,**kwargs)