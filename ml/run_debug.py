import os
import neptune
import sys

# from setup2 import setup_neptune_run
from run_finetune import compute_finetune
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

data_dir = '/DATA'


setup_id = 'pretrain'
run_id = 'RCC-3011'

# run_id = setup_neptune_run(data_dir,setup_id=setup_id,
#                            with_run_id=run_id,
#                            neptune_mode='async',
#                            overwrite_existing_kwargs=True,
#                            batch_size=32)
                        #    encoder_kind='VAE',
                        #    encoder_kwargs=encoder_kwargs)

# sweep_id = 'debug MSKCC'
# sweep_kwargs =  {
#             'holdout_frac': 0.2,
#             'head_hidden_layers': 0,
#             'encoder_kwargs__dropout_rate': 0.4,
#             'train_kwargs__num_epochs': 43,
#             'train_kwargs__early_stopping_patience': 5,
#             'train_kwargs__learning_rate':0.0008954842733052297,
#             'train_kwargs__l2_reg_weight': 0,
#             'train_kwargs__l1_reg_weight': 0,
#             'train_kwargs__noise_factor': 0.25,
#             'train_kwargs__weight_decay': False,
#             'train_kwargs__adversary_weight': 0,
#             'train_kwargs__encoder_weight': 0.0,
#             }

sweep_id = 'debug NIVO-OS'
sweep_kwargs =  {
            'holdout_frac': 0.2,
            'head_hidden_layers': 0,
            'encoder_kwargs__dropout_rate': 0.4,
            'train_kwargs__num_epochs': 72,
            'train_kwargs__early_stopping_patience': 5,
            'train_kwargs__learning_rate':0.004182345148653845,
            'train_kwargs__l2_reg_weight': 0,
            'train_kwargs__l1_reg_weight': 0.0024878414005126055,
            'train_kwargs__noise_factor': 0.05,
            'train_kwargs__weight_decay': 0.0006420308278493263,
            'train_kwargs__adversary_weight': 0,
            'train_kwargs__encoder_weight': 0.25,
            'train_kwargs__clip_grads_with_norm': True
            }



skip_random_init = True
compute_finetune(run_id,plot_latent_space=False,
                     n_trials=1,
                     data_dir=data_dir,
                     desc_str=sweep_id,
                     sweep_kwargs=sweep_kwargs,
                     skip_random_init=skip_random_init,
                     eval_name='val2')