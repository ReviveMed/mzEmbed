import optuna
import neptune
from utils_neptune import get_latest_dataset
from train4 import train_compound_model, evaluate_compound_model, CompoundDataset, create_dataloaders, create_dataloaders_old
import pandas as pd
from run_finetune import compute_finetune, update_finetune_data
import numpy as np
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='


import sys
if len(sys.argv)>1:
    n_optuna_trials = int(sys.argv[1])
else:
    n_optuna_trials = 200

if len(sys.argv)>2:
    run_id = sys.argv[2]
else:
    run_id = 'RCC-3011'

if len(sys.argv)>3:
    sweep_desc = sys.argv[3]
else:
    sweep_desc = 'both-OS'


data_dir = get_latest_dataset()

storage_name = 'optuna'
USE_WEBAPP_DB = True
SAVE_TRIALS = True
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

if sweep_desc == 'both-OS':
    key1_loc = 'eval/val2/Cox_OS__Concordance Index'
    key2_loc = 'eval/train2/Cox_OS__Concordance Index'
elif sweep_desc == 'NIVO-OS':
    key1_loc = 'eval/val2/Cox_NIVO OS__Concordance Index'
    key2_loc = 'eval/train2/Cox_NIVO OS__Concordance Index'
elif sweep_desc == 'EVER-OS':
    key1_loc = 'eval/val2/Cox_EVER OS__Concordance Index'
    key2_loc = 'eval/train2/Cox_EVER OS__Concordance Index'
elif sweep_desc == 'IMDC':
    key1_loc = 'eval/val2/Binary_IMDC__AUROC (micro)'
    key2_loc = 'eval/train2/Binary_IMDC__AUROC (micro)'
elif sweep_desc == 'MSKCC':
    key1_loc = 'eval/val2/Binary_MSKCC__AUROC (micro)'
    key2_loc = 'eval/train2/Binary_MSKCC__AUROC (micro)'
else:
    raise ValueError('sweep_desc not recognized')

def objective(trial):

    sweep_id = f'optuna_{sweep_desc}__{trial.number}'
    holdout_frac = 0
    num_epochs = trial.suggest_int('num_epochs', 1, 100, step=1)
    early_stopping_patience = 0
    if num_epochs > 20:
        early_stopping_patience = trial.suggest_int('early_stopping_patience', 0, 10, step=5)
        if early_stopping_patience == 0:
            holdout_frac = 0
        else:
            holdout_frac = 0.2

    skip_random_init = True
    use_l1_reg = trial.suggest_categorical('use_l1_reg', [True, False])
    use_l2_reg = trial.suggest_categorical('use_l2_reg', [True, False])
    use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])

    if use_l1_reg:
        train_kwargs__l1_reg_weight = trial.suggest_float('train_kwargs__l1_reg_weight', 1e-5, 1e-2, log=True)
    else:
        train_kwargs__l1_reg_weight = 0

    if use_l2_reg:
        train_kwargs__l2_reg_weight = trial.suggest_float('train_kwargs__l2_reg_weight', 1e-5, 1e-2, log=True)
    else:
        train_kwargs__l2_reg_weight = 0

    if use_weight_decay:
        train_kwargs__weight_decay = trial.suggest_float('train_kwargs__weight_decay', 1e-5, 1e-2, log=True)
    else:
        train_kwargs__weight_decay = 0

    sweep_kwargs = {
        'holdout_frac': holdout_frac,
        'head_hidden_layers': trial.suggest_int('head_hidden_layers', 0, 1, step=1),
        'encoder_kwargs__dropout_rate': trial.suggest_float('encoder_kwargs__dropout_rate', 0, 0.5,step=0.1),
        'train_kwargs__num_epochs': num_epochs,
        'train_kwargs__early_stopping_patience': early_stopping_patience,
        'train_kwargs__learning_rate': trial.suggest_float('train_kwargs__learning_rate', 1e-5, 1e-2, log=True),
        'train_kwargs__l2_reg_weight': train_kwargs__l2_reg_weight,
        'train_kwargs__l1_reg_weight': train_kwargs__l1_reg_weight,
        'train_kwargs__noise_factor': trial.suggest_float('train_kwargs__noise_factor', 0, 0.25, step=0.05),
        'train_kwargs__weight_decay': train_kwargs__weight_decay,
        'train_kwargs__adversary_weight': 0,
        'train_kwargs__encoder_weight': trial.suggest_float('train_kwargs__encoder_weight', 0, 1, step=0.25),
        }


    try:
        compute_finetune(run_id,plot_latent_space=False,
                            n_trials=10,
                            desc_str=sweep_id,
                            sweep_kwargs=sweep_kwargs,
                            skip_random_init=skip_random_init,
                            eval_name='val2')
    
    except ValueError as e:
        print('ValueError:', e)
        run = neptune.init_run(project='revivemed/RCC',
            api_token=NEPTUNE_API_TOKEN,
            with_id=run_id,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False)
        run.wait()
        run_struc= run.get_structure()
        if f'{sweep_id}_finetune' in run_struc:
            del run[f'{sweep_id}_finetune']
            if not skip_random_init:
                del run[f'{sweep_id}_randinit']
        run.stop()
        raise optuna.TrialPruned()
    

    run = neptune.init_run(project='revivemed/RCC',
            api_token=NEPTUNE_API_TOKEN,
            with_id=run_id,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False)
    
    result1_array = run[f'{sweep_id}_finetune/{key1_loc}'].fetch_values()

    result1 = np.mean(result1_array['value'])

    result2_array = run[f'{sweep_id}_finetune/{key2_loc}'].fetch_values()
    result2 = np.mean(result2_array['value'])
    trial.set_user_attr('val score',result1)
    trial.set_user_attr('train score',result2)

    run.wait()
    del run[f'{sweep_id}_finetune']
    if not skip_random_init:
        del run[f'{sweep_id}_randinit']
    

    # raise optuna.TrialPruned()

    # if key1_randinit_val > key1_finetune_val:
        # raise optuna.TrialPruned()
    run.stop()

    return result1
    # return result1/result2




if USE_WEBAPP_DB:
    print('using webapp database')
    storage_name = WEBAPP_DB_LOC

study_name = f'finetune_{sweep_desc}_{run_id}_May01'
 
redo = False
update_finetune_data('finetune_val',redo=redo)
update_finetune_data('finetune_train',redo=redo)
update_finetune_data('finetune_test',redo=redo)
update_finetune_data('finetune_trainval',redo=redo)



study = optuna.create_study(directions=['maximize'],
                study_name=study_name, 
                storage=storage_name, 
                load_if_exists=True)

print('############################################')
print('Run {} with {} trials'.format(study_name,n_optuna_trials))
print('############################################')


study.optimize(objective, n_trials=n_optuna_trials)
