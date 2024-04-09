import optuna
import neptune
from utils_neptune import get_latest_dataset
from train3 import train_compound_model, evaluate_compound_model, CompoundDataset, create_dataloaders, create_dataloaders_old
import pandas as pd
from run_finetune import compute_finetune

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='


data_dir = get_latest_dataset()

run_id = 'RCC-1915'
sweep_desc = 'IMDC'
key1 = 'IMDC Val AUROC'
key2 = 'IMDC Train AUROC'
storage_name = 'optuna'
USE_WEBAPP_DB = True
SAVE_TRIALS = True
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

def objective(trial):

    sweep_id = f'optuna_{sweep_desc}__{trial.number}'

    num_epochs = trial.suggest_int('num_epochs', 1, 50, step=2)
    if num_epochs > 20:
        early_stopping_patience = trial.suggest_int('early_stopping_patience', 0, 10 ,step=5)
        if early_stopping_patience == 0:
            holdout_frac = 0
        else:
            holdout_frac = 0.2

    sweep_kwargs = {
        'holdout_frac': holdout_frac,
        'encoder_kwargs__dropout_rate': trial.suggest_float('encoder_kwargs__dropout_rate', 0, 0.5,step=0.1),
        'train_kwargs__num_epochs': num_epochs,
        'train_kwargs__early_stopping_patience': early_stopping_patience,
        'train_kwargs__learning_rate': trial.suggest_float('train_kwargs__learning_rate', 1e-5, 1e-2, log=True),
        'train_kwargs__l2_reg_weight': trial.suggest_float('train_kwargs__l2_reg_weight', 1e-5, 1e-2, log=True),
        'train_kwargs__l1_reg_weight': trial.suggest_float('train_kwargs__l1_reg_weight', 1e-5, 1e-2, log=True),
        'train_kwargs__noise_factor': 0.1,
        'train_kwargs__weight_decay': 0,
        }



    compute_finetune(run_id,plot_latent_space=False,
                           n_trials=10,desc_str=sweep_id,sweep_kwargs=sweep_kwargs)
    

    run = neptune.init_run(project='revivemed/RCC',
            api_token=NEPTUNE_API_TOKEN,
            with_id=run_id,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
            mode='read-only')
    
    key1_finetune_val = run[f'summary/finetune_{sweep_id}/{key1} avg'].fetch()
    key1_randinit_val = run[f'summary/randinit_{sweep_id}/{key1} avg'].fetch()
    key2_finetune_val = run[f'summary/finetune_{sweep_id}/{key2} avg'].fetch()

    if key2_finetune_val < key1_finetune_val:
        key1_finetune_val = key2_finetune_val

    return key1_finetune_val, key2_finetune_val, key1_randinit_val




if USE_WEBAPP_DB:
    print('using webapp database')
    storage_name = WEBAPP_DB_LOC

study_name = f'finetune_{sweep_desc}_{run_id}'
 



study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'],
                study_name=study_name, 
                storage=storage_name, 
                load_if_exists=True)



study.optimize(objective, n_trials=10)
