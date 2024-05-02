# %%
import optuna
import neptune
from utils_neptune import get_latest_dataset
from train4 import train_compound_model, evaluate_compound_model, CompoundDataset, create_dataloaders, create_dataloaders_old
import pandas as pd
from run_finetune import compute_finetune, update_finetune_data
import numpy as np
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

# %%
storage_name = 'optuna'

USE_WEBAPP_DB = True
SAVE_TRIALS = True
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

run_id = 'RCC-3011'
sweep_desc = 'both-OS'

def retrieve_best_trial_num(study):
    best_trial = study.best_trial
    trial_num = best_trial.number
    return trial_num

def retrieve_trial_params(study,trial_num):
    trial = study.trials[trial_num]
    params = trial.params
    return params





if USE_WEBAPP_DB:
    print('using webapp database')
    storage_name = WEBAPP_DB_LOC

study_name = f'finetune_{sweep_desc}_{run_id}_May01'


study = optuna.load_study(study_name, storage=storage_name)

# %%
trial_num = 40

params = retrieve_trial_params(study,trial_num)
# %%
