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


run_id = 'RCC-1296'
kwargs = {}
kwargs['overwrite_existing_kwargs'] = True
kwargs['load_model_loc'] = 'pretrain'
kwargs['run_evaluation'] = False
kwargs['run_training'] = False
kwargs['save_latent_space'] = True
kwargs['plot_latent_space'] = 'both'
kwargs['plot_latent_space_cols'] = ['Study ID','Cohort Label','is Pediatric']

run_id = setup_neptune_run(data_dir,setup_id='pretrain',with_run_id=run_id,**kwargs)