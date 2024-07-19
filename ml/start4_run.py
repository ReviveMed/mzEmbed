

from setup4 import setup_wrapper
import os
import pandas as pd

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

project_id = 'revivemed/RCC'
# homedir = os.path.expanduser("~")

project_id = 'revivemed/RCC'
homedir = os.path.expanduser("~")
input_data_dir = f'{homedir}/INPUT_DATA'


# %%
# selections_df = None
selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv',index_col=0)

output_dir = f'{homedir}/PROCESSED_DATA'
os.makedirs(output_dir, exist_ok=True)
subdir_col = 'Study ID'

# setup_wrapper(
#     project_id = 'revivemed/RCC',
#     api_token = NEPTUNE_API_TOKEN,
#     setup_id = 'pretrain',
#     tags = 'setup4-test',
#     fit_subset_col = 'Pretrain Discovery Train',
#     eval_subset_col_list = ['Pretrain Discovery Val'],
#     selections_df = selections_df,
#     output_dir = output_dir)



setup_wrapper(
    project_id = 'revivemed/RCC',
    api_token = NEPTUNE_API_TOKEN,
    setup_id = 'pretrain',
    tags = 'setup4-test',
    fit_subset_col = 'Pretrain Discovery Train',
    eval_subset_col_list = ['Pretrain Discovery Val'],
    selections_df = selections_df,
    output_dir = output_dir,
    head_name_list=['Sex','Age'],
    overwrite_params_fit_kwargs={'num_epochs':10},
    overwrite_existing_params=True)


# setup_wrapper(
#     project_id = 'revivemed/RCC',
#     api_token = NEPTUNE_API_TOKEN,
#     setup_id = 'pretrain',
#     tags = 'setup4-test',
#     fit_subset_col = 'Pretrain Discovery Train',
#     eval_subset_col_list = ['Pretrain Discovery Val'],
#     selections_df = selections_df,
#     output_dir = output_dir,
#     head_name_list=['Sex','Age'],
#     encoder_model_id = 'RCC-4090'
#     overwrite_params_fit_kwargs={'num_epochs':100},
#     overwrite_existing_params=True)


    # project_id = kwargs.get('project_id',PROJECT_ID)
    # api_token = kwargs.get('api_token',NEPTUNE_API_TOKEN)
    # prefix = kwargs.get('prefix','training_run')
    # subdir_col = kwargs.get('subdir_col','Study ID')
    # selections_df = kwargs.get('selections_df',None)
    # output_dir = kwargs.get('output_dir',None)
    # yes_plot_latent_space = kwargs.get('yes_plot_latent_space',False)
    # fit_subset_col = kwargs.get('fit_subset_col','train')
    # eval_subset_col_list = kwargs.get('eval_subset_col_list',[])
    # eval_params_list = kwargs.get('eval_params_list',None)
    # tags = kwargs.get('tags',[])
    
    # resume_with_id = kwargs.get('resume_with_id',None)
    # encoder_project_id = kwargs.get('encoder_project_id',project_id)
    # encoder_model_id = kwargs.get('encoder_model_id',None)
    # encoder_is_a_run = kwargs.get('encoder_is_a_run',True) #is the encoder coming from a Neptune Model or Neptune Run object?
    # encoder_load_dir = kwargs.get('encoder_load_dir',None) 
    # head_name_list = kwargs.get('head_name_list',[])
    # adv_name_list = kwargs.get('adv_name_list',[])
    # optuna_study_info_dict = kwargs.get('optuna_study_info_dict',None)
    # num_iterations = kwargs.get('num_iterations',1)
    # setup_id = kwargs.get('setup_id','')
    # optuna_trial = kwargs.get('optuna_trial',None)
    # encoder_kind = kwargs.get('encoder_kind','VAE')

    # overwrite_default_params = kwargs.get('overwrite_existing_params',False)
    # overwrite_params_fit_kwargs = kwargs.get('overwrite_params_fit_kwargs',{})
    # overwrite_params_task_kwargs = kwargs.get('overwrite_params_task_kwargs',{})
    # overwrite_params_other_kwargs = kwargs.get('overwrite_params_other_kwargs',{})