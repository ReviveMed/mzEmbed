

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




# setup_wrapper(
#     project_id = 'revivemed/RCC',
#     api_token = NEPTUNE_API_TOKEN,
#     setup_id = 'pretrain',
#     tags = 'setup4-test',
#     fit_subset_col = 'Pretrain Discovery Train',
#     eval_subset_col_list = ['Pretrain Discovery Val','Pretrain Test'],
#     selections_df = selections_df,
#     output_dir = output_dir,
#     head_name_list=['Sex','Age'])



# eval_param_list = [
#     {'y_col_name': 'MSKCC',
#      'y_head': 'IMDC',
#      'y_cols' ['MSKCC']
#     }
# ]

# setup_wrapper(
#     project_id = 'revivemed/RCC',
#     api_token = NEPTUNE_API_TOKEN,
#     setup_id = 'finetune v0',
#     tags = 'setup4-test',
#     fit_subset_col = 'Finetune Discovery Train',
#     eval_subset_col_list = ['Finetune Discovery Val','Finetune Test'],
#     selections_df = selections_df,
#     output_dir = output_dir,
#     pretrained_model_id = 'RCC-4201',
#     pretrained_loc='pretrain',
#     overwrite_params_fit_kwargs={'num_epochs':10},
#     num_iterations = 5,
#     head_name_list=['IMDC'])
#     # eval_param_list = eval_param_list)


setup_wrapper(
    project_id = 'revivemed/RCC',
    api_token = NEPTUNE_API_TOKEN,
    setup_id = 'finetune v0',
    tags = 'setup4-test',
    fit_subset_col = 'Finetune Discovery Train',
    eval_subset_col_list = ['Finetune Discovery Val','Finetune Test'],
    selections_df = selections_df,
    output_dir = output_dir,
    pretrained_model_id = 'RCC-4214',
    pretrained_loc='pretrain',
    overwrite_params_fit_kwargs={'num_epochs':10},
    num_iterations = 5,
    head_name_list=['Both-OS'])
    # eval_param_list = eval_param_list)




# setup_wrapper(
#     project_id = 'revivemed/RCC',
#     api_token = NEPTUNE_API_TOKEN,
#     setup_id = 'pretrain',
#     tags = 'setup4-test',
#     fit_subset_col = 'Pretrain Discovery Train',
#     eval_subset_col_list = ['Pretrain Discovery Val','Pretrain Test'],
#     selections_df = selections_df,
#     output_dir = output_dir,
#     head_name_list=['Sex','Age'],
#     overwrite_params_fit_kwargs={'num_epochs':10},
#     overwrite_existing_params=True)

# def main(STUDY_INFO_DICT,num_trials=5):

#     def compute_objective(run_id):
#         return objective_func4(run_id,
#                                 study_info_dict=STUDY_INFO_DICT,
#                                 project_id=project_id,
#                                 neptune_api_token=neptune_api_token,
#                                 setup_id=setup_id,
#                                 eval_file_id=eval_file_id)


#     def objective(trial):

#         trial.set_user_attr('run_id',run_id)
#         trial.set_user_attr('setup_id',setup_id)

#         try:

#             setup_wrapper(
#                 project_id = 'revivemed/RCC',
#                 api_token = NEPTUNE_API_TOKEN,
#                 setup_id = 'finetune v0',
#                 tags = 'setup4-test',
#                 fit_subset_col = 'Finetune Discovery Train',
#                 eval_subset_col_list = ['Finetune Discovery Val','Finetune Test'],
#                 selections_df = selections_df,
#                 output_dir = output_dir,
#                 pretrained_model_id = 'RCC-4201',
#                 pretrained_loc='pretrain',
#                 overwrite_params_fit_kwargs={'num_epochs':10},
#                 num_iterations = 5,
#                 optuna_trial=trial,
#                 optuna_study_info_dict = STUDY_INFO_DICT,
#                 head_name_list=['IMDC'])

#             return compute_objective(run_id)
    
#     # except Exception as e:
#         except ValueError as e:
#             print(e)
#             # return float('nan')
#             raise optuna.TrialPruned()