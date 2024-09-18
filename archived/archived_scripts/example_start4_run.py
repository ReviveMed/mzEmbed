

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

# Example for pretraining the Age and Sex tasks

setup_wrapper(
    project_id = 'revivemed/RCC',
    api_token = NEPTUNE_API_TOKEN,
    setup_id = 'pretrain',
    tags = 'setup4-test',
    fit_subset_col = 'Pretrain Discovery Train',
    eval_subset_col_list = ['Pretrain Discovery Val','Pretrain Test'],
    selections_df = selections_df,
    output_dir = output_dir,
    head_name_list=['Sex','Age'])


# Example for finetuning the IMDC head task

setup_wrapper(
    project_id = 'revivemed/RCC',
    api_token = NEPTUNE_API_TOKEN,
    setup_id = 'IMDC finetune v0',
    tags = 'setup4-test',
    fit_subset_col = 'Finetune Discovery Train',
    eval_subset_col_list = ['Finetune Discovery Val','Finetune Test'],
    selections_df = selections_df,
    output_dir = output_dir,
    pretrained_model_id = 'RCC-3828',
    pretrained_loc='pretrain',
    overwrite_params_fit_kwargs={'num_epochs':10},
    num_iterations = 5,
    head_name_list=['IMDC'])

# Example for randinitialize the encoder before finetuning on the IMDC head task
setup_wrapper(
    project_id = 'revivemed/RCC',
    api_token = NEPTUNE_API_TOKEN,
    setup_id = 'IMDC randinit v0',
    tags = 'setup4-test',
    fit_subset_col = 'Finetune Discovery Train',
    eval_subset_col_list = ['Finetune Discovery Val','Finetune Test'],
    selections_df = selections_df,
    output_dir = output_dir,
    pretrained_model_id = 'RCC-3828',
    pretrained_loc='pretrain',
    overwrite_params_fit_kwargs={'num_epochs':10},
    num_iterations = 5,
    use_rand_init=True,
    head_name_list=['IMDC'])


# Example for finetuning the OS head task

setup_wrapper(
    project_id = 'revivemed/RCC',
    api_token = NEPTUNE_API_TOKEN,
    setup_id = 'finetune v0',
    tags = 'setup4-test',
    fit_subset_col = 'Finetune Discovery Train',
    eval_subset_col_list = ['Finetune Discovery Val','Finetune Test'],
    selections_df = selections_df,
    output_dir = output_dir,
    pretrained_model_id = 'RCC-3828',
    pretrained_loc='pretrain',
    num_iterations = 5,
    head_name_list=['Both-OS'])



# Example for finetuning the NIVO-OS head task with adversarial training on the EVER-OS

setup_wrapper(
    project_id = 'revivemed/RCC',
    api_token = NEPTUNE_API_TOKEN,
    setup_id = 'finetune v0',
    tags = 'setup4-test',
    fit_subset_col = 'Finetune Discovery Train',
    eval_subset_col_list = ['Finetune Discovery Val','Finetune Test'],
    selections_df = selections_df,
    output_dir = output_dir,
    pretrained_model_id = 'RCC-3828',
    pretrained_loc='pretrain',
    overwrite_params_fit_kwargs={'num_epochs':10},
    num_iterations = 5,
    head_name_list=['NIVO-OS'],
    adv_head_name_list=['EVER-OS'])