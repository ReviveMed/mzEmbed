
import os
import pandas as pd
from prep_run import create_selected_data, make_kwargs_set, get_task_head_kwargs
from utils_neptune import get_latest_dataset, convert_neptune_kwargs
from setup3 import setup_neptune_run
import neptune

from prep_run import get_selection_df

## 
# %% Load the latest data

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

project_id = 'revivemed/RCC'
homedir = os.path.expanduser("~")
input_data_dir = f'{homedir}/INPUT_DATA'
os.makedirs(input_data_dir, exist_ok=True)
input_data_dir = get_latest_dataset(data_dir=input_data_dir,api_token=NEPTUNE_API_TOKEN,project=project_id)


# %%
# selections_df = None
selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv',index_col=0)

output_dir = f'{homedir}/PROCESSED_DATA'
os.makedirs(output_dir, exist_ok=True)
subdir_col = 'Study ID'

fit_subset_col = 'Pretrain Discovery Train'
eval_subset_col = 'Pretrain Discovery Val'
setup_id = 'pretrain'


remove_y_nans = False


_, fit_file_id = create_selected_data(input_data_dir=input_data_dir,
                                               sample_selection_col=fit_subset_col,
                                               subdir_col=subdir_col,
                                               output_dir=output_dir,
                                               metadata_df=None,
                                               selections_df=selections_df)



_, eval_file_id = create_selected_data(input_data_dir=input_data_dir,
                                                sample_selection_col=eval_subset_col,
                                                subdir_col=subdir_col,
                                                output_dir=output_dir,
                                                metadata_df=None,
                                                selections_df=selections_df)

X_eval_file = f'{output_dir}/X_{eval_file_id}.csv'
y_eval_file = f'{output_dir}/y_{eval_file_id}.csv'
X_fit_file = f'{output_dir}/X_{fit_file_id}.csv'
y_fit_file = f'{output_dir}/y_{fit_file_id}.csv'

# %%

# %%


run_id_list = ['RCC-3244',
'RCC-3195',
'RCC-3211',
'RCC-3213',
'RCC-3269',
'RCC-3212',
'RCC-3276',
'RCC-3315',
'RCC-3285',
'RCC-3264']

weight_list = [1.0,0.0,2.0]
for smoke_weight in weight_list:
    for bmi_weight in weight_list:

        for run_id in run_id_list:

                existing_run = neptune.init_run(project=project_id,
                                        api_token=NEPTUNE_API_TOKEN,
                                        with_id=run_id,
                                        mode='read-only')

                # existing_run = neptune.init_run(project=project_id, run=run_id, mode='read-only', api_token=NEPTUNE_API_TOKEN)
                existing_kwargs = existing_run['pretrain/original_kwargs'].fetch()
                existing_kwargs = convert_neptune_kwargs(existing_kwargs)
                existing_run.stop()

                head_kwargs_dict = existing_kwargs['head_kwargs_dict']
                y_head_cols = existing_kwargs['y_head_cols']
                y_adv_cols = existing_kwargs['y_adv_cols']

                head_kwargs_dict['BMI'], y_head_cols = get_task_head_kwargs(head_kind='Regression',
                                                                    y_head_col='BMI',
                                                                    y_cols=y_head_cols,
                                                                    head_name='BMI',
                                                                    default_weight=bmi_weight)
                
                head_kwargs_dict['Smoking'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
                                                                    y_head_col='Smoking Status',
                                                                    y_cols=y_head_cols,
                                                                    head_name='Smoking',
                                                                    default_weight=smoke_weight)


                plot_latent_space_cols  = list(set(y_head_cols + y_adv_cols))
                plot_latent_space_cols = ['Study ID']

                updated_kwargs = {**existing_kwargs}
                updated_kwargs['head_kwargs_dict'] = head_kwargs_dict
                updated_kwargs['y_head_cols'] = y_head_cols
                updated_kwargs['y_adv_cols'] = y_adv_cols
                updated_kwargs['plot_latent_space_cols'] = plot_latent_space_cols
                updated_kwargs['source run_id'] = run_id

                _ = setup_neptune_run(input_data_dir,
                                            setup_id=setup_id,
                                            project_id=project_id,

                                            neptune_mode='async',
                                            yes_logging = True,
                                            neptune_api_token=NEPTUNE_API_TOKEN,
                                            tags=['v4'],
                                            **updated_kwargs)


    break
