
import os
import pandas as pd
from prep_run import create_selected_data, make_kwargs_set, get_task_head_kwargs
from utils_neptune import get_latest_dataset
from setup3 import setup_neptune_run



## 
# %% Load the latest data

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

homedir = os.path.expanduser("~")
input_data_dir = f'{homedir}/INPUT_DATA'
os.makedirs(input_data_dir, exist_ok=True)
input_data_dir = get_latest_dataset(data_dir=input_data_dir,api_token=NEPTUNE_API_TOKEN)


# %%
selections_df = pd.DataFrame()
output_dir = f'{homedir}/PROCESSED_DATA'
os.makedirs(output_dir, exist_ok=True)
subdir_col = 'Study ID'
fit_subset_col = 'Pretrain Discovery Train'
eval_subset_col = 'Pretrain Discovery Val'
setup_id = 'pretrain'

_, fit_file_id = create_selected_data(input_data_dir=input_data_dir,
                                               sample_selection_col=fit_subset_col,
                                               subdir_col=subdir_col,
                                               output_dir=output_dir,
                                               metadata_df=None)



_, eval_file_id = create_selected_data(input_data_dir=input_data_dir,
                                               sample_selection_col=eval_subset_col,
                                               subdir_col=subdir_col,
                                               output_dir=output_dir,
                                               metadata_df=selections_df)

X_eval_file = f'{output_dir}/X_{eval_file_id}.csv'
y_eval_file = f'{output_dir}/y_{eval_file_id}.csv'
X_fit_file = f'{output_dir}/X_{fit_file_id}.csv'
y_fit_file = f'{output_dir}/y_{fit_file_id}.csv'

# %%

# Determine the Task Heads
y_head_cols = []
y_adv_cols = []
head_kwargs_dict = {}
adv_kwargs_dict = {}

# head_kwargs_dict['Cohort Label'], y_head_cols = get_task_head_kwargs(head_kind='MultiClass',
#                                                      y_head_col='Cohort Label v0',
#                                                      y_cols=y_head_cols,
#                                                      head_name='Cohort Label',
#                                                      num_classes=4)

head_kwargs_dict['Exact Age'], y_head_cols = get_task_head_kwargs(head_kind='Regression',
                                                     y_head_col='Age',
                                                     y_cols=y_head_cols,
                                                     head_name='Exact Age')

# head_kwargs_dict['Both OS'], y_head_cols = get_task_head_kwargs(head_kind='Cox',
#                                                      y_head_col='OS',
#                                                      y_cols=y_head_cols,
#                                                      head_name='Both OS')

# print(y_head_cols)
# %%
encoder_kind = 'VAE'

kwargs = make_kwargs_set(sig_figs=2,
                encoder_kind=encoder_kind,
                activation_func= 'leakyrelu',
                use_batch_norm= False,
                head_kwargs_dict=head_kwargs_dict,
                adv_kwargs_dict=adv_kwargs_dict,
                num_epochs=10,
                latent_size=16,
                hidden_size=24,
                adv_weight=0.0)


# %%

plot_latent_space_cols  = list(set(y_head_cols + y_adv_cols))

run_id = setup_neptune_run(input_data_dir,
                            setup_id=setup_id,

                            neptune_mode='async',
                            yes_logging = True,
                            neptune_api_token=NEPTUNE_API_TOKEN,
                            tags=['v4'],
                            y_head_cols=y_head_cols,
                            y_adv_cols=y_adv_cols,
                            num_repeats=1,

                            run_training=True,
                            X_fit_file=X_fit_file,
                            y_fit_file=y_fit_file,
                            train_name=fit_file_id,

                            run_evaluation=True,
                            X_eval_file=X_eval_file,
                            y_eval_file=y_eval_file,
                            eval_name=eval_file_id,

                            save_latent_space=True,
                            plot_latent_space_cols=plot_latent_space_cols,
                            plot_latent_space = 'sns',
                            
                            with_run_id=None,
                            load_model_from_run_id=None,
                            load_model_loc = False,
                            load_encoder_loc= False,

                           **kwargs)