
import os
import pandas as pd
from prep_run import create_selected_data, make_kwargs_set, get_task_head_kwargs
from utils_neptune import get_latest_dataset
from setup3 import setup_neptune_run


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

# fit_subset_col = 'Pretrain Discovery Train'
# eval_subset_col = 'Pretrain Discovery Val'
# eval_subset_col  = 'Pretrain All'
# eval_subset_col = 'Pretrain Test'
# setup_id = 'pretrain'

fit_subset_col = 'Finetune Discovery Train'
eval_subset_col = 'Finetune Discovery Val'
setup_id = 'finetune v0'

with_run_id = 'RCC-3499'


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

# Determine the Task Heads
plot_latent_space_cols = []
y_head_cols = []
y_adv_cols = []
head_kwargs_dict = {}
adv_kwargs_dict = {}

# head_kwargs_dict['Cohort-Label'], y_head_cols = get_task_head_kwargs(head_kind='MultiClass',
#                                                      y_head_col='Cohort Label v0',
#                                                      y_cols=y_head_cols,
#                                                      head_name='Cohort-Label',
#                                                      num_classes=4,
#                                                      default_weight=5.4)

# head_kwargs_dict['is-Pediatric'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
#                                                      y_head_col='is Pediatric',
#                                                      y_cols=y_head_cols,
#                                                      head_name='is-Pediatric',
#                                                      default_weight=2.6)

# head_kwargs_dict['Age'], y_head_cols = get_task_head_kwargs(head_kind='Regression',
#                                                      y_head_col='Age',
#                                                      y_cols=y_head_cols,
#                                                      head_name='Age',
#                                                      default_weight=7.5)

# head_kwargs_dict['Sex'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
#                                                      y_head_col='Sex',
#                                                      y_cols=y_head_cols,
#                                                      head_name='Sex',
#                                                      default_weight=13)

# head_kwargs_dict['BMI'], y_head_cols = get_task_head_kwargs(head_kind='Regression',
#                                                      y_head_col='BMI',
#                                                      y_cols=y_head_cols,
#                                                      head_name='BMI',
#                                                      default_weight=7.5)

                                                     

head_kwargs_dict['IMDC'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
                                                     y_head_col='IMDC BINARY',
                                                     y_cols=y_head_cols,
                                                     head_name='IMDC')

# head_kwargs_dict['Both-OS'], y_head_cols = get_task_head_kwargs(head_kind='Cox',
#                                                      y_head_col='OS',
#                                                      y_cols=y_head_cols,
#                                                      head_name='Both-OS')

# head_kwargs_dict['NIVO-OS'], y_head_cols = get_task_head_kwargs(head_kind='Cox',
#                                                      y_head_col='NIVO OS',
#                                                      y_cols=y_head_cols,
#                                                      head_name='NIVO-OS')



# adv_kwargs_dict['Study ID'], y_adv_cols = get_task_head_kwargs(head_kind='MultiClass',
#                                                      y_head_col='Study ID',
#                                                      y_cols=y_adv_cols,
#                                                      head_name='Study ID',
#                                                      num_classes=22)

# adv_kwargs_dict['EVER-OS'], y_adv_cols = get_task_head_kwargs(head_kind='Cox',
#                                                      y_head_col='EVER OS',
#                                                      y_cols=y_adv_cols,
#                                                      head_name='EVER-OS')

print(y_head_cols)
# %%

encoder_kind = 'VAE'
finetune_kwargs = make_kwargs_set(sig_figs=2,
                encoder_kind=encoder_kind,
                activation_func= 'leakyrelu',
                use_batch_norm= False,
                head_kwargs_dict=head_kwargs_dict,
                adv_kwargs_dict=adv_kwargs_dict,
                batch_size=64,
                num_epochs=30,
                # latent_size=108,
                # hidden_size_mult=1.5,
                # hidden_size=-1,
                # num_hidden_layers=3,
                task_head_weight=1,
                encoder_weight=0,
                weight_decay=0.00008,
                head_weight=1,
                learning_rate=0.0011,
                noise_factor=0.1,
                dropout_rate=0.2,
                adv_weight=0.0)

# %%

# plot_latent_space_cols  = list(set(y_head_cols + y_adv_cols))
plot_latent_space_cols = ['IMDC','IMDC BINARY']

run_id = setup_neptune_run(input_data_dir,
                            setup_id=setup_id,
                            project_id=project_id,

                            neptune_mode='async',
                            yes_logging = True,
                            neptune_api_token=NEPTUNE_API_TOKEN,
                            tags=['v4'],
                            y_head_cols=y_head_cols,
                            y_adv_cols=y_adv_cols,
                            num_repeats=1, #number of times to repeat the finetune/eval process. model is only saved at the last iteration

                            ### Training ###
                            run_training=True, #finetune the model
                            X_fit_file=X_fit_file,
                            y_fit_file=y_fit_file,
                            train_name=fit_file_id,

                            ### Evaluation ###
                            run_evaluation=True, #evaluate the model
                            X_eval_file=X_eval_file,
                            y_eval_file=y_eval_file,
                            eval_name=eval_file_id,

                            ### latent space representations of the evaluation dataset ###
                            save_latent_space=True, # save a csv file of the umap latent space coordinates
                            plot_latent_space_cols=plot_latent_space_cols, # which columns of the metadata to color the latent space by
                            plot_latent_space = 'sns', # which plotting library to use, options are 'sns' or 'plotly' or 'both'
                            
                            ### Load pre-existing model ###
                            with_run_id=None, #continue an existing run
                            load_model_from_run_id=with_run_id, #create a new neptune run, but load data from an existing run
                            load_model_loc = None, #from where to load the full model (which directory in a run, for example "pretrain")
                            load_encoder_loc= 'pretrain', #from where to load the encoder model (which directory in a run for example "pretrain")
                            run_random_init = True, # if true, it will load the model architecture from the run, but random intialize the weights
                           **finetune_kwargs)
                            



