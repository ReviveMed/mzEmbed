
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

fit_subset_col = 'Finetune Discovery Train'
eval_subset_col = 'Finetune Discovery Val'
setup_id = 'both-OS finetune v0'

use_rand_init = False
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

                                                     

# head_kwargs_dict['IMDC'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
#                                                      y_head_col='IMDC BINARY',
#                                                      y_cols=y_head_cols,
#                                                      head_name='IMDC')

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


# %%
# encoder_kind = 'VAE'
# kwargs = make_kwargs_set(sig_figs=2,
#                 encoder_kind=encoder_kind,
#                 activation_func= 'leakyrelu',
#                 use_batch_norm= False,
#                 head_kwargs_dict=head_kwargs_dict,
#                 adv_kwargs_dict=adv_kwargs_dict,
#                 num_epochs=38,
#                 latent_size=108,
#                 hidden_size_mult=1.5,
#                 hidden_size=-1,
#                 num_hidden_layers=3,
#                 task_head_weight=-1,
#                 task_num_hidden_layers=0,
#                 encoder_weight=0.75,
#                 weight_decay=0,
#                 head_weight=1,
#                 learning_rate=0.0021,
#                 noise_factor=0.25,
#                 adv_weight=0.0,
#                 dropout_rate=0.4)

vnum = 0
num_repeats = 5
encoder_kind = 'VAE'
kwargs = make_kwargs_set(sig_figs=2,
                encoder_kind=encoder_kind,
                activation_func= 'leakyrelu',
                use_batch_norm= False,
                head_kwargs_dict=head_kwargs_dict,
                adv_kwargs_dict=adv_kwargs_dict,
                batch_size=64,
                num_epochs=70,
                latent_size=108,
                hidden_size_mult=1.5,
                hidden_size=-1,
                num_hidden_layers=3,
                task_num_hidden_layers=0,
                task_head_weight=-1,
                encoder_weight=1,
                weight_decay=0,
                head_weight=1,
                learning_rate=0.0005,
                noise_factor=0.2,
                adv_weight=0.0)

# head_kwargs_dict = {}
# adv_kwargs_dict = {}
# encoder_kind = 'metabFoundation'
# kwargs = make_kwargs_set(sig_figs=2,
#                 encoder_kind=encoder_kind,
#                 activation_func= 'leakyrelu',
#                 use_batch_norm= False,
#                 head_kwargs_dict=head_kwargs_dict,
#                 adv_kwargs_dict=adv_kwargs_dict,
#                 num_attention_heads=2,
#                 num_decoder_layers=2,
#                 embed_dim = 32,
#                 decoder_embed_dim = 8,
#                 default_hidden_fraction = 0.2,
#                 num_epochs=150,
#                 num_hidden_layers = 4,
#                 task_head_weight=-1,
#                 encoder_weight= 1,
#                 weight_decay=0.00001,
#                 head_weight=0,
#                 learning_rate=0.0005,
#                 noise_factor=0.1,
#                 adv_weight=0.0)

# %%
with_run_id = 'RCC-3183'

setup_id = f'both-OS finetune v{vnum}'

use_rand_init = False
remove_y_nans = False

plot_latent_space_cols  = list(set(y_head_cols + y_adv_cols))
# with_run_id = 'RCC-3188'
run_id = setup_neptune_run(input_data_dir,
                            setup_id=setup_id,
                            project_id=project_id,

                            neptune_mode='async',
                            yes_logging = True,
                            neptune_api_token=NEPTUNE_API_TOKEN,
                            tags=['v4'],
                            y_head_cols=y_head_cols,
                            y_adv_cols=y_adv_cols,
                            num_repeats=num_repeats,

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
                            
                            with_run_id=with_run_id,
                            # load_model_from_run_id=None,
                            load_model_loc = None,
                            load_encoder_loc= 'pretrain',

                           **kwargs)



setup_id = f'both-OS randinit v{vnum}'

use_rand_init = True
remove_y_nans = False

plot_latent_space_cols  = list(set(y_head_cols + y_adv_cols))
# with_run_id = 'RCC-3188'
run_id = setup_neptune_run(input_data_dir,
                            setup_id=setup_id,
                            project_id=project_id,

                            neptune_mode='async',
                            yes_logging = True,
                            neptune_api_token=NEPTUNE_API_TOKEN,
                            tags=['v4'],
                            y_head_cols=y_head_cols,
                            y_adv_cols=y_adv_cols,
                            num_repeats=num_repeats,

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
                            
                            with_run_id=with_run_id,
                            # load_model_from_run_id=None,
                            load_model_loc = None,
                            load_encoder_loc= 'pretrain',

                           **kwargs)

