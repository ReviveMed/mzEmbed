import os
import optuna
import pandas as pd
from pretrain.prep_run import create_selected_data, make_kwargs_set, get_task_head_kwargs, round_kwargs_to_sig, flatten_dict, \
    unflatten_dict
from utils.utils_neptune import get_latest_dataset, get_run_id_list
from pretrain.setup3 import setup_neptune_run

from pretrain.prep_study2 import objective_func4, reuse_run, get_study_objective_keys, get_study_objective_directions, \
    add_runs_to_study
from pretrain.prep_run import get_selection_df, convert_model_kwargs_list_to_dict, convert_distributions_to_suggestion
# import torch_cpu_loader

## 
# %% Load the latest data

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZDlmZGM4ZC05OGM2LTQ2YzctYmRhNi0zMjIwODMzMWM1ODYifQ=='
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

project_id = 'revivemed/RCC'

USE_WEBAPP_DB = True
SAVE_TRIALS = True
ADD_EXISTING_RUNS_TO_STUDY = False
limit_add = -1  # limit the number of runs added to the study

encoder_kind = 'VAE'
tag_study='pretrain-hidden-layer_2-4-latent-size_test'

STUDY_DICT = {
    # oputuna study parameters
    # study name is the optuna optimization study name
    'study_name': tag_study,
    'encoder_kind': encoder_kind,
    'objectives': {
        'reconstruction_loss': {
            'weight': 3,
            'name': 'Reconstruction Loss',
            'direction': 'minimize',
            'default_value': 9999
        }
    }
}


def get_study_kwargs(head_kwargs_dict, adv_kwargs_dict,
                     latent_size_min=128, latent_size_max=512,
                     num_hidden_layers_min=2, num_hidden_layers_max=4):
    kwargs = make_kwargs_set(encoder_kind=encoder_kind,
                             head_kwargs_dict=head_kwargs_dict,
                             adv_kwargs_dict=adv_kwargs_dict,

                             # vae model paremeters
                             # if you want to skip tuning, change the None the desired value
                             # -1 means the parameter was not used
                             # fixed value is fixed to the value
                             latent_size=None, latent_size_min=latent_size_min, latent_size_max=latent_size_max,
                             latent_size_step=16,
                             hidden_size=-1, hidden_size_min=16, hidden_size_max=64, hidden_size_step=16,
                             hidden_size_mult=1.5, hidden_size_mult_min=1.25, hidden_size_mult_max=2,
                             hidden_size_mult_step=0.25,
                             num_hidden_layers=None, num_hidden_layers_min=num_hidden_layers_min,
                             num_hidden_layers_max=num_hidden_layers_max,
                             num_hidden_layers_step=1,

                             # metabol-transformer model parameters
                             num_attention_heads=-1, num_attention_heads_min=1, num_attention_heads_max=5,
                             num_attention_heads_step=1,
                             num_decoder_layers=-1, num_decoder_layers_min=1, num_decoder_layers_max=5,
                             num_decoder_layers_step=1,
                             embed_dim=-1, embed_dim_min=4, embed_dim_max=64, embed_dim_step=4,
                             decoder_embed_dim=-1, decoder_embed_dim_min=4, decoder_embed_dim_max=64,
                             decoder_embed_dim_step=4,
                             default_hidden_fraction=-1, default_hidden_fraction_min=0, default_hidden_fraction_max=0.5,
                             default_hidden_fraction_step=0.1,

                             dropout_rate=0, dropout_rate_min=0, dropout_rate_max=0.5, dropout_rate_step=0.1,
                             encoder_weight=1.0, encoder_weight_min=0, encoder_weight_max=2, encoder_weight_step=0.5,
                             head_weight=1.0, head_weight_min=0, head_weight_max=2, head_weight_step=0.5,
                             adv_weight=1.0, adv_weight_min=0, adv_weight_max=2, adv_weight_step=0.5,

                             task_head_weight=None, task_head_weight_min=0, task_head_weight_max=15,
                             task_head_weight_step=0.05,
                             task_adv_weight=-1, task_adv_weight_min=0, task_adv_weight_max=10,
                             task_adv_weight_step=0.1,
                             task_hidden_size=4, task_hidden_size_min=4, task_hidden_size_max=64,
                             task_hidden_size_step=4,
                             task_num_hidden_layers=1, task_num_hidden_layers_min=1, task_num_hidden_layers_max=3,
                             task_num_hidden_layers_step=1,

                             weight_decay=None, weight_decay_min=1e-5, weight_decay_max=1e-2, weight_decay_step=0.00001,
                             weight_decay_log=True,
                             l1_reg_weight=0, l1_reg_weight_min=0, l1_reg_weight_max=0.01, l1_reg_weight_step=0.0001,
                             l1_reg_weight_log=False,
                             l2_reg_weight=0, l2_reg_weight_min=0, l2_reg_weight_max=0.01, l2_reg_weight_step=0.0001,
                             l2_reg_weight_log=False,

                             batch_size=96, batch_size_min=32, batch_size_max=128, batch_size_step=32,
                             noise_factor=None, noise_factor_min=0, noise_factor_max=0.25, noise_factor_step=0.05,
                             num_epochs=100, num_epochs_min=50, num_epochs_max=400, num_epochs_step=25,
                             num_epochs_log=False,
                             learning_rate=None, learning_rate_min=0.00001, learning_rate_max=0.005,
                             learning_rate_step=None, learning_rate_log=True,
                             early_stopping_patience=0, early_stopping_patience_min=0, early_stopping_patience_max=50,
                             early_stopping_patience_step=5,
                             adversarial_start_epoch=0, adversarial_start_epoch_min=-1, adversarial_start_epoch_max=10,
                             adversarial_start_epoch_step=2,
                             )
    return kwargs


def main(STUDY_INFO_DICT, num_trials=5,
         latent_size_min=128, latent_size_max=512,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='tag'):
    neptune_api_token = NEPTUNE_API_TOKEN
    homedir = os.path.expanduser("~")
    input_data_dir = f'{homedir}/INPUT_DATA'
    os.makedirs(input_data_dir, exist_ok=True)
    input_data_dir = get_latest_dataset(data_dir=input_data_dir, api_token=NEPTUNE_API_TOKEN, project=project_id)

    selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv', index_col=0)

    output_dir = f'{homedir}/PROCESSED_DATA'
    PROCESSED_DATA_dir=output_dir
    os.makedirs(output_dir, exist_ok=True)
    subdir_col = 'Study ID'

    fit_subset_col = 'Pretrain Discovery Train' #Finetune_Discovery_Train
    eval_subset_col = 'Pretrain Discovery Val'
    test_subset_col = 'Pretrain Test'
    setup_id = 'pretrain'

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
    
    _, test_file_id = create_selected_data(input_data_dir=input_data_dir,
                                           sample_selection_col=test_subset_col,
                                           subdir_col=subdir_col,
                                           output_dir=output_dir,
                                           metadata_df=None,
                                           selections_df=selections_df)
    


    X_eval_file = f'{output_dir}/X_{eval_file_id}.csv'
    y_eval_file = f'{output_dir}/y_{eval_file_id}.csv'
    X_fit_file = f'{output_dir}/X_{fit_file_id}.csv'
    y_fit_file = f'{output_dir}/y_{fit_file_id}.csv'
    X_test_file = f'{output_dir}/X_{test_file_id}.csv'
    y_test_file = f'{output_dir}/y_{test_file_id}.csv'

    # Determine the Task Heads
    plot_latent_space_cols = []
    y_head_cols = []
    y_adv_cols = []
    head_kwargs_dict = {}
    adv_kwargs_dict = {}

    def compute_objective(run_id):
        return objective_func4(run_id,
                               study_info_dict=STUDY_INFO_DICT,
                               project_id=project_id,
                               neptune_api_token=neptune_api_token,
                               setup_id=setup_id,
                               eval_file_id=eval_file_id)

    def objective(trial):

        try:
            kwargs = get_study_kwargs(head_kwargs_dict, adv_kwargs_dict,
                                      latent_size_min=latent_size_min, latent_size_max=latent_size_max,
                                      num_hidden_layers_min=num_hidden_layers_min,
                                      num_hidden_layers_max=num_hidden_layers_max)

            kwargs = convert_model_kwargs_list_to_dict(kwargs)
            kwargs = flatten_dict(kwargs)  # flatten the dict for optuna compatibility
            kwargs = convert_distributions_to_suggestion(kwargs,
                                                         trial)  # convert the distributions to optuna suggestions
            kwargs = round_kwargs_to_sig(kwargs, sig_figs=2)
            kwargs = unflatten_dict(kwargs)  # unflatten the dict for the setup function

            kwargs['study_info_dict'] = STUDY_INFO_DICT

            run_id = setup_neptune_run(PROCESSED_DATA_dir, input_data_dir,
                                       setup_id=setup_id,
                                       project_id=project_id,

                                       neptune_mode='async',
                                       yes_logging=True,
                                       neptune_api_token=neptune_api_token,
                                       tags=[tag],
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

                                       X_test_file=X_test_file,
                                       y_test_file=y_test_file,
                                       test_name=test_file_id,

                                       save_latent_space=True,
                                       plot_latent_space_cols=plot_latent_space_cols,
                                       plot_latent_space='sns',

                                       # with_run_id=with_run_id,
                                       # load_model_from_run_id=None,
                                       # load_model_loc = None,
                                       # load_encoder_loc= 'pretrain',

                                       **kwargs)

            trial.set_user_attr('run_id', run_id)
            trial.set_user_attr('setup_id', setup_id)

            return compute_objective(run_id)

        # except Exception as e:
        except ValueError as e:
            print(e)
            # return float('nan')
            raise optuna.TrialPruned()

    if USE_WEBAPP_DB:
        print('using webapp database')
        storage_name = WEBAPP_DB_LOC

    if 'study_name' in STUDY_INFO_DICT:
        if 'encoder_kind' in STUDY_INFO_DICT:
            study_name = STUDY_INFO_DICT['study_name'] + f' {STUDY_INFO_DICT["encoder_kind"]}'
        else:
            study_name = STUDY_INFO_DICT['study_name']
    else:
        study_name = f'{encoder_kind} Study'

    study = optuna.create_study(directions=get_study_objective_directions(STUDY_INFO_DICT),
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)

    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    # 250-350
    main(STUDY_DICT, num_trials=200,
         latent_size_min=128, latent_size_max=512,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         # change neptune run tag to keep track of the runs
         tag=tag_study
         )