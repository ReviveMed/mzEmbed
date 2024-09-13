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

STUDY_DICT = {
    'study_name': 'RCC all data shanghai removed segmented',
    'encoder_kind': encoder_kind,
    'objectives': {
        'reconstruction_loss': {
            'weight': 3,
            'name': 'Reconstruction Loss',
            'direction': 'minimize',
            'transform': 'log10',
            'default_value': 9999
        },
        'Binary_is-Pediatric__AUROC (micro)': {
            'weight': 1,
            'name': 'Pediatric Prediction',
            'direction': 'maximize',
            'default_value': 0,
        },
        'MultiClass_Cohort-Label__AUROC (ovo, macro)': {
            'weight': 1,
            'name': 'Cohort Label Prediction',
            'direction': 'maximize',
            'default_value': 0
        },

        'Binary_Sex__AUROC (micro)': {
            'weight': 2,
            'name': 'Gender Prediction',
            'direction': 'maximize',
            'default_value': 0
        },
        'Regression_Age__MAE': {
            'weight': 2,
            'name': 'Age Prediction',
            'direction': 'minimize',
            'transform': 'log10',
            'default_value': 9999
        },
        'Regression_BMI__MAE': {
            'weight': 2,
            'name': 'BMI Prediction',
            'direction': 'minimize',
            'transform': 'log10',
            'default_value': 9999
        },
        'Binary_Smoking__AUROC (micro)': {
            'weight': 1,
            'name': 'Smoking Prediction',
            'direction': 'maximize',
            'default_value': 0
        }
    }
}


def get_study_kwargs(head_kwargs_dict, adv_kwargs_dict,
                     latent_size_min=100, latent_size_max=254,
                     num_hidden_layers_min=2, num_hidden_layers_max=4):
    kwargs = make_kwargs_set(encoder_kind=encoder_kind,
                             head_kwargs_dict=head_kwargs_dict,
                             adv_kwargs_dict=adv_kwargs_dict,

                             # vae model paremeters
                             latent_size=None, latent_size_min=latent_size_min, latent_size_max=latent_size_max,
                             latent_size_step=4,
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
                             num_epochs=None, num_epochs_min=50, num_epochs_max=400, num_epochs_step=25,
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
         latent_size_min=100, latent_size_max=254,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='tag'):
    neptune_api_token = NEPTUNE_API_TOKEN
    homedir = os.path.expanduser("~")
    input_data_dir = f'{homedir}/INPUT_DATA'
    os.makedirs(input_data_dir, exist_ok=True)
    input_data_dir = get_latest_dataset(data_dir=input_data_dir, api_token=NEPTUNE_API_TOKEN, project=project_id)

    selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv', index_col=0)

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

    # Determine the Task Heads
    plot_latent_space_cols = []
    y_head_cols = []
    y_adv_cols = []
    head_kwargs_dict = {}
    adv_kwargs_dict = {}

    head_kwargs_dict['Cohort-Label'], y_head_cols = get_task_head_kwargs(head_kind='MultiClass',
                                                                         y_head_col='Cohort Label v0',
                                                                         y_cols=y_head_cols,
                                                                         head_name='Cohort-Label',
                                                                         num_classes=4,
                                                                         default_weight=5.4)

    head_kwargs_dict['is-Pediatric'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
                                                                         y_head_col='is Pediatric',
                                                                         y_cols=y_head_cols,
                                                                         head_name='is-Pediatric',
                                                                         default_weight=2.6)

    head_kwargs_dict['Age'], y_head_cols = get_task_head_kwargs(head_kind='Regression',
                                                                y_head_col='Age',
                                                                y_cols=y_head_cols,
                                                                head_name='Age',
                                                                default_weight=7.5)

    head_kwargs_dict['Sex'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
                                                                y_head_col='Sex',
                                                                y_cols=y_head_cols,
                                                                head_name='Sex',
                                                                default_weight=13)

    head_kwargs_dict['BMI'], y_head_cols = get_task_head_kwargs(head_kind='Regression',
                                                                y_head_col='BMI',
                                                                y_cols=y_head_cols,
                                                                head_name='BMI',
                                                                default_weight=7.5)

    head_kwargs_dict['Smoking'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
                                                                    y_head_col='Smoking Status',
                                                                    y_cols=y_head_cols,
                                                                    head_name='Smoking',
                                                                    default_weight=0)

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

            run_id = setup_neptune_run(input_data_dir,
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
    # 120-140
    main(STUDY_DICT, num_trials=50,
         latent_size_min=120, latent_size_max=140,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )
    # 140-160
    main(STUDY_DICT, num_trials=50,
         latent_size_min=120, latent_size_max=160,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )
    # 160-180
    main(STUDY_DICT, num_trials=50,
         latent_size_min=160, latent_size_max=180,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )
    # 180-200
    main(STUDY_DICT, num_trials=50,
         latent_size_min=180, latent_size_max=200,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )
    # 200-220
    main(STUDY_DICT, num_trials=50,
         latent_size_min=200, latent_size_max=220,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )
    # 220-240
    main(STUDY_DICT, num_trials=50,
         latent_size_min=220, latent_size_max=240,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )
    # 240-260
    main(STUDY_DICT, num_trials=50,
         latent_size_min=240, latent_size_max=260,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )
    # 260-280
    main(STUDY_DICT, num_trials=50,
         latent_size_min=260, latent_size_max=280,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )
    # 280-300
    main(STUDY_DICT, num_trials=50,
         latent_size_min=280, latent_size_max=300,
         num_hidden_layers_min=2, num_hidden_layers_max=4,
         tag='segmented pretrain shanghai removed Aug 10'
         )



    # res = objective_func4('RCC-3188',
    #                 study_info_dict=STUDY_DICT,
    #                 project_id=project_id,
    #                 neptune_api_token=NEPTUNE_API_TOKEN,
    #                 setup_id='pretrain',
    #                 eval_file_id='Pretrain_Discovery_Val')

    # print(res)
