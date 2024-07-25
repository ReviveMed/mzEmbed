# Finetune the top pretrained models using 3 sets of finetuning hyper-parameters

import os
import optuna
import pandas as pd
from prep_run import create_selected_data, make_kwargs_set, get_task_head_kwargs, round_kwargs_to_sig, flatten_dict, \
    unflatten_dict
from utils_neptune import get_latest_dataset, get_run_id_list, get_run_id_list_from_query
from setup3 import setup_neptune_run
import time
from prep_study2 import objective_func4, reuse_run, get_study_objective_keys, get_study_objective_directions, \
    add_runs_to_study
from prep_run import get_selection_df, convert_model_kwargs_list_to_dict, convert_distributions_to_suggestion

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


# %%


def main(run_id, yes_plot_latent_space=False, which_finetune_nums=[], task_name_list=[]):
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
    # eval_subset_col = 'Pretrain All'
    eval_subset_col = 'Pretrain Discovery Train'
    eval_subset_col1 = 'Pretrain Discovery Val'
    eval_subset_col2 = 'Pretrain Test'

    finetune_fit_subset_col = 'Finetune Discovery Train'
    finetune_eval_subset_col = 'Finetune Discovery Val'

    _, finetune_fit_file_id = create_selected_data(input_data_dir=input_data_dir,
                                                   sample_selection_col=finetune_fit_subset_col,
                                                   subdir_col=subdir_col,
                                                   output_dir=output_dir,
                                                   metadata_df=None,
                                                   selections_df=selections_df)

    _, finetune_eval_file_id = create_selected_data(input_data_dir=input_data_dir,
                                                    sample_selection_col=finetune_eval_subset_col,
                                                    subdir_col=subdir_col,
                                                    output_dir=output_dir,
                                                    metadata_df=None,
                                                    selections_df=selections_df)

    X_finetune_eval_file = f'{output_dir}/X_{finetune_eval_file_id}.csv'
    y_finetune_eval_file = f'{output_dir}/y_{finetune_eval_file_id}.csv'
    X_finetune_fit_file = f'{output_dir}/X_{finetune_fit_file_id}.csv'
    y_finetune_fit_file = f'{output_dir}/y_{finetune_fit_file_id}.csv'

    if yes_plot_latent_space:
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

        _, eval_file_id2 = create_selected_data(input_data_dir=input_data_dir,
                                                sample_selection_col=eval_subset_col2,
                                                subdir_col=subdir_col,
                                                output_dir=output_dir,
                                                metadata_df=None,
                                                selections_df=selections_df)

        _, eval_file_id1 = create_selected_data(input_data_dir=input_data_dir,
                                                sample_selection_col=eval_subset_col1,
                                                subdir_col=subdir_col,
                                                output_dir=output_dir,
                                                metadata_df=None,
                                                selections_df=selections_df)

        X_eval_file = f'{output_dir}/X_{eval_file_id}.csv'
        y_eval_file = f'{output_dir}/y_{eval_file_id}.csv'

        X_eval_file1 = f'{output_dir}/X_{eval_file_id1}.csv'
        y_eval_file1 = f'{output_dir}/y_{eval_file_id1}.csv'

        X_eval_file2 = f'{output_dir}/X_{eval_file_id2}.csv'
        y_eval_file2 = f'{output_dir}/y_{eval_file_id2}.csv'

        X_fit_file = f'{output_dir}/X_{fit_file_id}.csv'
        y_fit_file = f'{output_dir}/y_{fit_file_id}.csv'

        plot_latent_space_cols = []
        _ = setup_neptune_run(input_data_dir,
                              setup_id=f'pretrain',
                              project_id=project_id,

                              neptune_mode='async',
                              yes_logging=True,
                              neptune_api_token=neptune_api_token,
                              tags=['shanghai (ST002773) removed pretrain recon only'],

                              run_training=False,
                              X_fit_file=X_fit_file,
                              y_fit_file=y_fit_file,
                              train_name=fit_file_id,

                              run_evaluation=True,
                              # run_evaluation=False,
                              X_eval_file=X_eval_file,
                              y_eval_file=y_eval_file,
                              eval_name=eval_file_id,

                              save_latent_space=True,
                              plot_latent_space_cols=plot_latent_space_cols,
                              plot_latent_space='sns',

                              with_run_id=run_id,
                              # load_model_from_run_id=None,
                              load_model_loc='pretrain')

        _ = setup_neptune_run(input_data_dir,
                              setup_id=f'pretrain',
                              project_id=project_id,

                              neptune_mode='async',
                              yes_logging=True,
                              neptune_api_token=neptune_api_token,
                              tags=['shanghai (ST002773) removed pretrain recon only'],

                              run_training=False,
                              X_fit_file=X_fit_file,
                              y_fit_file=y_fit_file,
                              train_name=fit_file_id,

                              run_evaluation=True,
                              # run_evaluation=False,
                              X_eval_file=X_eval_file1,
                              y_eval_file=y_eval_file1,
                              eval_name=eval_file_id1,

                              save_latent_space=True,
                              plot_latent_space_cols=plot_latent_space_cols,
                              plot_latent_space='sns',

                              with_run_id=run_id,
                              # load_model_from_run_id=None,
                              load_model_loc='pretrain')

        _ = setup_neptune_run(input_data_dir,
                              setup_id=f'pretrain',
                              project_id=project_id,

                              neptune_mode='async',
                              yes_logging=True,
                              neptune_api_token=neptune_api_token,
                              tags=['shanghai (ST002773) removed pretrain recon only'],

                              run_training=False,
                              X_fit_file=X_fit_file,
                              y_fit_file=y_fit_file,
                              train_name=fit_file_id,

                              run_evaluation=True,
                              # run_evaluation=False,
                              X_eval_file=X_eval_file2,
                              y_eval_file=y_eval_file2,
                              eval_name=eval_file_id2,

                              save_latent_space=True,
                              plot_latent_space_cols=plot_latent_space_cols,
                              plot_latent_space='sns',

                              with_run_id=run_id,
                              # load_model_from_run_id=None,
                              load_model_loc='pretrain')

    # setup_num = 3
    # num_repeats=5
    # restart_run = True
    # restart_rand_run = True
    # remove_y_nans = True

    restart_run = False
    restart_rand_run = False
    num_repeats = 5
    upload_models_to_neptune = False

    for task_name in task_name_list:

        ### choose the Task heads
        print('Finetune on Task: ', task_name)

        plot_latent_space_cols = []
        y_head_cols = []
        y_adv_cols = []
        head_kwargs_dict = {}
        adv_kwargs_dict = {}

        if task_name == 'Both-OS':
            head_kwargs_dict['Both-OS'], y_head_cols = get_task_head_kwargs(head_kind='Cox',
                                                                            y_head_col='OS',
                                                                            y_cols=y_head_cols,
                                                                            head_name='Both-OS')

        elif task_name == 'IMDC':
            head_kwargs_dict['IMDC'], y_head_cols = get_task_head_kwargs(head_kind='Binary',
                                                                         y_head_col='IMDC BINARY',
                                                                         y_cols=[],
                                                                         head_name='IMDC')
        else:
            raise ValueError('Task not recognized')

        #### Specify the finetune kwargs
        # Summary of the finetune #6 hyperparams are: has reconstrusctuon loss, has a hidden layer in between the encoder and the head, has dropout, has encoder weight, has head weight, has weight decay
        finetune_kwargs6 = make_kwargs_set(encoder_kind=encoder_kind,
                                           head_kwargs_dict=head_kwargs_dict,
                                           num_epochs=30,
                                           batch_size=64,
                                           noise_factor=0.1,
                                           dropout_rate=0,
                                           encoder_weight=0.5,  #
                                           task_num_hidden_layers=1,
                                           head_weight=1.0,
                                           weight_decay=0.0005,
                                           learning_rate=0.0005)
        finetune_kwargs6['remove_y_nans'] = True
        finetune_kwargs6['num_repeats'] = num_repeats
        finetune_kwargs6['setup_num'] = 6

        # summary of the finetune #5 hyperparams are: has no reconstruction loss
        finetune_kwargs5 = make_kwargs_set(encoder_kind=encoder_kind,
                                           head_kwargs_dict=head_kwargs_dict,
                                           num_epochs=30,
                                           batch_size=64,
                                           noise_factor=0.1,
                                           dropout_rate=0,
                                           encoder_weight=0,
                                           task_num_hidden_layers=0,
                                           head_weight=1.0,
                                           weight_decay=0.0005,
                                           learning_rate=0.0005)
        finetune_kwargs5['remove_y_nans'] = True
        finetune_kwargs5['num_repeats'] = num_repeats
        finetune_kwargs5['setup_num'] = 5

        finetune_kwargs4 = make_kwargs_set(encoder_kind=encoder_kind,
                                           head_kwargs_dict=head_kwargs_dict,
                                           num_epochs=30,
                                           batch_size=64,
                                           noise_factor=0.1,
                                           dropout_rate=0.2,
                                           encoder_weight=0,
                                           task_num_hidden_layers=0,
                                           head_weight=1.0,
                                           weight_decay=0.0,
                                           learning_rate=0.0005)
        finetune_kwargs4['remove_y_nans'] = True
        finetune_kwargs4['num_repeats'] = num_repeats
        finetune_kwargs4['setup_num'] = 4

        finetune_kwargs_list = [finetune_kwargs4, finetune_kwargs5, finetune_kwargs6]

        ### Run the finetune
        for finetune_kwargs in finetune_kwargs_list:
            setup_num = finetune_kwargs['setup_num']
            if setup_num not in which_finetune_nums:
                continue

            print('setup_num: ', setup_num)
            print(finetune_kwargs)
            try:
                _ = setup_neptune_run(input_data_dir,
                                      setup_id=f'{task_name} finetune v{setup_num}',
                                      project_id=project_id,

                                      neptune_mode='async',
                                      yes_logging=True,
                                      neptune_api_token=neptune_api_token,
                                      tags=['shanghai (ST002773) removed finetune recon only'],
                                      y_head_cols=y_head_cols,
                                      y_adv_cols=y_adv_cols,
                                      restart_run=restart_run,

                                      run_training=True,
                                      X_fit_file=X_finetune_fit_file,
                                      y_fit_file=y_finetune_fit_file,
                                      train_name=finetune_fit_file_id,
                                      upload_models_to_neptune=upload_models_to_neptune,

                                      run_evaluation=True,
                                      X_eval_file=X_finetune_eval_file,
                                      y_eval_file=y_finetune_eval_file,
                                      eval_name=finetune_eval_file_id,

                                      save_latent_space=False,
                                      plot_latent_space_cols=plot_latent_space_cols,
                                      plot_latent_space='',

                                      with_run_id=run_id,
                                      # load_model_from_run_id=None,
                                      # load_model_loc = None,
                                      load_encoder_loc='pretrain',
                                      **finetune_kwargs)

                _ = setup_neptune_run(input_data_dir,
                                      setup_id=f'{task_name} randinit v{setup_num}',
                                      project_id=project_id,

                                      neptune_mode='async',
                                      yes_logging=True,
                                      neptune_api_token=neptune_api_token,
                                      tags=['shanghai (ST002773) removed finetune recon only'],
                                      y_head_cols=y_head_cols,
                                      y_adv_cols=y_adv_cols,
                                      # overwrite_existing_kwargs=True,
                                      restart_run=restart_rand_run,

                                      run_training=True,
                                      X_fit_file=X_finetune_fit_file,
                                      y_fit_file=y_finetune_fit_file,
                                      train_name=finetune_fit_file_id,
                                      upload_models_to_neptune=upload_models_to_neptune,

                                      run_evaluation=True,
                                      X_eval_file=X_finetune_eval_file,
                                      y_eval_file=y_finetune_eval_file,
                                      eval_name=finetune_eval_file_id,

                                      save_latent_space=False,
                                      plot_latent_space_cols=plot_latent_space_cols,
                                      plot_latent_space='',

                                      with_run_id=run_id,
                                      # load_model_from_run_id=None,
                                      # load_model_loc = None,
                                      load_encoder_loc='pretrain',
                                      run_random_init=True,
                                      **finetune_kwargs)


            except Exception as e:
                print('Error: ', e)

    return


if __name__ == '__main__':

    # plot the latent space representation for a few top models
    run_id_list = ['RCC-4722', 'RCC-4765', 'RCC-4741']
    for run_id in run_id_list:
        print('##############################################')
        print('##############################################')
        print('Running: ', run_id)
        start_time = time.time()
        main(run_id,
             yes_plot_latent_space=True,
             which_finetune_nums=[],
             # which_finetune_nums=[4,5,6],
             task_name_list=[])
        # task_name_list=['IMDC'])
        print('Minutes elapsed: ', (time.time() - start_time) / 60)

    # query to select models to finetune
    # query1 = '(`pretrain/original_kwargs/source run_id`:string = "RCC-3213") OR (`pretrain/original_kwargs/source run_id`:string = "RCC-3276")'
    # query2 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "Recon Minimize July15 v0") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.38)'
    # query3 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "Multi Obj July12v1") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.39)'
    # query4 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "MultiObj Minimize July16 v0") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.39) AND (`sys/creation_time`:datetime > "-1d")'
    # query5 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "Recon Minimize July16 v0") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.39)'

    # query2 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "Recon Minimize July15 v0") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float > 0.39) AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.4)'
    # query3 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "Multi Obj July12v1") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.4)'
    # query4 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "MultiObj Minimize July16 v0") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float > 0.39) AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.4)'
    # query5 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "Recon Minimize July16 v0") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.39)'

    query4 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "MultiObj Minimize July16 v0") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.4) AND \
        (`pretrain/avg/Pretrain_Discovery_Val MultiClass_Cohort-Label__AUROC (ovo, macro)`:float > 0.98) AND (`pretrain/avg/Pretrain_Discovery_Val Regression_Age__MAE`:float < 9.0) AND \
        (`pretrain/avg/Pretrain_Discovery_Val Binary_is-Pediatric__AUROC (micro)`:float > 0.98) AND (`pretrain/avg/Pretrain_Discovery_Val Regression_BMI__MAE`:float < 4.8) AND \
        (`pretrain/avg/Pretrain_Discovery_Val Binary_Sex__AUROC (micro)`:float > 0.9) AND (`pretrain/avg/Pretrain_Discovery_Val Binary_Smoking__AUROC (micro)`:float > 0.95)'

    # query_list = [query2,query3,query4,query5]
    # query_list = [query4]
    # run_id_list = []
    # for query in query_list:
    #     run_id_list += get_run_id_list_from_query(query=query, limit=100, project_id=project_id)

    # run_id_list = ['RCC-3610','RCC-3537','RCC-3597','RCC-3370','RCC-3345']

    print('number of models to finetune: ', len(run_id_list))
    # exit()
    for run_id in run_id_list:
        print('##############################################')
        print('##############################################')
        print('Running: ', run_id)
        start_time = time.time()
        main(run_id,
             yes_plot_latent_space=False,
             # which_finetune_nums=[],
             which_finetune_nums=[4, 5, 6],
             # task_name_list=[])
             task_name_list=['IMDC', 'Both-OS'])
        print('Minutes elapsed: ', (time.time() - start_time) / 60)
        # break
