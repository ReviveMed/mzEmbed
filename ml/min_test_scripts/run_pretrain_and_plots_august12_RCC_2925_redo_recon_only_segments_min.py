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

    """
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
    """

    X_finetune_eval_file = f'{output_dir}/X_finetune_val.csv'
    y_finetune_eval_file = f'{output_dir}/y_finetune_val.csv'
    X_finetune_fit_file = f'{output_dir}/X_finetune_train.csv'
    y_finetune_fit_file = f'{output_dir}/y_finetune_train.csv'

    if yes_plot_latent_space:
        """
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
        """

        X_eval_file = f'{output_dir}/X_pretrain_train.csv'
        y_eval_file = f'{output_dir}/y_pretrain_train.csv'

        X_eval_file1 = f'{output_dir}/X_pretrain_val.csv'
        y_eval_file1 = f'{output_dir}/y_pretrain_val.csv'

        X_eval_file2 = f'{output_dir}/X_pretrain_test.csv'
        y_eval_file2 = f'{output_dir}/y_pretrain_test.csv'

        X_fit_file = f'{output_dir}/X_pretrain_train.csv'
        y_fit_file = f'{output_dir}/y_pretrain_train.csv'

        # plot_latent_space_cols = ['Cohort Label v0', 'Study ID', 'is Pediatric', 'Age', 'Sex']
        plot_latent_space_cols = []
        """
        _ = setup_neptune_run(input_data_dir,
                              setup_id=f'pretrain',
                              project_id=project_id,

                              neptune_mode='async',
                              yes_logging=True,
                              neptune_api_token=neptune_api_token,
                              tags=['add test'],

                              run_training=False,
                              X_fit_file=X_fit_file,
                              y_fit_file=y_fit_file,
                              train_name='pretrain_train',

                              run_evaluation=True,
                              # run_evaluation=False,
                              X_eval_file=X_eval_file,
                              y_eval_file=y_eval_file,
                              eval_name='pretrain_train',

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
                              tags=['add test'],

                              run_training=False,
                              X_fit_file=X_fit_file,
                              y_fit_file=y_fit_file,
                              train_name='pretrain_train',

                              run_evaluation=True,
                              # run_evaluation=False,
                              X_eval_file=X_eval_file1,
                              y_eval_file=y_eval_file1,
                              eval_name='pretrain_val',

                              save_latent_space=True,
                              plot_latent_space_cols=plot_latent_space_cols,
                              plot_latent_space='sns',

                              with_run_id=run_id,
                              # load_model_from_run_id=None,
                              load_model_loc='pretrain')
        """
        _ = setup_neptune_run(input_data_dir,
                              setup_id=f'pretrain',
                              project_id=project_id,

                              neptune_mode='async',
                              yes_logging=True,
                              neptune_api_token=neptune_api_token,
                              tags=['add test'],

                              run_training=False,
                              X_fit_file=X_fit_file,
                              y_fit_file=y_fit_file,
                              train_name='pretrain_train',

                              run_evaluation=True,
                              # run_evaluation=False,
                              X_eval_file=X_eval_file2,
                              y_eval_file=y_eval_file2,
                              eval_name='pretrain_test',

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
                                      tags=['RCC-2925 redo using original data finetune'],
                                      y_head_cols=y_head_cols,
                                      y_adv_cols=y_adv_cols,
                                      restart_run=restart_run,

                                      run_training=True,
                                      X_fit_file=X_finetune_fit_file,
                                      y_fit_file=y_finetune_fit_file,
                                      train_name='finetune_train',
                                      upload_models_to_neptune=upload_models_to_neptune,

                                      run_evaluation=True,
                                      X_eval_file=X_finetune_eval_file,
                                      y_eval_file=y_finetune_eval_file,
                                      eval_name='finetune_val',

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
                                      tags=['RCC-2925 redo using original data finetune'],
                                      y_head_cols=y_head_cols,
                                      y_adv_cols=y_adv_cols,
                                      # overwrite_existing_kwargs=True,
                                      restart_run=restart_rand_run,

                                      run_training=True,
                                      X_fit_file=X_finetune_fit_file,
                                      y_fit_file=y_finetune_fit_file,
                                      train_name='finetune_train',
                                      upload_models_to_neptune=upload_models_to_neptune,

                                      run_evaluation=True,
                                      X_eval_file=X_finetune_eval_file,
                                      y_eval_file=y_finetune_eval_file,
                                      eval_name='finetune_val',

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
    run_id_list = [
        'RCC-9166', 'RCC-9162', 'RCC-9159', 'RCC-9158', 'RCC-9155', 'RCC-9151',
        'RCC-9150', 'RCC-9146', 'RCC-9143', 'RCC-9141', 'RCC-9138', 'RCC-9134',
        'RCC-9132', 'RCC-9130', 'RCC-9126', 'RCC-9124', 'RCC-9123', 'RCC-9120',
        'RCC-9116', 'RCC-9114', 'RCC-9112', 'RCC-9109', 'RCC-9107', 'RCC-9104',
        'RCC-9102', 'RCC-9099', 'RCC-9096', 'RCC-9095', 'RCC-9093', 'RCC-9090',
        'RCC-9088', 'RCC-9085', 'RCC-9083', 'RCC-9079', 'RCC-9075', 'RCC-9073',
        'RCC-9071', 'RCC-9068', 'RCC-9066', 'RCC-9065', 'RCC-9063', 'RCC-9060',
        'RCC-9057', 'RCC-9056', 'RCC-9051', 'RCC-9049', 'RCC-9047', 'RCC-9046',
        'RCC-9042', 'RCC-9040', 'RCC-9038', 'RCC-9036', 'RCC-9034', 'RCC-9030',
        'RCC-9028', 'RCC-9027', 'RCC-9024', 'RCC-9021', 'RCC-9019', 'RCC-9017',
        'RCC-9014', 'RCC-9010', 'RCC-9009', 'RCC-9007', 'RCC-9005', 'RCC-9003',
        'RCC-8999', 'RCC-8996', 'RCC-8995', 'RCC-8991', 'RCC-8989', 'RCC-8987',
        'RCC-8984', 'RCC-8980', 'RCC-8979', 'RCC-8976', 'RCC-8974', 'RCC-8971',
        'RCC-8968', 'RCC-8966', 'RCC-8963', 'RCC-8962', 'RCC-8958', 'RCC-8957',
        'RCC-8953', 'RCC-8950', 'RCC-8949', 'RCC-8947', 'RCC-8944', 'RCC-8941',
        'RCC-8938', 'RCC-8935', 'RCC-8933', 'RCC-8930', 'RCC-8929', 'RCC-8927',
        'RCC-8925', 'RCC-8922', 'RCC-8919', 'RCC-8917', 'RCC-8915', 'RCC-8913',
        'RCC-8910', 'RCC-8908', 'RCC-8905', 'RCC-8902', 'RCC-8900', 'RCC-8898',
        'RCC-8897', 'RCC-8894', 'RCC-8890', 'RCC-8889', 'RCC-8887', 'RCC-8886',
        'RCC-8883', 'RCC-8880', 'RCC-8878', 'RCC-8876', 'RCC-8874', 'RCC-8870',
        'RCC-8869', 'RCC-8866', 'RCC-8865', 'RCC-8862', 'RCC-8860', 'RCC-8856',
        'RCC-8855', 'RCC-8851', 'RCC-8850', 'RCC-8846', 'RCC-8843', 'RCC-8841',
        'RCC-8839', 'RCC-8837', 'RCC-8835', 'RCC-8833', 'RCC-8830', 'RCC-8827',
        'RCC-8825', 'RCC-8823', 'RCC-8821', 'RCC-8819', 'RCC-8815', 'RCC-8813',
        'RCC-8810', 'RCC-8808', 'RCC-8806', 'RCC-8803', 'RCC-8802', 'RCC-8800',
        'RCC-8799', 'RCC-8796', 'RCC-8793', 'RCC-8791', 'RCC-8788', 'RCC-8786',
        'RCC-8783', 'RCC-8780', 'RCC-8778', 'RCC-8775', 'RCC-8773', 'RCC-8769',
        'RCC-8767', 'RCC-8763', 'RCC-8760', 'RCC-8758', 'RCC-8756', 'RCC-8754',
        'RCC-8751', 'RCC-8749', 'RCC-8747', 'RCC-8745', 'RCC-8742', 'RCC-8738',
        'RCC-8736', 'RCC-8734', 'RCC-8732', 'RCC-8729', 'RCC-8727', 'RCC-8723',
        'RCC-8722', 'RCC-8719', 'RCC-8717', 'RCC-8714', 'RCC-8711', 'RCC-8710',
        'RCC-8706', 'RCC-8704', 'RCC-8702', 'RCC-8701', 'RCC-8698', 'RCC-8697',
        'RCC-8694', 'RCC-8689', 'RCC-8688', 'RCC-8686', 'RCC-8684', 'RCC-8681',
        'RCC-8680', 'RCC-8678', 'RCC-8676', 'RCC-8674', 'RCC-8671', 'RCC-8669',
        'RCC-8665', 'RCC-8663', 'RCC-8662', 'RCC-8659', 'RCC-8654', 'RCC-8653',
        'RCC-8650', 'RCC-8648', 'RCC-8644', 'RCC-8642', 'RCC-8640', 'RCC-8637',
        'RCC-8634', 'RCC-8632', 'RCC-8630', 'RCC-8628', 'RCC-8625', 'RCC-8623',
        'RCC-8619', 'RCC-8617', 'RCC-8614', 'RCC-8613', 'RCC-8607', 'RCC-8606',
        'RCC-8604', 'RCC-8602', 'RCC-8598', 'RCC-8595', 'RCC-8594', 'RCC-8591',
        'RCC-8589', 'RCC-8587', 'RCC-8585', 'RCC-8582', 'RCC-8580', 'RCC-8578',
        'RCC-8573', 'RCC-8571', 'RCC-8569', 'RCC-8567', 'RCC-8564', 'RCC-8562',
        'RCC-8561', 'RCC-8559', 'RCC-8557', 'RCC-8553', 'RCC-8549', 'RCC-8548',
        'RCC-8544', 'RCC-8538', 'RCC-8536', 'RCC-8534', 'RCC-8532', 'RCC-8529',
        'RCC-8528', 'RCC-8524', 'RCC-8522', 'RCC-8520', 'RCC-8516', 'RCC-8513',
        'RCC-8511', 'RCC-8509', 'RCC-8506', 'RCC-8505', 'RCC-8502', 'RCC-8501',
        'RCC-8498', 'RCC-8496', 'RCC-8494', 'RCC-8489', 'RCC-8486', 'RCC-8484',
        'RCC-8482', 'RCC-8480', 'RCC-8475', 'RCC-8474', 'RCC-8473', 'RCC-8470',
        'RCC-8466', 'RCC-9166', 'RCC-9162', 'RCC-9159', 'RCC-9158', 'RCC-9155', 'RCC-9151',
        'RCC-9150', 'RCC-9146', 'RCC-9143', 'RCC-9141', 'RCC-9138', 'RCC-9134',
        'RCC-9132', 'RCC-9130', 'RCC-9126', 'RCC-9124', 'RCC-9123', 'RCC-9120',
        'RCC-9116', 'RCC-9114', 'RCC-9112', 'RCC-9109', 'RCC-9107', 'RCC-9104',
        'RCC-9102', 'RCC-9099', 'RCC-9096', 'RCC-9095', 'RCC-9093', 'RCC-9090',
        'RCC-9088', 'RCC-9085', 'RCC-9083', 'RCC-9079', 'RCC-9075', 'RCC-9073',
        'RCC-9071', 'RCC-9068', 'RCC-9066', 'RCC-9065', 'RCC-9063', 'RCC-9060',
        'RCC-9057', 'RCC-9056', 'RCC-9051', 'RCC-9049', 'RCC-9047', 'RCC-9046',
        'RCC-9042', 'RCC-9040', 'RCC-9038', 'RCC-9036', 'RCC-9034', 'RCC-9030',
        'RCC-9028', 'RCC-9027', 'RCC-9024', 'RCC-9021', 'RCC-9019', 'RCC-9017',
        'RCC-9014', 'RCC-9010', 'RCC-9009', 'RCC-9007', 'RCC-9005', 'RCC-9003',
        'RCC-8999', 'RCC-8996', 'RCC-8995', 'RCC-8991', 'RCC-8989', 'RCC-8987',
        'RCC-8984', 'RCC-8980', 'RCC-8979', 'RCC-8976', 'RCC-8974', 'RCC-8971',
        'RCC-8968', 'RCC-8966', 'RCC-8963', 'RCC-8962', 'RCC-8958', 'RCC-8957',
        'RCC-8953', 'RCC-8950', 'RCC-8949', 'RCC-8947', 'RCC-8944', 'RCC-8941',
        'RCC-8938', 'RCC-8935', 'RCC-8933', 'RCC-8930', 'RCC-8929', 'RCC-8927',
        'RCC-8925', 'RCC-8922', 'RCC-8919', 'RCC-8917', 'RCC-8915', 'RCC-8913',
        'RCC-8910', 'RCC-8908', 'RCC-8905', 'RCC-8902', 'RCC-8900', 'RCC-8898',
        'RCC-8897', 'RCC-8894', 'RCC-8890', 'RCC-8889', 'RCC-8887', 'RCC-8886',
        'RCC-8883', 'RCC-8880', 'RCC-8878', 'RCC-8876', 'RCC-8874', 'RCC-8870',
        'RCC-8869', 'RCC-8866', 'RCC-8865', 'RCC-8862', 'RCC-8860', 'RCC-8856',
        'RCC-8855', 'RCC-8851', 'RCC-8850', 'RCC-8846', 'RCC-8843', 'RCC-8841',
        'RCC-8839', 'RCC-8837', 'RCC-8835', 'RCC-8833', 'RCC-8830', 'RCC-8827',
        'RCC-8825', 'RCC-8823', 'RCC-8821', 'RCC-8819', 'RCC-8815', 'RCC-8813',
        'RCC-8810', 'RCC-8808', 'RCC-8806', 'RCC-8803', 'RCC-8802', 'RCC-8800',
        'RCC-8799', 'RCC-8796', 'RCC-8793', 'RCC-8791', 'RCC-8788', 'RCC-8786',
        'RCC-8783', 'RCC-8780', 'RCC-8778', 'RCC-8775', 'RCC-8773', 'RCC-8769',
        'RCC-8767', 'RCC-8763', 'RCC-8760', 'RCC-8758', 'RCC-8756', 'RCC-8754',
        'RCC-8751', 'RCC-8749', 'RCC-8747', 'RCC-8745', 'RCC-8742', 'RCC-8738',
        'RCC-8736', 'RCC-8734', 'RCC-8732', 'RCC-8729', 'RCC-8727', 'RCC-8723',
        'RCC-8722', 'RCC-8719', 'RCC-8717', 'RCC-8714', 'RCC-8711', 'RCC-8710',
        'RCC-8706', 'RCC-8704', 'RCC-8702', 'RCC-8701', 'RCC-8698', 'RCC-8697',
        'RCC-8694', 'RCC-8689', 'RCC-8688', 'RCC-8686', 'RCC-8684', 'RCC-8681',
        'RCC-8680', 'RCC-8678', 'RCC-8676', 'RCC-8674', 'RCC-8671', 'RCC-8669',
        'RCC-8665', 'RCC-8663', 'RCC-8662', 'RCC-8659', 'RCC-8654', 'RCC-8653',
        'RCC-8650', 'RCC-8648', 'RCC-8644', 'RCC-8642', 'RCC-8640', 'RCC-8637',
        'RCC-8634', 'RCC-8632', 'RCC-8630', 'RCC-8628', 'RCC-8625', 'RCC-8623',
        'RCC-8619', 'RCC-8617', 'RCC-8614', 'RCC-8613', 'RCC-8607', 'RCC-8606',
        'RCC-8604', 'RCC-8602', 'RCC-8598', 'RCC-8595', 'RCC-8594', 'RCC-8591',
        'RCC-8589', 'RCC-8587', 'RCC-8585', 'RCC-8582', 'RCC-8580', 'RCC-8578',
        'RCC-8573', 'RCC-8571', 'RCC-8569', 'RCC-8567', 'RCC-8564', 'RCC-8562',
        'RCC-8561', 'RCC-8559', 'RCC-8557', 'RCC-8553', 'RCC-8549', 'RCC-8548',
        'RCC-8544', 'RCC-8538', 'RCC-8536', 'RCC-8534', 'RCC-8532', 'RCC-8529',
        'RCC-8528', 'RCC-8524', 'RCC-8522', 'RCC-8520', 'RCC-8516', 'RCC-8513',
        'RCC-8511', 'RCC-8509', 'RCC-8506', 'RCC-8505', 'RCC-8502', 'RCC-8501',
        'RCC-8498', 'RCC-8496', 'RCC-8494', 'RCC-8489', 'RCC-8486', 'RCC-8484',
        'RCC-8482', 'RCC-8480', 'RCC-8475', 'RCC-8474', 'RCC-8473', 'RCC-8470',
        'RCC-8466', 'RCC-8464', 'RCC-8462', 'RCC-8461', 'RCC-8457', 'RCC-8454', 'RCC-8453',
        'RCC-8449', 'RCC-8447', 'RCC-8444', 'RCC-8442', 'RCC-8439', 'RCC-8438',
        'RCC-8435', 'RCC-8432', 'RCC-8428', 'RCC-8426', 'RCC-8423', 'RCC-8420',
        'RCC-8419', 'RCC-8416', 'RCC-8413', 'RCC-8411', 'RCC-8407', 'RCC-8404',
        'RCC-8402', 'RCC-8399', 'RCC-8397', 'RCC-8395', 'RCC-8392', 'RCC-8389',
        'RCC-8387', 'RCC-8384', 'RCC-8382', 'RCC-8378', 'RCC-8376', 'RCC-8373',
        'RCC-8370', 'RCC-8368', 'RCC-8364', 'RCC-8361', 'RCC-8357', 'RCC-8355',
        'RCC-8352', 'RCC-8351', 'RCC-8348', 'RCC-8343', 'RCC-8341', 'RCC-8339',
        'RCC-8338', 'RCC-8333', 'RCC-8331', 'RCC-8329', 'RCC-8326', 'RCC-8323',
        'RCC-8321', 'RCC-8319', 'RCC-8315', 'RCC-8314', 'RCC-8310', 'RCC-8309',
        'RCC-8306', 'RCC-8304', 'RCC-8302', 'RCC-8299', 'RCC-8296', 'RCC-8294',
        'RCC-8293', 'RCC-8289', 'RCC-8286', 'RCC-8284', 'RCC-8281', 'RCC-8278',
        'RCC-8276', 'RCC-8273', 'RCC-8270', 'RCC-8269', 'RCC-8266', 'RCC-8264',
        'RCC-8260', 'RCC-8257', 'RCC-8256', 'RCC-8252', 'RCC-8250', 'RCC-8247',
        'RCC-8245', 'RCC-8244', 'RCC-8240', 'RCC-8237', 'RCC-8236', 'RCC-8234',
        'RCC-8230', 'RCC-8228', 'RCC-8225', 'RCC-8222', 'RCC-8219', 'RCC-8216',
        'RCC-8215', 'RCC-8213', 'RCC-8210', 'RCC-8207', 'RCC-8203', 'RCC-8201',
        'RCC-8199', 'RCC-8197', 'RCC-8193', 'RCC-8189', 'RCC-8188', 'RCC-8184',
        'RCC-8182', 'RCC-8179', 'RCC-8176', 'RCC-8173', 'RCC-8171', 'RCC-8167',
        'RCC-8166', 'RCC-8162', 'RCC-8159', 'RCC-8156', 'RCC-8154', 'RCC-8153',
        'RCC-8149', 'RCC-8147', 'RCC-8146', 'RCC-8143', 'RCC-8140', 'RCC-8137',
        'RCC-8134', 'RCC-8132', 'RCC-8129', 'RCC-8124', 'RCC-8122', 'RCC-8121',
        'RCC-8119', 'RCC-8116', 'RCC-8114', 'RCC-8108', 'RCC-8105', 'RCC-8103',
        'RCC-8100', 'RCC-8099', 'RCC-8097', 'RCC-8093', 'RCC-8091', 'RCC-8088',
        'RCC-8085', 'RCC-8083', 'RCC-8081', 'RCC-8078', 'RCC-8077', 'RCC-8076',
        'RCC-8073', 'RCC-8070', 'RCC-8067', 'RCC-8064', 'RCC-8062', 'RCC-8060',
        'RCC-8056', 'RCC-8053', 'RCC-8051', 'RCC-8047', 'RCC-8045', 'RCC-8042',
        'RCC-8039', 'RCC-8037', 'RCC-8034', 'RCC-8031', 'RCC-8029'
    ]
    failed_run_id_error = {}
    for run_id in run_id_list:
        print('##############################################')
        print('##############################################')
        print('Running: ', run_id)
        start_time = time.time()
        try:
            main(run_id,
                 yes_plot_latent_space=True,
                 which_finetune_nums=[],
                 # which_finetune_nums=[4,5,6],
                 task_name_list=[])
            # task_name_list=['IMDC'])
            print('Minutes elapsed: ', (time.time() - start_time) / 60)
        except Exception as e:
            print('Error: ', e)
            failed_run_id_error[run_id] = e
    print(failed_run_id_error)

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
    """
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
    """
