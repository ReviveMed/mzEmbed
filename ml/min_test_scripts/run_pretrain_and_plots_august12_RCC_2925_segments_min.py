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
        'RCC-9508', 'RCC-9506', 'RCC-9505', 'RCC-9502', 'RCC-9500', 'RCC-9497',
        'RCC-9495', 'RCC-9494', 'RCC-9492', 'RCC-9490', 'RCC-9488', 'RCC-9485',
        'RCC-9483', 'RCC-9481', 'RCC-9480', 'RCC-9476', 'RCC-9475', 'RCC-9472',
        'RCC-9470', 'RCC-9467', 'RCC-9465', 'RCC-9461', 'RCC-9460', 'RCC-9459',
        'RCC-9456', 'RCC-9455', 'RCC-9451', 'RCC-9449', 'RCC-9447', 'RCC-9444',
        'RCC-9443', 'RCC-9441', 'RCC-9440', 'RCC-9436', 'RCC-9434', 'RCC-9432',
        'RCC-9430', 'RCC-9429', 'RCC-9424', 'RCC-9423', 'RCC-9421', 'RCC-9418',
        'RCC-9415', 'RCC-9411', 'RCC-9408', 'RCC-9407', 'RCC-9404', 'RCC-9403',
        'RCC-9402', 'RCC-9397', 'RCC-9394', 'RCC-9393', 'RCC-9390', 'RCC-9388',
        'RCC-9383', 'RCC-9382', 'RCC-9378', 'RCC-9377', 'RCC-9375', 'RCC-9373',
        'RCC-9370', 'RCC-9366', 'RCC-9364', 'RCC-9363', 'RCC-9361', 'RCC-9358',
        'RCC-9357', 'RCC-9353', 'RCC-9352', 'RCC-9349', 'RCC-9347', 'RCC-9345',
        'RCC-9343', 'RCC-9340', 'RCC-9338', 'RCC-9335', 'RCC-9334', 'RCC-9333',
        'RCC-9330', 'RCC-9329', 'RCC-9326', 'RCC-9322', 'RCC-9319', 'RCC-9316',
        'RCC-9314', 'RCC-9312', 'RCC-9309', 'RCC-9306', 'RCC-9305', 'RCC-9303',
        'RCC-9299', 'RCC-9297', 'RCC-9294', 'RCC-9291', 'RCC-9289', 'RCC-9287',
        'RCC-9285', 'RCC-9283', 'RCC-9280', 'RCC-9279', 'RCC-9275', 'RCC-9271',
        'RCC-9270', 'RCC-9266', 'RCC-9265', 'RCC-9262', 'RCC-9259', 'RCC-9256',
        'RCC-9253', 'RCC-9252', 'RCC-9250', 'RCC-9248', 'RCC-9246', 'RCC-9243',
        'RCC-9241', 'RCC-9238', 'RCC-9234', 'RCC-9232', 'RCC-9230', 'RCC-9227',
        'RCC-9225', 'RCC-9223', 'RCC-9221', 'RCC-9220', 'RCC-9216', 'RCC-9214',
        'RCC-9212', 'RCC-9209', 'RCC-9208', 'RCC-9205', 'RCC-9202', 'RCC-9201',
        'RCC-9198', 'RCC-9195', 'RCC-9193', 'RCC-9191', 'RCC-9188', 'RCC-9185',
        'RCC-9183', 'RCC-9182', 'RCC-9179', 'RCC-9176', 'RCC-9173', 'RCC-9172',
        'RCC-9170', 'RCC-9167', 'RCC-9165', 'RCC-9160', 'RCC-9157', 'RCC-9152',
        'RCC-9149', 'RCC-9144', 'RCC-9140', 'RCC-9135', 'RCC-9133', 'RCC-9128',
        'RCC-9125', 'RCC-9122', 'RCC-9117', 'RCC-9113', 'RCC-9110', 'RCC-9105',
        'RCC-9103', 'RCC-9098', 'RCC-9094', 'RCC-9091', 'RCC-9086', 'RCC-9084',
        'RCC-9077', 'RCC-9074', 'RCC-9072', 'RCC-9067', 'RCC-9064', 'RCC-9058',
        'RCC-9055', 'RCC-9048', 'RCC-9045', 'RCC-9041', 'RCC-9039', 'RCC-9035',
        'RCC-9029', 'RCC-9023', 'RCC-9020', 'RCC-9016', 'RCC-9011', 'RCC-9006',
        'RCC-9004', 'RCC-8997', 'RCC-8993', 'RCC-8988', 'RCC-8985', 'RCC-8983',
        'RCC-8982', 'RCC-8978', 'RCC-8973', 'RCC-8970', 'RCC-8965', 'RCC-8961',
        'RCC-8956', 'RCC-8952', 'RCC-8948', 'RCC-8942', 'RCC-8939', 'RCC-8934',
        'RCC-8928', 'RCC-8924', 'RCC-8918', 'RCC-8916', 'RCC-8914', 'RCC-8909',
        'RCC-8901', 'RCC-8899', 'RCC-8896', 'RCC-8891', 'RCC-8888', 'RCC-8884',
        'RCC-8882', 'RCC-8877', 'RCC-8872', 'RCC-8867', 'RCC-8863', 'RCC-8861',
        'RCC-8858', 'RCC-8854', 'RCC-8852', 'RCC-8849', 'RCC-8845', 'RCC-8842',
        'RCC-8836', 'RCC-8831', 'RCC-8828', 'RCC-8824', 'RCC-8820', 'RCC-8814',
        'RCC-8812', 'RCC-8809', 'RCC-8805', 'RCC-8801', 'RCC-8798', 'RCC-8794',
        'RCC-8789', 'RCC-8785', 'RCC-8781', 'RCC-8777', 'RCC-8772', 'RCC-8768',
        'RCC-8762', 'RCC-8757', 'RCC-8753', 'RCC-8750', 'RCC-8746', 'RCC-8739',
        'RCC-8737', 'RCC-8731', 'RCC-8728', 'RCC-8724', 'RCC-8721', 'RCC-8716',
        'RCC-8712', 'RCC-8708', 'RCC-8705', 'RCC-8700', 'RCC-8696', 'RCC-8690',
        'RCC-8687', 'RCC-8683', 'RCC-8679', 'RCC-8675', 'RCC-8672', 'RCC-8666',
        'RCC-8664', 'RCC-8660', 'RCC-8656', 'RCC-8651', 'RCC-8649', 'RCC-8643',
        'RCC-8641', 'RCC-8638', 'RCC-8633', 'RCC-8629', 'RCC-8626', 'RCC-8622', 'RCC-8618',
        'RCC-8616', 'RCC-8611', 'RCC-8605', 'RCC-8603', 'RCC-8597', 'RCC-8593',
        'RCC-8590', 'RCC-8586', 'RCC-8581', 'RCC-8576', 'RCC-8570', 'RCC-8568',
        'RCC-8563', 'RCC-8560', 'RCC-8558', 'RCC-8551', 'RCC-8546', 'RCC-8542',
        'RCC-8541', 'RCC-8540', 'RCC-8535', 'RCC-8533', 'RCC-8530', 'RCC-8525',
        'RCC-8521', 'RCC-8517', 'RCC-8512', 'RCC-8510', 'RCC-8507', 'RCC-8503',
        'RCC-8499', 'RCC-8497', 'RCC-8490', 'RCC-8487', 'RCC-8485', 'RCC-8483',
        'RCC-8477', 'RCC-8472', 'RCC-8467', 'RCC-8463', 'RCC-8459', 'RCC-8455',
        'RCC-8452', 'RCC-8446', 'RCC-8443', 'RCC-8440', 'RCC-8437', 'RCC-8431',
        'RCC-8429', 'RCC-8424', 'RCC-8421', 'RCC-8418', 'RCC-8414', 'RCC-8412',
        'RCC-8408', 'RCC-8403', 'RCC-8400', 'RCC-8398', 'RCC-8394', 'RCC-8390',
        'RCC-8388', 'RCC-8386', 'RCC-8381', 'RCC-8379', 'RCC-8377', 'RCC-8374',
        'RCC-8371', 'RCC-8367', 'RCC-8366', 'RCC-8362', 'RCC-8359', 'RCC-8356',
        'RCC-8353', 'RCC-8349', 'RCC-8345', 'RCC-8342', 'RCC-8340', 'RCC-8337',
        'RCC-8332', 'RCC-8330', 'RCC-8328', 'RCC-8322', 'RCC-8320', 'RCC-8316',
        'RCC-8312', 'RCC-8308', 'RCC-8305', 'RCC-8303', 'RCC-8297', 'RCC-8295',
        'RCC-8292', 'RCC-8288', 'RCC-8285', 'RCC-8279', 'RCC-8275', 'RCC-8271',
        'RCC-8267', 'RCC-8262', 'RCC-8258', 'RCC-8253', 'RCC-8249', 'RCC-8248',
        'RCC-8246', 'RCC-8243', 'RCC-8239', 'RCC-8235', 'RCC-8232', 'RCC-8227',
        'RCC-8226', 'RCC-8221', 'RCC-8218', 'RCC-8212', 'RCC-8208', 'RCC-8204',
        'RCC-8202', 'RCC-8198', 'RCC-8196', 'RCC-8190', 'RCC-8187', 'RCC-8185',
        'RCC-8181', 'RCC-8178', 'RCC-8175', 'RCC-8172', 'RCC-8169', 'RCC-8168',
        'RCC-8163', 'RCC-8160', 'RCC-8157', 'RCC-8152', 'RCC-8150', 'RCC-8148',
        'RCC-8144', 'RCC-8139', 'RCC-8138', 'RCC-8133', 'RCC-8130', 'RCC-8125',
        'RCC-8120', 'RCC-8117', 'RCC-8115', 'RCC-8112', 'RCC-8109', 'RCC-8107',
        'RCC-8104', 'RCC-8101', 'RCC-8096', 'RCC-8092', 'RCC-8089', 'RCC-8086',
        'RCC-8082', 'RCC-8079', 'RCC-8074', 'RCC-8072', 'RCC-8068', 'RCC-8065',
        'RCC-8061', 'RCC-8057', 'RCC-8055', 'RCC-8050', 'RCC-8044', 'RCC-8043',
        'RCC-8040', 'RCC-8036', 'RCC-8035', 'RCC-8030', 'RCC-8028'
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
