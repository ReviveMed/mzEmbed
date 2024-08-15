# Finetune the top pretrained models using 3 sets of finetuning hyper-parameters

import os
import optuna
import pandas as pd
from prep_run import create_selected_data, make_kwargs_set, get_task_head_kwargs, round_kwargs_to_sig, flatten_dict, \
    unflatten_dict
from utils_neptune import get_latest_dataset, get_run_id_list, get_run_id_list_from_query, get_filtered_run_ids_by_tag
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
                              tags=['add test'],

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
                                      tags=['RCC all studies recon only finetune'],
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
                                      tags=['RCC all studies recon only finetune'],
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
    # Define the tags you want to filter by
    include_tag = "RCC all data new split pretrain"
    exclude_tag = "training failed"

    # Create a query to filter by the specified tags
    # run_id_list = get_filtered_run_ids_by_tag(include_tag, exclude_tag, project_id=project_id)
    run_id_list = [
        'RCC-10516', 'RCC-10513', 'RCC-10510', 'RCC-10507', 'RCC-10504', 'RCC-10501',
        'RCC-10498', 'RCC-10493', 'RCC-10491', 'RCC-10488', 'RCC-10484', 'RCC-10481',
        'RCC-10476', 'RCC-10473', 'RCC-10469', 'RCC-10467', 'RCC-10462', 'RCC-10460',
        'RCC-10457', 'RCC-10453', 'RCC-10452', 'RCC-10448', 'RCC-10444', 'RCC-10441',
        'RCC-10438', 'RCC-10434', 'RCC-10431', 'RCC-10427', 'RCC-10424', 'RCC-10420',
        'RCC-10418', 'RCC-10413', 'RCC-10410', 'RCC-10407', 'RCC-10404', 'RCC-10400',
        'RCC-10396', 'RCC-10393', 'RCC-10390', 'RCC-10386', 'RCC-10382', 'RCC-10379',
        'RCC-10375', 'RCC-10372', 'RCC-10369', 'RCC-10365', 'RCC-10363', 'RCC-10359',
        'RCC-10357', 'RCC-10352', 'RCC-10349', 'RCC-10346', 'RCC-10343', 'RCC-10340',
        'RCC-10337', 'RCC-10333', 'RCC-10327', 'RCC-10325', 'RCC-10321', 'RCC-10317',
        'RCC-10314', 'RCC-10311', 'RCC-10308', 'RCC-10303', 'RCC-10301', 'RCC-10296',
        'RCC-10293', 'RCC-10289', 'RCC-10286', 'RCC-10283', 'RCC-10278', 'RCC-10276',
        'RCC-10272', 'RCC-10269', 'RCC-10266', 'RCC-10262', 'RCC-10259', 'RCC-10254',
        'RCC-10249', 'RCC-10246', 'RCC-10242', 'RCC-10238', 'RCC-10235', 'RCC-10231',
        'RCC-10228', 'RCC-10224', 'RCC-10220', 'RCC-10216', 'RCC-10212', 'RCC-10209',
        'RCC-10205', 'RCC-10201', 'RCC-10197', 'RCC-10194', 'RCC-10192', 'RCC-10189',
        'RCC-10186', 'RCC-10184', 'RCC-10181', 'RCC-10178', 'RCC-10176', 'RCC-10172',
        'RCC-10170', 'RCC-10167', 'RCC-10165', 'RCC-10162', 'RCC-10158', 'RCC-10156',
        'RCC-10153', 'RCC-10150', 'RCC-10147', 'RCC-10144', 'RCC-10141', 'RCC-10138',
        'RCC-10136', 'RCC-10133', 'RCC-10131', 'RCC-10128', 'RCC-10125', 'RCC-10121',
        'RCC-10119', 'RCC-10114', 'RCC-10111', 'RCC-10107', 'RCC-10103', 'RCC-10101',
        'RCC-10097', 'RCC-10095', 'RCC-10091', 'RCC-10088', 'RCC-10085', 'RCC-10081',
        'RCC-10078', 'RCC-10074', 'RCC-10070', 'RCC-10066', 'RCC-10062', 'RCC-10059',
        'RCC-10055', 'RCC-10052', 'RCC-10048', 'RCC-10044', 'RCC-10040', 'RCC-10035',
        'RCC-10032', 'RCC-10028', 'RCC-10023', 'RCC-10020', 'RCC-10016', 'RCC-10012',
        'RCC-10007', 'RCC-10003', 'RCC-9999', 'RCC-9995', 'RCC-9990', 'RCC-9986',
        'RCC-9983', 'RCC-9980', 'RCC-9977', 'RCC-9972', 'RCC-9970', 'RCC-9965',
        'RCC-9963', 'RCC-9958', 'RCC-9956', 'RCC-9952', 'RCC-9949', 'RCC-9947',
        'RCC-9943', 'RCC-9939', 'RCC-9936', 'RCC-9931', 'RCC-9927', 'RCC-9923',
        'RCC-9918', 'RCC-9914', 'RCC-9911', 'RCC-9907', 'RCC-9903', 'RCC-9899',
        'RCC-9896', 'RCC-9894', 'RCC-9890', 'RCC-9886', 'RCC-9884', 'RCC-9880',
        'RCC-9876', 'RCC-9872', 'RCC-9866', 'RCC-9865', 'RCC-9863', 'RCC-9859',
        'RCC-9855', 'RCC-9852', 'RCC-9847', 'RCC-9844', 'RCC-9840', 'RCC-9836',
        'RCC-9833', 'RCC-9831', 'RCC-9827', 'RCC-9823', 'RCC-9819', 'RCC-9816',
        'RCC-9812', 'RCC-9807', 'RCC-9804', 'RCC-9800', 'RCC-9797', 'RCC-9793',
        'RCC-9790', 'RCC-9785', 'RCC-9781', 'RCC-9776', 'RCC-9772', 'RCC-9769',
        'RCC-9764', 'RCC-9760', 'RCC-9756', 'RCC-9752', 'RCC-9748', 'RCC-9745',
        'RCC-9742', 'RCC-9739', 'RCC-9735', 'RCC-9730', 'RCC-9726', 'RCC-9723',
        'RCC-9719', 'RCC-9714', 'RCC-9712', 'RCC-9708', 'RCC-9704', 'RCC-9700',
        'RCC-9697', 'RCC-9694', 'RCC-9691', 'RCC-9685', 'RCC-9682', 'RCC-9677',
        'RCC-9674', 'RCC-9667', 'RCC-9664', 'RCC-9660', 'RCC-9656', 'RCC-9650',
        'RCC-9646', 'RCC-9643', 'RCC-9640', 'RCC-9634', 'RCC-9630', 'RCC-9626',
        'RCC-9623', 'RCC-9618', 'RCC-9614', 'RCC-9611', 'RCC-9606', 'RCC-9602',
        'RCC-9598', 'RCC-9595', 'RCC-9590', 'RCC-9587', 'RCC-9583', 'RCC-9579',
        'RCC-9574', 'RCC-9571', 'RCC-9566', 'RCC-9562', 'RCC-9559', 'RCC-9552',
        'RCC-9548', 'RCC-9545', 'RCC-9540', 'RCC-9537', 'RCC-9533', 'RCC-9529',
        'RCC-9526', 'RCC-9522', 'RCC-9517', 'RCC-9514', 'RCC-9510', 'RCC-9503',
        'RCC-9496', 'RCC-9491', 'RCC-9484', 'RCC-9477', 'RCC-9471', 'RCC-9464',
        'RCC-9458', 'RCC-9450', 'RCC-9445', 'RCC-9438', 'RCC-9433', 'RCC-9425',
        'RCC-9419', 'RCC-9413', 'RCC-9406', 'RCC-9400', 'RCC-9395', 'RCC-9389',
        'RCC-9385', 'RCC-9379', 'RCC-9374', 'RCC-9368', 'RCC-9362', 'RCC-9354',
        'RCC-9350', 'RCC-9344', 'RCC-9339', 'RCC-9332', 'RCC-9325', 'RCC-9320',
        'RCC-9315', 'RCC-9308', 'RCC-9302', 'RCC-9295', 'RCC-9290', 'RCC-9284',
        'RCC-9277', 'RCC-9272', 'RCC-9267', 'RCC-9260', 'RCC-9254', 'RCC-9249',
        'RCC-9244', 'RCC-9236', 'RCC-9229', 'RCC-9224', 'RCC-9218', 'RCC-9211',
        'RCC-9206', 'RCC-9199', 'RCC-9194', 'RCC-9189', 'RCC-9184', 'RCC-9178',
        'RCC-9171', 'RCC-9164', 'RCC-9154', 'RCC-9145', 'RCC-9139', 'RCC-9129',
        'RCC-9119', 'RCC-9111', 'RCC-9100', 'RCC-9092', 'RCC-9081', 'RCC-9070',
        'RCC-9062', 'RCC-9054', 'RCC-9044', 'RCC-9033', 'RCC-9025', 'RCC-9013',
        'RCC-9002', 'RCC-8990', 'RCC-8977', 'RCC-8967', 'RCC-8955', 'RCC-8945',
        'RCC-8932', 'RCC-8923', 'RCC-8912', 'RCC-8904', 'RCC-8892', 'RCC-8881',
        'RCC-8868', 'RCC-8857', 'RCC-8844', 'RCC-8834', 'RCC-8822', 'RCC-8807',
        'RCC-8797', 'RCC-8787', 'RCC-8776', 'RCC-8765', 'RCC-8755', 'RCC-8748',
        'RCC-8735', 'RCC-8725', 'RCC-8715', 'RCC-8707', 'RCC-8699', 'RCC-8692',
        'RCC-8685', 'RCC-8677', 'RCC-8667', 'RCC-8657', 'RCC-8647', 'RCC-8639',
        'RCC-8631', 'RCC-8621', 'RCC-8609', 'RCC-8601', 'RCC-8592', 'RCC-8583',
        'RCC-8575', 'RCC-8566', 'RCC-8556', 'RCC-8543', 'RCC-8537', 'RCC-8526',
        'RCC-8515', 'RCC-8504', 'RCC-8491', 'RCC-8479', 'RCC-8465', 'RCC-8456',
        'RCC-8445', 'RCC-8433', 'RCC-8422', 'RCC-8415', 'RCC-8406', 'RCC-8396',
        'RCC-8385', 'RCC-8372', 'RCC-8358', 'RCC-8346', 'RCC-8335', 'RCC-8327',
        'RCC-8318', 'RCC-8311', 'RCC-8300', 'RCC-8290', 'RCC-8277', 'RCC-8265',
        'RCC-8255', 'RCC-8242', 'RCC-8229', 'RCC-8220', 'RCC-8205', 'RCC-8195',
        'RCC-8191', 'RCC-8186', 'RCC-8177', 'RCC-8165', 'RCC-8155', 'RCC-8141',
        'RCC-8128', 'RCC-8123', 'RCC-8110', 'RCC-8098', 'RCC-8090', 'RCC-8080',
        'RCC-8066', 'RCC-8063', 'RCC-8052', 'RCC-8041', 'RCC-8033', 'RCC-8025'
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

    """
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
    """
