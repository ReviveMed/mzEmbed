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
        'RCC-6488', 'RCC-6766', 'RCC-6682', 'RCC-7101', 'RCC-7073', 'RCC-6797',
        'RCC-6517', 'RCC-6868', 'RCC-5952', 'RCC-6874', 'RCC-7203', 'RCC-6912',
        'RCC-6782', 'RCC-6241', 'RCC-6417', 'RCC-7036', 'RCC-6181', 'RCC-6943',
        'RCC-6095', 'RCC-5927', 'RCC-7111', 'RCC-6359', 'RCC-5739', 'RCC-6408',
        'RCC-6340', 'RCC-7155', 'RCC-6596', 'RCC-7201', 'RCC-6752', 'RCC-7146',
        'RCC-6839', 'RCC-6526', 'RCC-7154', 'RCC-6878', 'RCC-6122', 'RCC-7191',
        'RCC-6735', 'RCC-7148', 'RCC-7089', 'RCC-6295', 'RCC-7193', 'RCC-5982',
        'RCC-6960', 'RCC-7099', 'RCC-6430', 'RCC-6173', 'RCC-6984', 'RCC-6351',
        'RCC-7211', 'RCC-6826', 'RCC-7164', 'RCC-7176', 'RCC-7022', 'RCC-7165',
        'RCC-7119', 'RCC-7135', 'RCC-7160', 'RCC-6544', 'RCC-7218', 'RCC-7085',
        'RCC-6275', 'RCC-7152', 'RCC-7212', 'RCC-7207', 'RCC-6889', 'RCC-6815',
        'RCC-7133', 'RCC-7213', 'RCC-7209', 'RCC-7222', 'RCC-7030', 'RCC-7076',
        'RCC-7091', 'RCC-5462', 'RCC-7169', 'RCC-7171', 'RCC-7096', 'RCC-7046',
        'RCC-7008', 'RCC-7041', 'RCC-7140', 'RCC-7182', 'RCC-5493', 'RCC-7127',
        'RCC-7163', 'RCC-7123', 'RCC-7217', 'RCC-6221', 'RCC-6658', 'RCC-7174',
        'RCC-7115', 'RCC-7166', 'RCC-7167', 'RCC-5757', 'RCC-5814', 'RCC-5582',
        'RCC-7157', 'RCC-7172', 'RCC-6060', 'RCC-5777', 'RCC-6713', 'RCC-6998',
        'RCC-7153', 'RCC-6927', 'RCC-6830', 'RCC-5765', 'RCC-6272', 'RCC-6613',
        'RCC-6390', 'RCC-6803', 'RCC-7170', 'RCC-5474', 'RCC-6462', 'RCC-6246',
        'RCC-6633', 'RCC-7168', 'RCC-7056', 'RCC-6102', 'RCC-6904', 'RCC-5898',
        'RCC-6749', 'RCC-7131', 'RCC-6728', 'RCC-5888', 'RCC-6902', 'RCC-7184',
        'RCC-6161', 'RCC-6268', 'RCC-7205', 'RCC-6168', 'RCC-6236', 'RCC-7202',
        'RCC-7139', 'RCC-7049', 'RCC-6693', 'RCC-7018', 'RCC-7215', 'RCC-6886',
        'RCC-6401', 'RCC-7107', 'RCC-6821', 'RCC-7137', 'RCC-6813', 'RCC-7150',
        'RCC-5694', 'RCC-7052', 'RCC-6819', 'RCC-6937', 'RCC-6845', 'RCC-5831',
        'RCC-6853', 'RCC-7221', 'RCC-7061', 'RCC-6987', 'RCC-7183', 'RCC-7044',
        'RCC-7104', 'RCC-6800', 'RCC-7219', 'RCC-6972', 'RCC-6834', 'RCC-6992',
        'RCC-6997', 'RCC-6952', 'RCC-6745', 'RCC-6650', 'RCC-5906', 'RCC-7081',
        'RCC-6969', 'RCC-7188', 'RCC-6302', 'RCC-7210', 'RCC-7208', 'RCC-6232',
        'RCC-6453', 'RCC-6794', 'RCC-7054', 'RCC-6196', 'RCC-7129', 'RCC-6980',
        'RCC-7190', 'RCC-6778', 'RCC-7012', 'RCC-6863', 'RCC-6975', 'RCC-6564',
        'RCC-7006', 'RCC-7156', 'RCC-7105', 'RCC-6907', 'RCC-7143', 'RCC-6575',
        'RCC-7189', 'RCC-7158', 'RCC-6395', 'RCC-6698', 'RCC-6322', 'RCC-6253',
        'RCC-6205', 'RCC-6277', 'RCC-6723', 'RCC-6899', 'RCC-6610', 'RCC-6940',
        'RCC-6859', 'RCC-6470', 'RCC-5728', 'RCC-7021', 'RCC-5572', 'RCC-7039',
        'RCC-7003', 'RCC-6958', 'RCC-6919', 'RCC-6856', 'RCC-6202', 'RCC-7192',
        'RCC-6764', 'RCC-6212', 'RCC-7180', 'RCC-7194', 'RCC-6287', 'RCC-7178',
        'RCC-6440', 'RCC-6539', 'RCC-6283', 'RCC-7223', 'RCC-6480', 'RCC-6588',
        'RCC-7083', 'RCC-7220', 'RCC-6499', 'RCC-6947', 'RCC-7121', 'RCC-6314',
        'RCC-6556', 'RCC-7063', 'RCC-6424', 'RCC-6704', 'RCC-7113', 'RCC-6308',
        'RCC-5851', 'RCC-5410', 'RCC-6739', 'RCC-6217', 'RCC-6229', 'RCC-6154',
        'RCC-6624', 'RCC-7179', 'RCC-5517', 'RCC-7108', 'RCC-6474', 'RCC-5433',
        'RCC-5391', 'RCC-6954', 'RCC-6916', 'RCC-6644', 'RCC-6756', 'RCC-6335',
        'RCC-7125', 'RCC-6893', 'RCC-7195', 'RCC-7016', 'RCC-6806', 'RCC-7181',
        'RCC-5867', 'RCC-6506', 'RCC-5796', 'RCC-7214', 'RCC-6018', 'RCC-7197',
        'RCC-6075', 'RCC-6290', 'RCC-6368', 'RCC-7206', 'RCC-6882', 'RCC-7175',
        'RCC-7177', 'RCC-6111', 'RCC-7080', 'RCC-7066', 'RCC-7173', 'RCC-6262', 'RCC-7162',
        'RCC-6933', 'RCC-6258', 'RCC-6742', 'RCC-5605', 'RCC-5647', 'RCC-6377',
        'RCC-6209', 'RCC-6135', 'RCC-6620', 'RCC-6148', 'RCC-6812', 'RCC-6225',
        'RCC-7094', 'RCC-6249', 'RCC-6929', 'RCC-5371', 'RCC-6760', 'RCC-7216',
        'RCC-6003', 'RCC-6311', 'RCC-7031', 'RCC-6372', 'RCC-6582', 'RCC-7161',
        'RCC-6298', 'RCC-6305', 'RCC-6967', 'RCC-5665', 'RCC-6030', 'RCC-6329',
        'RCC-6008', 'RCC-5555', 'RCC-7117', 'RCC-6281', 'RCC-7200', 'RCC-6668',
        'RCC-7034', 'RCC-5970', 'RCC-6291', 'RCC-6866', 'RCC-7068', 'RCC-5625',
        'RCC-6838', 'RCC-7198', 'RCC-5718', 'RCC-6604', 'RCC-5806', 'RCC-5844',
        'RCC-6379', 'RCC-6309', 'RCC-6386', 'RCC-5448', 'RCC-7204', 'RCC-7025',
        'RCC-5596', 'RCC-5484', 'RCC-6676', 'RCC-6923', 'RCC-5565', 'RCC-7027',
        'RCC-6447', 'RCC-7058', 'RCC-7199', 'RCC-7196', 'RCC-7187', 'RCC-7186',
        'RCC-7185', 'RCC-7159', 'RCC-7087', 'RCC-7071', 'RCC-7015', 'RCC-7011',
        'RCC-6911', 'RCC-6897', 'RCC-6865', 'RCC-6851', 'RCC-6774', 'RCC-6734',
        'RCC-6718', 'RCC-6468', 'RCC-6446', 'RCC-6245', 'RCC-6094', 'RCC-6089',
        'RCC-6042', 'RCC-6029', 'RCC-5951', 'RCC-5950', 'RCC-5825', 'RCC-5805',
        'RCC-5802', 'RCC-5756', 'RCC-5725', 'RCC-5723', 'RCC-5560', 'RCC-5532',
        'RCC-5461', 'RCC-5458', 'RCC-5388'
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
