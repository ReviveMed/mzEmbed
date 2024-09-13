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
        'RCC-11240', 'RCC-11232', 'RCC-11225', 'RCC-11217', 'RCC-11215', 'RCC-11209', 'RCC-11204',
        'RCC-11198', 'RCC-11192', 'RCC-11185', 'RCC-11179', 'RCC-11173', 'RCC-11166', 'RCC-11158',
        'RCC-11151', 'RCC-11144', 'RCC-11139', 'RCC-11134', 'RCC-11126', 'RCC-11119', 'RCC-11115',
        'RCC-11107', 'RCC-11100', 'RCC-11095', 'RCC-11090', 'RCC-11082', 'RCC-11076', 'RCC-11069',
        'RCC-11059', 'RCC-11053', 'RCC-11046', 'RCC-11041', 'RCC-11039', 'RCC-11031', 'RCC-11023',
        'RCC-11016', 'RCC-11007', 'RCC-11002', 'RCC-10998', 'RCC-10991', 'RCC-10983', 'RCC-10975',
        'RCC-10965', 'RCC-10960', 'RCC-10955', 'RCC-10949', 'RCC-10940', 'RCC-10936', 'RCC-10930',
        'RCC-10923', 'RCC-10918', 'RCC-10911', 'RCC-10910', 'RCC-10905', 'RCC-10901', 'RCC-10894',
        'RCC-10889', 'RCC-10885', 'RCC-10878', 'RCC-10871', 'RCC-10862', 'RCC-10854', 'RCC-10848',
        'RCC-10843', 'RCC-10837', 'RCC-10832', 'RCC-10825', 'RCC-10817', 'RCC-10809', 'RCC-10799',
        'RCC-10794', 'RCC-10788', 'RCC-10783', 'RCC-10774', 'RCC-10768', 'RCC-10760', 'RCC-10759',
        'RCC-10750', 'RCC-10746', 'RCC-10738', 'RCC-10731', 'RCC-10725', 'RCC-10716', 'RCC-10707',
        'RCC-10703', 'RCC-10698', 'RCC-10692', 'RCC-10687', 'RCC-10684', 'RCC-10680', 'RCC-10676',
        'RCC-10673', 'RCC-10668', 'RCC-10664', 'RCC-10660', 'RCC-10655', 'RCC-10650', 'RCC-10645',
        'RCC-10640', 'RCC-10638', 'RCC-10632', 'RCC-10628', 'RCC-10623', 'RCC-10620', 'RCC-10616',
        'RCC-10613', 'RCC-10608', 'RCC-10606', 'RCC-10601', 'RCC-10599', 'RCC-10595', 'RCC-10592',
        'RCC-10588', 'RCC-10585', 'RCC-10581', 'RCC-10578', 'RCC-10574', 'RCC-10571', 'RCC-10567',
        'RCC-10563', 'RCC-10561', 'RCC-10556', 'RCC-10552', 'RCC-10549', 'RCC-10544', 'RCC-10541',
        'RCC-10538', 'RCC-10535', 'RCC-10531', 'RCC-10530', 'RCC-10527', 'RCC-10523', 'RCC-10519',
        'RCC-10514', 'RCC-10508', 'RCC-10503', 'RCC-10497', 'RCC-10495', 'RCC-10489', 'RCC-10483',
        'RCC-10477', 'RCC-10472', 'RCC-10466', 'RCC-10459', 'RCC-10455', 'RCC-10450', 'RCC-10446',
        'RCC-10440', 'RCC-10437', 'RCC-10432', 'RCC-10428', 'RCC-10421', 'RCC-10417', 'RCC-10411',
        'RCC-10408', 'RCC-10402', 'RCC-10398', 'RCC-10392', 'RCC-10387', 'RCC-10384', 'RCC-10377',
        'RCC-10371', 'RCC-10368', 'RCC-10362', 'RCC-10355', 'RCC-10350', 'RCC-10344', 'RCC-10338',
        'RCC-10331', 'RCC-10328', 'RCC-10324', 'RCC-10319', 'RCC-10313', 'RCC-10310', 'RCC-10305',
        'RCC-10299', 'RCC-10291', 'RCC-10287', 'RCC-10280', 'RCC-10275', 'RCC-10270', 'RCC-10263',
        'RCC-10261', 'RCC-10257', 'RCC-10252', 'RCC-10248', 'RCC-10243', 'RCC-10239', 'RCC-10233',
        'RCC-10229', 'RCC-10223', 'RCC-10217', 'RCC-10211', 'RCC-10206', 'RCC-10198', 'RCC-10195',
        'RCC-10190', 'RCC-10187', 'RCC-10182', 'RCC-10179', 'RCC-10175', 'RCC-10169', 'RCC-10164',
        'RCC-10161', 'RCC-10159', 'RCC-10155', 'RCC-10151', 'RCC-10149', 'RCC-10145', 'RCC-10142',
        'RCC-10139', 'RCC-10134', 'RCC-10130', 'RCC-10126', 'RCC-10122', 'RCC-10117', 'RCC-10116',
        'RCC-10112', 'RCC-10105', 'RCC-10098', 'RCC-10092', 'RCC-10086', 'RCC-10080', 'RCC-10077',
        'RCC-10071', 'RCC-10065', 'RCC-10061', 'RCC-10056', 'RCC-10049', 'RCC-10046', 'RCC-10042',
        'RCC-10037', 'RCC-10033', 'RCC-10031', 'RCC-10026', 'RCC-10022', 'RCC-10018', 'RCC-10013',
        'RCC-10010', 'RCC-10002', 'RCC-9996', 'RCC-9992', 'RCC-9987', 'RCC-9985', 'RCC-9979',
        'RCC-9974', 'RCC-9968', 'RCC-9962', 'RCC-9959', 'RCC-9951', 'RCC-9945', 'RCC-9941',
        'RCC-9940', 'RCC-9934', 'RCC-9929', 'RCC-9926', 'RCC-9920', 'RCC-9917', 'RCC-9909',
        'RCC-9902', 'RCC-9898', 'RCC-9893', 'RCC-9889', 'RCC-9883', 'RCC-9877', 'RCC-9873',
        'RCC-9868', 'RCC-9861', 'RCC-9856', 'RCC-9853', 'RCC-9849', 'RCC-9846', 'RCC-9841',
        'RCC-9834', 'RCC-9828', 'RCC-9822', 'RCC-9817', 'RCC-9815', 'RCC-9810', 'RCC-9805',
        'RCC-9799', 'RCC-9796', 'RCC-9789', 'RCC-9786', 'RCC-9780', 'RCC-9775', 'RCC-9770',
        'RCC-9765', 'RCC-9762', 'RCC-9754', 'RCC-9747', 'RCC-9740', 'RCC-9733', 'RCC-9725',
        'RCC-9720', 'RCC-9717', 'RCC-9710', 'RCC-9705', 'RCC-9702', 'RCC-9696', 'RCC-9689',
        'RCC-9686', 'RCC-9678', 'RCC-9675', 'RCC-9670', 'RCC-9666', 'RCC-9661', 'RCC-9658',
        'RCC-9653', 'RCC-9649', 'RCC-9644', 'RCC-9638', 'RCC-9633', 'RCC-9625', 'RCC-9621',
        'RCC-9617', 'RCC-9612', 'RCC-9608', 'RCC-9605', 'RCC-9600', 'RCC-9596', 'RCC-9589',
        'RCC-9585', 'RCC-9581', 'RCC-9577', 'RCC-9570', 'RCC-9563', 'RCC-9555', 'RCC-9549',
        'RCC-9543', 'RCC-9541', 'RCC-9538', 'RCC-9530', 'RCC-9524', 'RCC-9519', 'RCC-9513',
        'RCC-9507', 'RCC-9499', 'RCC-9487', 'RCC-9478', 'RCC-9469', 'RCC-9463', 'RCC-9453',
        'RCC-9439', 'RCC-9427', 'RCC-9416', 'RCC-9409', 'RCC-9398', 'RCC-9391', 'RCC-9384',
        'RCC-9381', 'RCC-9369', 'RCC-9356', 'RCC-9342', 'RCC-9328', 'RCC-9324', 'RCC-9310',
        'RCC-9304', 'RCC-9298', 'RCC-9286', 'RCC-9276', 'RCC-9274', 'RCC-9263', 'RCC-9258',
        'RCC-9247', 'RCC-9239', 'RCC-9235', 'RCC-9228', 'RCC-9213', 'RCC-9204', 'RCC-9197',
        'RCC-9186', 'RCC-9177', 'RCC-9169', 'RCC-9147', 'RCC-9136', 'RCC-9118', 'RCC-9101',
        'RCC-9082', 'RCC-9076', 'RCC-9053', 'RCC-9037', 'RCC-9015', 'RCC-9008', 'RCC-9001',
        'RCC-8981', 'RCC-8959', 'RCC-8951', 'RCC-8946', 'RCC-8937', 'RCC-8920', 'RCC-8907',
        'RCC-8893', 'RCC-8871', 'RCC-8859', 'RCC-8848', 'RCC-8832', 'RCC-8817', 'RCC-8811',
        'RCC-8790', 'RCC-8784', 'RCC-8774', 'RCC-8766', 'RCC-8761', 'RCC-8744', 'RCC-8741',
        'RCC-8726', 'RCC-8718', 'RCC-8693', 'RCC-8668', 'RCC-8652', 'RCC-8636', 'RCC-8615',
        'RCC-8610', 'RCC-8596', 'RCC-8584', 'RCC-8574', 'RCC-8555', 'RCC-8550', 'RCC-8547',
        'RCC-8523', 'RCC-8514', 'RCC-8493', 'RCC-8492', 'RCC-8481', 'RCC-8469', 'RCC-8450',
        'RCC-8427', 'RCC-8409', 'RCC-8391', 'RCC-8369', 'RCC-8360', 'RCC-8347', 'RCC-8324',
        'RCC-8301', 'RCC-8283', 'RCC-8263', 'RCC-8238', 'RCC-8214', 'RCC-8206', 'RCC-8192',
        'RCC-8170', 'RCC-8151', 'RCC-8131', 'RCC-8127', 'RCC-8102', 'RCC-8087', 'RCC-8058',
        'RCC-8048', 'RCC-8026'
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
