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
        'RCC-6080', 'RCC-6078', 'RCC-6076', 'RCC-6073', 'RCC-6071', 'RCC-6069',
        'RCC-6067', 'RCC-6064', 'RCC-6062', 'RCC-6059', 'RCC-6058', 'RCC-6052',
        'RCC-6051', 'RCC-6049', 'RCC-6047', 'RCC-6046', 'RCC-6043', 'RCC-6041',
        'RCC-6037', 'RCC-6036', 'RCC-6033', 'RCC-6032', 'RCC-6028', 'RCC-6026',
        'RCC-6024', 'RCC-6021', 'RCC-6019', 'RCC-6017', 'RCC-6014', 'RCC-6013',
        'RCC-6010', 'RCC-6006', 'RCC-6005', 'RCC-6002', 'RCC-6000', 'RCC-5998',
        'RCC-5996', 'RCC-5993', 'RCC-5991', 'RCC-5989', 'RCC-5986', 'RCC-5984',
        'RCC-5983', 'RCC-5980', 'RCC-5977', 'RCC-5974', 'RCC-5972', 'RCC-5971',
        'RCC-5967', 'RCC-5964', 'RCC-5960', 'RCC-5958', 'RCC-5957', 'RCC-5955',
        'RCC-5949', 'RCC-5946', 'RCC-5945', 'RCC-5942', 'RCC-5939', 'RCC-5937',
        'RCC-5935', 'RCC-5933', 'RCC-5928', 'RCC-5926', 'RCC-5923', 'RCC-5920',
        'RCC-5919', 'RCC-5917', 'RCC-5914', 'RCC-5910', 'RCC-5907', 'RCC-5905',
        'RCC-5903', 'RCC-5900', 'RCC-5897', 'RCC-5895', 'RCC-5894', 'RCC-5890',
        'RCC-5887', 'RCC-5885', 'RCC-5884', 'RCC-5880', 'RCC-5877', 'RCC-5876',
        'RCC-5874', 'RCC-5870', 'RCC-5868', 'RCC-5865', 'RCC-5862', 'RCC-5859',
        'RCC-5856', 'RCC-5854', 'RCC-5850', 'RCC-5848', 'RCC-5846', 'RCC-5843',
        'RCC-5839', 'RCC-5837', 'RCC-5835', 'RCC-5832', 'RCC-5829', 'RCC-5826',
        'RCC-5822', 'RCC-5821', 'RCC-5818', 'RCC-5816', 'RCC-5813', 'RCC-5810',
        'RCC-5807', 'RCC-5800', 'RCC-5799', 'RCC-5797', 'RCC-5794', 'RCC-5791',
        'RCC-5790', 'RCC-5788', 'RCC-5786', 'RCC-5783', 'RCC-5781', 'RCC-5779',
        'RCC-5776', 'RCC-5774', 'RCC-5769', 'RCC-5766', 'RCC-5764', 'RCC-5762',
        'RCC-5760', 'RCC-5758', 'RCC-5754', 'RCC-5752', 'RCC-5748', 'RCC-5745',
        'RCC-5743', 'RCC-5740', 'RCC-5737', 'RCC-5733', 'RCC-5732', 'RCC-5729',
        'RCC-5727', 'RCC-5722', 'RCC-5719', 'RCC-5717', 'RCC-5714', 'RCC-5711',
        'RCC-5707', 'RCC-5705', 'RCC-5703', 'RCC-5701', 'RCC-5698', 'RCC-5696',
        'RCC-5693', 'RCC-5689', 'RCC-5686', 'RCC-5683', 'RCC-5681', 'RCC-5678',
        'RCC-5676', 'RCC-5673', 'RCC-5669', 'RCC-5667', 'RCC-5664', 'RCC-5661',
        'RCC-5659', 'RCC-5656', 'RCC-5654', 'RCC-5651', 'RCC-5649', 'RCC-5646',
        'RCC-5645', 'RCC-5642', 'RCC-5639', 'RCC-5637', 'RCC-5635', 'RCC-5631',
        'RCC-5629', 'RCC-5627', 'RCC-5623', 'RCC-5621', 'RCC-5618', 'RCC-5616',
        'RCC-5613', 'RCC-5611', 'RCC-5608', 'RCC-5604', 'RCC-5602', 'RCC-5599',
        'RCC-5595', 'RCC-5593', 'RCC-5591', 'RCC-5587', 'RCC-5584', 'RCC-5581',
        'RCC-5578', 'RCC-5577', 'RCC-5573', 'RCC-5571', 'RCC-5568', 'RCC-5563',
        'RCC-5561', 'RCC-5558', 'RCC-5552', 'RCC-5550', 'RCC-5547', 'RCC-5545',
        'RCC-5541', 'RCC-5540', 'RCC-5536', 'RCC-5533', 'RCC-5529', 'RCC-5527',
        'RCC-5524', 'RCC-5520', 'RCC-5518', 'RCC-5515', 'RCC-5513', 'RCC-5510',
        'RCC-5508', 'RCC-5504', 'RCC-5503', 'RCC-5500', 'RCC-5498', 'RCC-5495',
        'RCC-5494', 'RCC-5490', 'RCC-5488', 'RCC-5485', 'RCC-5483', 'RCC-5480',
        'RCC-5477', 'RCC-5475', 'RCC-5471', 'RCC-5469', 'RCC-5467', 'RCC-5463',
        'RCC-5457', 'RCC-5456', 'RCC-5454', 'RCC-5449', 'RCC-5447', 'RCC-5443',
        'RCC-5440', 'RCC-5436', 'RCC-5434', 'RCC-5432', 'RCC-5427', 'RCC-5424',
        'RCC-5422', 'RCC-5421', 'RCC-5418', 'RCC-5414', 'RCC-5409', 'RCC-5406',
        'RCC-5403', 'RCC-5399', 'RCC-5396', 'RCC-5393', 'RCC-5387', 'RCC-5386',
        'RCC-5383', 'RCC-5380', 'RCC-5376', 'RCC-5372', 'RCC-5368', 'RCC-5367',
        'RCC-5366', 'RCC-5363', 'RCC-5360', 'RCC-5359', 'RCC-5357', 'RCC-5354',
        'RCC-5352', 'RCC-5349', 'RCC-5348', 'RCC-5345', 'RCC-5342', 'RCC-5341',
        'RCC-5340', 'RCC-5337', 'RCC-5336', 'RCC-5334', 'RCC-5332', 'RCC-5330', 'RCC-5329',
        'RCC-5327', 'RCC-5325', 'RCC-5323', 'RCC-5322', 'RCC-5320', 'RCC-5318',
        'RCC-5317', 'RCC-5315', 'RCC-5314', 'RCC-5312', 'RCC-5311', 'RCC-5309'
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
