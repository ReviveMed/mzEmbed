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
        'RCC-6724', 'RCC-6721', 'RCC-6719', 'RCC-6716', 'RCC-6714', 'RCC-6711',
        'RCC-6709', 'RCC-6707', 'RCC-6706', 'RCC-6702', 'RCC-6701', 'RCC-6699',
        'RCC-6697', 'RCC-6694', 'RCC-6692', 'RCC-6691', 'RCC-6688', 'RCC-6685',
        'RCC-6684', 'RCC-6680', 'RCC-6679', 'RCC-6677', 'RCC-6674', 'RCC-6672',
        'RCC-6669', 'RCC-6666', 'RCC-6664', 'RCC-6662', 'RCC-6660', 'RCC-6659',
        'RCC-6656', 'RCC-6655', 'RCC-6652', 'RCC-6649', 'RCC-6648', 'RCC-6647',
        'RCC-6646', 'RCC-6642', 'RCC-6641', 'RCC-6639', 'RCC-6637', 'RCC-6635',
        'RCC-6632', 'RCC-6631', 'RCC-6628', 'RCC-6623', 'RCC-6622', 'RCC-6621',
        'RCC-6618', 'RCC-6615', 'RCC-6612', 'RCC-6611', 'RCC-6608', 'RCC-6606',
        'RCC-6603', 'RCC-6601', 'RCC-6599', 'RCC-6598', 'RCC-6595', 'RCC-6593',
        'RCC-6591', 'RCC-6589', 'RCC-6587', 'RCC-6585', 'RCC-6581', 'RCC-6580',
        'RCC-6577', 'RCC-6574', 'RCC-6572', 'RCC-6570', 'RCC-6569', 'RCC-6567',
        'RCC-6565', 'RCC-6563', 'RCC-6562', 'RCC-6558', 'RCC-6557', 'RCC-6554',
        'RCC-6552', 'RCC-6550', 'RCC-6548', 'RCC-6546', 'RCC-6542', 'RCC-6541',
        'RCC-6538', 'RCC-6537', 'RCC-6533', 'RCC-6531', 'RCC-6529', 'RCC-6527',
        'RCC-6524', 'RCC-6522', 'RCC-6520', 'RCC-6519', 'RCC-6516', 'RCC-6515',
        'RCC-6512', 'RCC-6511', 'RCC-6510', 'RCC-6508', 'RCC-6504', 'RCC-6503',
        'RCC-6501', 'RCC-6498', 'RCC-6496', 'RCC-6494', 'RCC-6493', 'RCC-6491',
        'RCC-6487', 'RCC-6486', 'RCC-6485', 'RCC-6482', 'RCC-6481', 'RCC-6477',
        'RCC-6476', 'RCC-6472', 'RCC-6469', 'RCC-6467', 'RCC-6466', 'RCC-6463',
        'RCC-6461', 'RCC-6460', 'RCC-6457', 'RCC-6455', 'RCC-6452', 'RCC-6450',
        'RCC-6448', 'RCC-6445', 'RCC-6442', 'RCC-6441', 'RCC-6439', 'RCC-6437',
        'RCC-6435', 'RCC-6433', 'RCC-6431', 'RCC-6429', 'RCC-6427', 'RCC-6425',
        'RCC-6423', 'RCC-6422', 'RCC-6418', 'RCC-6416', 'RCC-6415', 'RCC-6414',
        'RCC-6411', 'RCC-6409', 'RCC-6407', 'RCC-6406', 'RCC-6404', 'RCC-6403',
        'RCC-6402', 'RCC-6399', 'RCC-6397', 'RCC-6396', 'RCC-6393', 'RCC-6391',
        'RCC-6388', 'RCC-6387', 'RCC-6385', 'RCC-6384', 'RCC-6383', 'RCC-6380',
        'RCC-6378', 'RCC-6376', 'RCC-6375', 'RCC-6373', 'RCC-6371', 'RCC-6369',
        'RCC-6366', 'RCC-6365', 'RCC-6364', 'RCC-6363', 'RCC-6362', 'RCC-6360',
        'RCC-6357', 'RCC-6356', 'RCC-6355', 'RCC-6354', 'RCC-6352', 'RCC-6350',
        'RCC-6348', 'RCC-6346', 'RCC-6345', 'RCC-6344', 'RCC-6343', 'RCC-6342',
        'RCC-6337', 'RCC-6336', 'RCC-6334', 'RCC-6333', 'RCC-6331', 'RCC-6328',
        'RCC-6327', 'RCC-6326', 'RCC-6324', 'RCC-6323', 'RCC-6320', 'RCC-6318',
        'RCC-6316', 'RCC-6315', 'RCC-6201', 'RCC-6199', 'RCC-6197', 'RCC-6194',
        'RCC-6191', 'RCC-6189', 'RCC-6186', 'RCC-6185', 'RCC-6183', 'RCC-6180',
        'RCC-6176', 'RCC-6175', 'RCC-6172', 'RCC-6169', 'RCC-6167', 'RCC-6164',
        'RCC-6163', 'RCC-6160', 'RCC-6158', 'RCC-6156', 'RCC-6153', 'RCC-6151',
        'RCC-6150', 'RCC-6147', 'RCC-6145', 'RCC-6142', 'RCC-6141', 'RCC-6140',
        'RCC-6136', 'RCC-6134', 'RCC-6133', 'RCC-6131', 'RCC-6129', 'RCC-6128',
        'RCC-6126', 'RCC-6123', 'RCC-6120', 'RCC-6119', 'RCC-6117', 'RCC-6115',
        'RCC-6112', 'RCC-6108', 'RCC-6106', 'RCC-6105', 'RCC-6103', 'RCC-6100',
        'RCC-6097', 'RCC-6096', 'RCC-6093', 'RCC-6090', 'RCC-6088', 'RCC-6085',
        'RCC-6082', 'RCC-6079', 'RCC-6074', 'RCC-6072', 'RCC-6068', 'RCC-6065',
        'RCC-6061', 'RCC-6057', 'RCC-6050', 'RCC-6048', 'RCC-6044', 'RCC-6039',
        'RCC-6035', 'RCC-6031', 'RCC-6027', 'RCC-6023', 'RCC-6020', 'RCC-6015',
        'RCC-6012', 'RCC-6007', 'RCC-6004', 'RCC-6001', 'RCC-5997', 'RCC-5995',
        'RCC-5992', 'RCC-5988', 'RCC-5985', 'RCC-5981', 'RCC-5976', 'RCC-5973', 'RCC-5968',
        'RCC-5965', 'RCC-5961', 'RCC-5959', 'RCC-5956', 'RCC-5953', 'RCC-5947',
        'RCC-5944', 'RCC-5940', 'RCC-5936', 'RCC-5934', 'RCC-5931', 'RCC-5929',
        'RCC-5924', 'RCC-5921', 'RCC-5918', 'RCC-5912', 'RCC-5908', 'RCC-5904',
        'RCC-5899', 'RCC-5896', 'RCC-5893', 'RCC-5889', 'RCC-5886', 'RCC-5883',
        'RCC-5878', 'RCC-5875', 'RCC-5872', 'RCC-5871', 'RCC-5866', 'RCC-5863',
        'RCC-5857', 'RCC-5855', 'RCC-5852', 'RCC-5847', 'RCC-5845', 'RCC-5842',
        'RCC-5838', 'RCC-5833', 'RCC-5828', 'RCC-5824', 'RCC-5820', 'RCC-5817',
        'RCC-5812', 'RCC-5811', 'RCC-5809', 'RCC-5803', 'RCC-5798', 'RCC-5793',
        'RCC-5789', 'RCC-5787', 'RCC-5784', 'RCC-5780', 'RCC-5778', 'RCC-5775',
        'RCC-5773', 'RCC-5767', 'RCC-5763', 'RCC-5759', 'RCC-5753', 'RCC-5750',
        'RCC-5744', 'RCC-5742', 'RCC-5741', 'RCC-5736', 'RCC-5735', 'RCC-5730',
        'RCC-5726', 'RCC-5720', 'RCC-5715', 'RCC-5713', 'RCC-5709', 'RCC-5706',
        'RCC-5704', 'RCC-5702', 'RCC-5700', 'RCC-5695', 'RCC-5692', 'RCC-5690',
        'RCC-5688', 'RCC-5684', 'RCC-5680', 'RCC-5679', 'RCC-5675', 'RCC-5672',
        'RCC-5668', 'RCC-5662', 'RCC-5660', 'RCC-5657', 'RCC-5652', 'RCC-5648',
        'RCC-5644', 'RCC-5641', 'RCC-5638', 'RCC-5634', 'RCC-5632', 'RCC-5630',
        'RCC-5626', 'RCC-5622', 'RCC-5619', 'RCC-5615', 'RCC-5614', 'RCC-5612',
        'RCC-5609', 'RCC-5603', 'RCC-5601', 'RCC-5600', 'RCC-5594', 'RCC-5592',
        'RCC-5589', 'RCC-5586', 'RCC-5583', 'RCC-5579', 'RCC-5574', 'RCC-5569',
        'RCC-5567', 'RCC-5562', 'RCC-5557', 'RCC-5551', 'RCC-5548', 'RCC-5542',
        'RCC-5539', 'RCC-5537', 'RCC-5534', 'RCC-5530', 'RCC-5528', 'RCC-5526',
        'RCC-5522', 'RCC-5519', 'RCC-5516', 'RCC-5512', 'RCC-5509', 'RCC-5505',
        'RCC-5501', 'RCC-5497', 'RCC-5496', 'RCC-5492', 'RCC-5487', 'RCC-5486',
        'RCC-5481', 'RCC-5479', 'RCC-5476', 'RCC-5473', 'RCC-5470', 'RCC-5468',
        'RCC-5465', 'RCC-5464', 'RCC-5460', 'RCC-5455', 'RCC-5452', 'RCC-5446',
        'RCC-5445', 'RCC-5439', 'RCC-5437', 'RCC-5435', 'RCC-5428', 'RCC-5425',
        'RCC-5423', 'RCC-5420', 'RCC-5417', 'RCC-5415', 'RCC-5413', 'RCC-5408',
        'RCC-5404', 'RCC-5402', 'RCC-5398', 'RCC-5395', 'RCC-5394', 'RCC-5392',
        'RCC-5389', 'RCC-5385', 'RCC-5382', 'RCC-5379', 'RCC-5378', 'RCC-5377',
        'RCC-5375', 'RCC-5373', 'RCC-5369', 'RCC-5365', 'RCC-5361', 'RCC-5358',
        'RCC-5356', 'RCC-5355', 'RCC-5351', 'RCC-5350', 'RCC-5346', 'RCC-5343',
        'RCC-5338', 'RCC-5335', 'RCC-5333', 'RCC-5331', 'RCC-5328', 'RCC-5326',
        'RCC-5324', 'RCC-5321', 'RCC-5319', 'RCC-5316', 'RCC-5313', 'RCC-5310',
        'RCC-5308', 'RCC-5307']
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
