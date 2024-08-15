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
        "RCC-11590", "RCC-11550", "RCC-11529", "RCC-11486", "RCC-11463", "RCC-11426", "RCC-11608", "RCC-11606",
        "RCC-11584", "RCC-11517",
        "RCC-11507", "RCC-11497", "RCC-11466", "RCC-11456", "RCC-11432", "RCC-11406", "RCC-11371", "RCC-11352",
        "RCC-11616", "RCC-11577",
        "RCC-11570", "RCC-11567", "RCC-11558", "RCC-11535", "RCC-11492", "RCC-11449", "RCC-11413", "RCC-11381",
        "RCC-11374", "RCC-11542",
        "RCC-11525", "RCC-11504", "RCC-11346", "RCC-11511", "RCC-11480", "RCC-11442", "RCC-11438", "RCC-11391",
        "RCC-11598", "RCC-11501",
        "RCC-11472", "RCC-11420", "RCC-11416", "RCC-11399", "RCC-11365", "RCC-11357", "RCC-11324", "RCC-11310",
        "RCC-11222", "RCC-11174",
        "RCC-11063", "RCC-11213", "RCC-11200", "RCC-11111", "RCC-11072", "RCC-11064", "RCC-11056", "RCC-11336",
        "RCC-11271", "RCC-11245",
        "RCC-11125", "RCC-11121", "RCC-11078", "RCC-11004", "RCC-10997", "RCC-11330", "RCC-11318", "RCC-11314",
        "RCC-11299", "RCC-11291",
        "RCC-11284", "RCC-11278", "RCC-11259", "RCC-11183", "RCC-11160", "RCC-11155", "RCC-11114", "RCC-11093",
        "RCC-11086", "RCC-11264",
        "RCC-11228", "RCC-11205", "RCC-11047", "RCC-11029", "RCC-11020", "RCC-11012", "RCC-11304", "RCC-11251",
        "RCC-11235", "RCC-11191",
        "RCC-11167", "RCC-11147", "RCC-11138", "RCC-11131", "RCC-11102", "RCC-11037", "RCC-10745", "RCC-10705",
        "RCC-10697", "RCC-10977",
        "RCC-10969", "RCC-10860", "RCC-10800", "RCC-10792", "RCC-10782", "RCC-10776", "RCC-10762", "RCC-10719",
        "RCC-10701", "RCC-10957",
        "RCC-10951", "RCC-10926", "RCC-10920", "RCC-10912", "RCC-10903", "RCC-10875", "RCC-10838", "RCC-10772",
        "RCC-10723", "RCC-10712",
        "RCC-10993", "RCC-10881", "RCC-10868", "RCC-10855", "RCC-10846", "RCC-10821", "RCC-10813", "RCC-10806",
        "RCC-10752", "RCC-10690",
        "RCC-10681", "RCC-10985", "RCC-10962", "RCC-10944", "RCC-10933", "RCC-10897", "RCC-10890", "RCC-10819",
        "RCC-10765", "RCC-10737",
        "RCC-10699", "RCC-10693", "RCC-10974", "RCC-10830", "RCC-10730", "RCC-10685", "RCC-10670", "RCC-10611",
        "RCC-10590", "RCC-10587",
        "RCC-10577", "RCC-10557", "RCC-10546", "RCC-10496", "RCC-10486", "RCC-10485", "RCC-10110", "RCC-10642",
        "RCC-10615", "RCC-10572",
        "RCC-10569", "RCC-10565", "RCC-10542", "RCC-10537", "RCC-10533", "RCC-10505", "RCC-10124", "RCC-10631",
        "RCC-10609", "RCC-10518",
        "RCC-10511", "RCC-10479", "RCC-10471", "RCC-10106", "RCC-10651", "RCC-10622", "RCC-10602", "RCC-10559",
        "RCC-10548", "RCC-10524",
        "RCC-10118", "RCC-10109", "RCC-10675", "RCC-10662", "RCC-10634", "RCC-10626", "RCC-10619", "RCC-10604",
        "RCC-10597", "RCC-10554",
        "RCC-10528", "RCC-10667", "RCC-10658", "RCC-10654", "RCC-10647", "RCC-10636", "RCC-10594", "RCC-10582",
        "RCC-10492", "RCC-10465",
        "RCC-10463", "RCC-10456", "RCC-10405", "RCC-10395", "RCC-10380", "RCC-10306", "RCC-10298", "RCC-10290",
        "RCC-10241", "RCC-10237",
        "RCC-10232", "RCC-10082", "RCC-10072", "RCC-10051", "RCC-10015", "RCC-10423", "RCC-10383", "RCC-10353",
        "RCC-10330", "RCC-10323",
        "RCC-10318", "RCC-10315", "RCC-10300", "RCC-10284", "RCC-10213", "RCC-10207", "RCC-10089", "RCC-10058",
        "RCC-10043", "RCC-10036",
        "RCC-9981", "RCC-9967", "RCC-9953", "RCC-9919", "RCC-10435", "RCC-10415", "RCC-10414", "RCC-10265", "RCC-10253",
        "RCC-10251",
        "RCC-10099", "RCC-10053", "RCC-10000", "RCC-9994", "RCC-9960", "RCC-9946", "RCC-9938", "RCC-9924", "RCC-9915",
        "RCC-9913",
        "RCC-9906", "RCC-9904", "RCC-9892", "RCC-10335", "RCC-10274", "RCC-10226", "RCC-10219", "RCC-10215",
        "RCC-10203", "RCC-10075",
        "RCC-10025", "RCC-10006", "RCC-9997", "RCC-9955", "RCC-9930", "RCC-9878", "RCC-10430", "RCC-10366", "RCC-10360",
        "RCC-10341",
        "RCC-10281", "RCC-10267", "RCC-10256", "RCC-10245", "RCC-10104", "RCC-9975", "RCC-9969", "RCC-9933", "RCC-9910",
        "RCC-9888",
        "RCC-10447", "RCC-10442", "RCC-10399", "RCC-10389", "RCC-10374", "RCC-10347", "RCC-10336", "RCC-10294",
        "RCC-10222", "RCC-10094",
        "RCC-10069", "RCC-10064", "RCC-10039", "RCC-10029", "RCC-10017", "RCC-10009", "RCC-10004", "RCC-9989",
        "RCC-9922", "RCC-9900",
        "RCC-9881", "RCC-9830", "RCC-9757", "RCC-9734", "RCC-9727", "RCC-9721", "RCC-9709", "RCC-9681", "RCC-9669",
        "RCC-9657",
        "RCC-9860", "RCC-9801", "RCC-9783", "RCC-9737", "RCC-9715", "RCC-9869", "RCC-9839", "RCC-9826", "RCC-9813",
        "RCC-9779",
        "RCC-9763", "RCC-9731", "RCC-9692", "RCC-9843", "RCC-9809", "RCC-9768", "RCC-9750", "RCC-9744", "RCC-9699",
        "RCC-9688",
        "RCC-9671", "RCC-9857", "RCC-9850", "RCC-9820", "RCC-9794", "RCC-9791", "RCC-9774", "RCC-9753", "RCC-9706",
        "RCC-9680",
        "RCC-9672", "RCC-9662", "RCC-9874", "RCC-9870", "RCC-9837", "RCC-9803", "RCC-9787", "RCC-9778", "RCC-9759",
        "RCC-9729",
        "RCC-9684", "RCC-9648", "RCC-9620", "RCC-9565", "RCC-9558", "RCC-9535", "RCC-9525", "RCC-9501", "RCC-9473",
        "RCC-9448",
        "RCC-9420", "RCC-9632", "RCC-9627", "RCC-9593", "RCC-9557", "RCC-9554", "RCC-9527", "RCC-9511", "RCC-9486",
        "RCC-9412",
        "RCC-9386", "RCC-9359", "RCC-9637", "RCC-9635", "RCC-9616", "RCC-9610", "RCC-9584", "RCC-9572", "RCC-9556",
        "RCC-9544",
        "RCC-9367", "RCC-9575", "RCC-9568", "RCC-9550", "RCC-9454", "RCC-9428", "RCC-9372", "RCC-9629", "RCC-9603",
        "RCC-9599",
        "RCC-9592", "RCC-9578", "RCC-9520", "RCC-9516", "RCC-9466", "RCC-9437", "RCC-9399", "RCC-9654", "RCC-9652",
        "RCC-9641",
        "RCC-9532", "RCC-9311", "RCC-9237", "RCC-9203", "RCC-9026", "RCC-8879", "RCC-8873", "RCC-8972", "RCC-8826",
        "RCC-9219",
        "RCC-8994", "RCC-8926", "RCC-8770", "RCC-9323", "RCC-9318", "RCC-9257", "RCC-9153", "RCC-9142", "RCC-9108",
        "RCC-9018",
        "RCC-8960", "RCC-9269", "RCC-9050", "RCC-9032", "RCC-8992", "RCC-9300", "RCC-9131", "RCC-9087", "RCC-9080",
        "RCC-8969",
        "RCC-8779", "RCC-8733", "RCC-8620", "RCC-8608", "RCC-8519", "RCC-8495", "RCC-8430", "RCC-8380", "RCC-8334",
        "RCC-8287",
        "RCC-8046", "RCC-9337", "RCC-9180", "RCC-8906", "RCC-8838", "RCC-8658", "RCC-8545", "RCC-8350", "RCC-8274",
        "RCC-8268",
        "RCC-8136", "RCC-9281", "RCC-9264", "RCC-9190", "RCC-9175", "RCC-8847", "RCC-8792", "RCC-8691", "RCC-8646",
        "RCC-8527",
        "RCC-8476", "RCC-8471", "RCC-8405", "RCC-8095", "RCC-8071", "RCC-9348", "RCC-9293", "RCC-9231", "RCC-9215",
        "RCC-8759",
        "RCC-8709", "RCC-8670", "RCC-8627", "RCC-8599", "RCC-8572", "RCC-8552", "RCC-8363", "RCC-8282", "RCC-8224",
        "RCC-8032",
        "RCC-8024", "RCC-10199", "RCC-9242", "RCC-9161", "RCC-9059", "RCC-8936", "RCC-8743", "RCC-8458", "RCC-8233",
        "RCC-8211",
        "RCC-8183", "RCC-8161", "RCC-10202", "RCC-8940", "RCC-8818", "RCC-8579", "RCC-8436", "RCC-8313", "RCC-8254",
        "RCC-8217",
        "RCC-8200", "RCC-8142", "RCC-8113"
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
