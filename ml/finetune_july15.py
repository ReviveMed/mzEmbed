
import os
import optuna
import pandas as pd
from prep_run import create_selected_data, make_kwargs_set, get_task_head_kwargs, round_kwargs_to_sig, flatten_dict, unflatten_dict
from utils_neptune import get_latest_dataset, get_run_id_list, get_run_id_list_from_query
from setup3 import setup_neptune_run
import time
from prep_study2 import objective_func4, reuse_run, get_study_objective_keys, get_study_objective_directions, add_runs_to_study
from prep_run import get_selection_df, convert_model_kwargs_list_to_dict, convert_distributions_to_suggestion

## 
# %% Load the latest data

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

project_id = 'revivemed/RCC'


USE_WEBAPP_DB = True
SAVE_TRIALS = True
ADD_EXISTING_RUNS_TO_STUDY = False
limit_add = -1 # limit the number of runs added to the study

encoder_kind = 'VAE'


# %%





def main(run_id):

    neptune_api_token = NEPTUNE_API_TOKEN
    homedir = os.path.expanduser("~")
    input_data_dir = f'{homedir}/INPUT_DATA'
    os.makedirs(input_data_dir, exist_ok=True)
    input_data_dir = get_latest_dataset(data_dir=input_data_dir,api_token=NEPTUNE_API_TOKEN,project=project_id)

    selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv',index_col=0)

    output_dir = f'{homedir}/PROCESSED_DATA'
    os.makedirs(output_dir, exist_ok=True)
    subdir_col = 'Study ID'

    fit_subset_col = 'Pretrain Discovery Train'
    eval_subset_col = 'Pretrain Discovery Val'
    setup_id = 'pretrain'

    finetune_fit_subset_col = 'Finetune Discovery Train'
    finetune_eval_subset_col = 'Finetune Discovery Val'

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

    X_eval_file = f'{output_dir}/X_{eval_file_id}.csv'
    y_eval_file = f'{output_dir}/y_{eval_file_id}.csv'
    X_fit_file = f'{output_dir}/X_{fit_file_id}.csv'
    y_fit_file = f'{output_dir}/y_{fit_file_id}.csv'

    X_finetune_eval_file = f'{output_dir}/X_{finetune_eval_file_id}.csv'
    y_finetune_eval_file = f'{output_dir}/y_{finetune_eval_file_id}.csv'
    X_finetune_fit_file = f'{output_dir}/X_{finetune_fit_file_id}.csv'
    y_finetune_fit_file = f'{output_dir}/y_{finetune_fit_file_id}.csv'

    # Determine the Task Heads
    plot_latent_space_cols = []
    y_head_cols = []
    y_adv_cols = []
    head_kwargs_dict = {}
    adv_kwargs_dict = {}

    head_kwargs_dict2 = {}
    head_kwargs_dict2['Both-OS'], y_head_cols2 = get_task_head_kwargs(head_kind='Cox',
                                                        y_head_col='OS',
                                                        y_cols=[],
                                                        head_name='Both-OS')
    
    # setup_num = 3
    # num_repeats=5
    # restart_run = True
    # restart_rand_run = True
    # remove_y_nans = True
    # finetune_kwargs = make_kwargs_set(encoder_kind=encoder_kind,
    #                                   head_kwargs_dict = head_kwargs_dict2,
    #                                 num_epochs=25,
    #                                 batch_size=32,
    #                                 noise_factor=0.2,
    #                                 dropout_rate=0.25,
    #                                 encoder_weight=0.5,
    #                                 task_num_hidden_layers=1,
    #                                 head_weight=1.0,
    #                                 weight_decay=0.001,
    #                                 learning_rate=0.0005)


    setup_num = 1
    num_repeats=10
    restart_run = False
    restart_rand_run = False
    remove_y_nans = False
    finetune_kwargs = make_kwargs_set(encoder_kind=encoder_kind,
                                      head_kwargs_dict = head_kwargs_dict2,
                                    num_epochs=70,
                                    batch_size=64,
                                    noise_factor=0.2,
                                    dropout_rate=0.0,
                                    encoder_weight=0,
                                    task_num_hidden_layers=1,
                                    head_weight=1.0,
                                    weight_decay=0.0001,
                                    learning_rate=0.0005)


    print(finetune_kwargs)
    try:
        _ = setup_neptune_run(input_data_dir,
                                    setup_id=f'both-OS finetune v{setup_num}',
                                    project_id=project_id,

                                    neptune_mode='async',
                                    yes_logging = True,
                                    neptune_api_token=neptune_api_token,
                                    tags=['v4'],
                                    y_head_cols=y_head_cols2,
                                    y_adv_cols=y_adv_cols,
                                    num_repeats=num_repeats,
                                    restart_run = restart_run,

                                    run_training=True,
                                    X_fit_file=X_finetune_fit_file,
                                    y_fit_file=y_finetune_fit_file,
                                    train_name=finetune_fit_file_id,

                                    run_evaluation=True,
                                    X_eval_file=X_finetune_eval_file,
                                    y_eval_file=y_finetune_eval_file,
                                    eval_name=finetune_eval_file_id,

                                    save_latent_space=False,
                                    plot_latent_space_cols=plot_latent_space_cols,
                                    plot_latent_space = '',
                                    
                                    with_run_id=run_id,
                                    # load_model_from_run_id=None,
                                    # load_model_loc = None,
                                    load_encoder_loc= 'pretrain',
                                    remove_y_nans = remove_y_nans,

                                    **finetune_kwargs)

        _ = setup_neptune_run(input_data_dir,
                                    setup_id=f'both-OS randinit v{setup_num}',
                                    project_id=project_id,

                                    neptune_mode='async',
                                    yes_logging = True,
                                    neptune_api_token=neptune_api_token,
                                    tags=['v4'],
                                    y_head_cols=y_head_cols2,
                                    y_adv_cols=y_adv_cols,
                                    num_repeats=num_repeats,
                                    overwrite_existing_kwargs=True,
                                    restart_run = restart_rand_run,

                                    run_training=True,
                                    X_fit_file=X_finetune_fit_file,
                                    y_fit_file=y_finetune_fit_file,
                                    train_name=finetune_fit_file_id,

                                    run_evaluation=True,
                                    X_eval_file=X_finetune_eval_file,
                                    y_eval_file=y_finetune_eval_file,
                                    eval_name=finetune_eval_file_id,

                                    save_latent_space=False,
                                    plot_latent_space_cols=plot_latent_space_cols,
                                    plot_latent_space = '',
                                    
                                    with_run_id=run_id,
                                    # load_model_from_run_id=None,
                                    # load_model_loc = None,
                                    load_encoder_loc= 'pretrain',
                                    run_random_init=True,
                                    remove_y_nans = remove_y_nans,

                                    **finetune_kwargs)

        ###############

        head_kwargs_dict3 = {}
        head_kwargs_dict3['IMDC'], y_head_cols3 = get_task_head_kwargs(head_kind='Binary',
                                                            y_head_col='IMDC BINARY',
                                                            y_cols=[],
                                                            head_name='IMDC')


        finetune_kwargs['head_kwargs_dict'] = head_kwargs_dict3

        _ = setup_neptune_run(input_data_dir,
                                    setup_id=f'IMDC finetune v{setup_num}',
                                    project_id=project_id,

                                    neptune_mode='async',
                                    yes_logging = True,
                                    neptune_api_token=neptune_api_token,
                                    tags=['v4'],
                                    y_head_cols=y_head_cols3,
                                    y_adv_cols=y_adv_cols,
                                    num_repeats=num_repeats,
                                    restart_run = restart_run,

                                    run_training=True,
                                    X_fit_file=X_finetune_fit_file,
                                    y_fit_file=y_finetune_fit_file,
                                    train_name=finetune_fit_file_id,

                                    run_evaluation=True,
                                    X_eval_file=X_finetune_eval_file,
                                    y_eval_file=y_finetune_eval_file,
                                    eval_name=finetune_eval_file_id,

                                    save_latent_space=False,
                                    plot_latent_space_cols=plot_latent_space_cols,
                                    plot_latent_space = '',
                                    
                                    with_run_id=run_id,
                                    # load_model_from_run_id=None,
                                    # load_model_loc = None,
                                    load_encoder_loc= 'pretrain',
                                    remove_y_nans = remove_y_nans,

                                    **finetune_kwargs)

        _ = setup_neptune_run(input_data_dir,
                                    setup_id=f'IMDC randinit v{setup_num}',
                                    project_id=project_id,

                                    neptune_mode='async',
                                    yes_logging = True,
                                    neptune_api_token=neptune_api_token,
                                    tags=['v4'],
                                    y_head_cols=y_head_cols3,
                                    y_adv_cols=y_adv_cols,
                                    num_repeats=num_repeats,
                                    restart_run = restart_rand_run,

                                    run_training=True,
                                    X_fit_file=X_finetune_fit_file,
                                    y_fit_file=y_finetune_fit_file,
                                    train_name=finetune_fit_file_id,

                                    run_evaluation=True,
                                    X_eval_file=X_finetune_eval_file,
                                    y_eval_file=y_finetune_eval_file,
                                    eval_name=finetune_eval_file_id,

                                    save_latent_space=False,
                                    plot_latent_space_cols=plot_latent_space_cols,
                                    plot_latent_space = '',
                                    
                                    with_run_id=run_id,
                                    # load_model_from_run_id=None,
                                    # load_model_loc = None,
                                    load_encoder_loc= 'pretrain',
                                    run_random_init=True,
                                    remove_y_nans = remove_y_nans,

                                    **finetune_kwargs)
    except Exception as e:
        print('Error: ',e)


    return


if __name__ == '__main__':


    query1 = '(`pretrain/original_kwargs/source run_id`:string = "RCC-3213") OR (`pretrain/original_kwargs/source run_id`:string = "RCC-3276")'
    query2 = '(`pretrain/original_kwargs/study_info_dict/study_name`:string = "Recon Minimize July15 v0") AND (`pretrain/avg/Pretrain_Discovery_Val reconstruction_loss`:float < 0.38)'
    
    
    run_id_list1 = get_run_id_list_from_query(query=query1,limit=100,project_id=project_id)
    run_id_list2 = get_run_id_list_from_query(query=query2,limit=100,project_id=project_id)
    # run_id_list = get_run_id_list(encoder_kind='VAE',tag='v4',project_id='revivemed/RCC')
    print(len(run_id_list1))
    print(len(run_id_list2))
    # run_id_list = ['RCC-3323']
    # run_id_list = ['RCC-3183']

    run_id_list = run_id_list1 + run_id_list2

    for run_id in run_id_list:
        print('##############################################')
        print('Running: ',run_id)
        start_time  = time.time()
        main(run_id)
        print('Minutes elapsed: ',(time.time()-start_time)/60)
        # break