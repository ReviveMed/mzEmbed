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
        "RCC-10001", "RCC-10005", "RCC-10008", "RCC-10011", "RCC-10014", "RCC-10019", "RCC-10021", "RCC-10024",
        "RCC-10027", "RCC-10030", "RCC-10034", "RCC-10038", "RCC-10041", "RCC-10045", "RCC-10047", "RCC-10050",
        "RCC-10054", "RCC-10057", "RCC-10060", "RCC-10063", "RCC-10067", "RCC-10068", "RCC-10073", "RCC-10076",
        "RCC-10079", "RCC-10083", "RCC-10084", "RCC-10087", "RCC-10090", "RCC-10093", "RCC-10096", "RCC-10100",
        "RCC-10102", "RCC-10108", "RCC-10113", "RCC-10115", "RCC-10120", "RCC-10123", "RCC-10127", "RCC-10129",
        "RCC-10132", "RCC-10135", "RCC-10137", "RCC-10140", "RCC-10143", "RCC-10146", "RCC-10148", "RCC-10152",
        "RCC-10154", "RCC-10157", "RCC-10160", "RCC-10163", "RCC-10166", "RCC-10168", "RCC-10171", "RCC-10173",
        "RCC-10174", "RCC-10177", "RCC-10180", "RCC-10183", "RCC-10185", "RCC-10188", "RCC-10191", "RCC-10193",
        "RCC-10196", "RCC-10200", "RCC-10204", "RCC-10208", "RCC-10210", "RCC-10214", "RCC-10218", "RCC-10221",
        "RCC-10225", "RCC-10227", "RCC-10230", "RCC-10234", "RCC-10236", "RCC-10240", "RCC-10244", "RCC-10247",
        "RCC-10250", "RCC-10255", "RCC-10258", "RCC-10260", "RCC-10264", "RCC-10268", "RCC-10271", "RCC-10273",
        "RCC-10277", "RCC-10279", "RCC-10282", "RCC-10285", "RCC-10288", "RCC-10292", "RCC-10295", "RCC-10297",
        "RCC-10302", "RCC-10304", "RCC-10307", "RCC-10309", "RCC-10312", "RCC-10316", "RCC-10320", "RCC-10322",
        "RCC-10326", "RCC-10329", "RCC-10332", "RCC-10334", "RCC-10339", "RCC-10342", "RCC-10345", "RCC-10348",
        "RCC-10351", "RCC-10354", "RCC-10356", "RCC-10358", "RCC-10361", "RCC-10364", "RCC-10367", "RCC-10370",
        "RCC-10373", "RCC-10376", "RCC-10378", "RCC-10381", "RCC-10385", "RCC-10388", "RCC-10391", "RCC-10394",
        "RCC-10397", "RCC-10401", "RCC-10403", "RCC-10406", "RCC-10409", "RCC-10412", "RCC-10416", "RCC-10419",
        "RCC-10422", "RCC-10425", "RCC-10426", "RCC-10429", "RCC-10433", "RCC-10436", "RCC-10439", "RCC-10443",
        "RCC-10445", "RCC-10449", "RCC-10451", "RCC-10454", "RCC-10458", "RCC-10461", "RCC-10464", "RCC-10468",
        "RCC-10470", "RCC-10474", "RCC-10475", "RCC-10478", "RCC-10480", "RCC-10482", "RCC-10487", "RCC-10490",
        "RCC-10494", "RCC-10499", "RCC-10500", "RCC-10502", "RCC-10506", "RCC-10509", "RCC-10512", "RCC-10515",
        "RCC-10517", "RCC-10520", "RCC-10521", "RCC-10522", "RCC-10525", "RCC-10526", "RCC-10529", "RCC-10532",
        "RCC-10534", "RCC-10536", "RCC-10539", "RCC-10540", "RCC-10543", "RCC-10545", "RCC-10547", "RCC-10550",
        "RCC-10551", "RCC-10553", "RCC-10555", "RCC-10558", "RCC-10560", "RCC-10562", "RCC-10564", "RCC-10566",
        "RCC-10568", "RCC-10570", "RCC-10573", "RCC-10575", "RCC-10576", "RCC-10579", "RCC-10580", "RCC-10583",
        "RCC-10584", "RCC-10586", "RCC-10589", "RCC-10591", "RCC-10593", "RCC-10596", "RCC-10598", "RCC-10600",
        "RCC-10603", "RCC-10605", "RCC-10607", "RCC-10610", "RCC-10612", "RCC-10614", "RCC-10617", "RCC-10618",
        "RCC-10621", "RCC-10624", "RCC-10625", "RCC-10627", "RCC-10629", "RCC-10630", "RCC-10633", "RCC-10635",
        "RCC-10637", "RCC-10639", "RCC-10641", "RCC-10643", "RCC-10644", "RCC-10646", "RCC-10648", "RCC-10649",
        "RCC-10652", "RCC-10653", "RCC-10656", "RCC-10657", "RCC-10659", "RCC-10661", "RCC-10663", "RCC-10665",
        "RCC-10666", "RCC-10669", "RCC-10671", "RCC-10672", "RCC-10674", "RCC-10677", "RCC-10678", "RCC-10679",
        "RCC-10682", "RCC-10683", "RCC-10686", "RCC-10688", "RCC-10689", "RCC-10691", "RCC-10694", "RCC-10695",
        "RCC-10696", "RCC-10700", "RCC-10702", "RCC-10704", "RCC-10706", "RCC-10708", "RCC-10711", "RCC-10714",
        "RCC-10718", "RCC-10722", "RCC-10726", "RCC-10729", "RCC-10733", "RCC-10735", "RCC-10739", "RCC-10742",
        "RCC-10747", "RCC-10749", "RCC-10754", "RCC-10755", "RCC-10758", "RCC-10764", "RCC-10767", "RCC-10771",
        "RCC-10775", "RCC-10779", "RCC-10780", "RCC-10785", "RCC-10787", "RCC-10791", "RCC-10795", "RCC-10798",
        "RCC-10802", "RCC-10805", "RCC-10808", "RCC-10811", "RCC-10815", "RCC-10818", "RCC-10822", "RCC-10826",
        "RCC-10828", "RCC-10833", "RCC-10835", "RCC-10840", "RCC-10842", "RCC-10845", "RCC-10849", "RCC-10852",
        "RCC-10857", "RCC-10859", "RCC-10864", "RCC-10866", "RCC-10870", "RCC-10874", "RCC-10877", "RCC-10880",
        "RCC-10883", "RCC-10887", "RCC-10891", "RCC-10895", "RCC-10898", "RCC-10902", "RCC-10906", "RCC-10909",
        "RCC-10914", "RCC-10916", "RCC-10921", "RCC-10925", "RCC-10928", "RCC-10931", "RCC-10935", "RCC-10939",
        "RCC-10941", "RCC-10945", "RCC-10948", "RCC-10952", "RCC-10956", "RCC-10961", "RCC-10963", "RCC-10967",
        "RCC-10970", "RCC-10973", "RCC-10978", "RCC-10981", "RCC-10986", "RCC-10989", "RCC-10992", "RCC-10994",
        "RCC-10999", "RCC-11001", "RCC-11006", "RCC-11008", "RCC-11011", "RCC-11014", "RCC-11018", "RCC-11022",
        "RCC-11025", "RCC-11027", "RCC-11030", "RCC-11034", "RCC-11036", "RCC-11042", "RCC-11044", "RCC-11049",
        "RCC-11052", "RCC-11055", "RCC-11060", "RCC-11065", "RCC-11068", "RCC-11071", "RCC-11073", "RCC-11077",
        "RCC-11081", "RCC-11083", "RCC-11088", "RCC-11091", "RCC-11096", "RCC-11098", "RCC-11103", "RCC-11106",
        "RCC-11108", "RCC-11113", "RCC-11117", "RCC-11122", "RCC-11127", "RCC-11129", "RCC-11133", "RCC-11136",
        "RCC-11140", "RCC-11143", "RCC-11146", "RCC-11150", "RCC-11153", "RCC-11157", "RCC-11161", "RCC-11163",
        "RCC-11168", "RCC-11171", "RCC-11175", "RCC-11178", "RCC-11182", "RCC-11184", "RCC-11188", "RCC-11190",
        "RCC-11195", "RCC-11197", "RCC-11201", "RCC-11206", "RCC-11210", "RCC-11214", "RCC-11219", "RCC-11220",
        "RCC-11223", "RCC-11227", "RCC-11231", "RCC-11234", "RCC-11238", "RCC-11241", "RCC-11244", "RCC-11248",
        "RCC-11252", "RCC-11255", "RCC-11257", "RCC-11261", "RCC-11265", "RCC-11267", "RCC-11270", "RCC-11274",
        "RCC-11277", "RCC-11280", "RCC-11283", "RCC-11287", "RCC-11290", "RCC-11294", "RCC-11297", "RCC-11301",
        "RCC-11303", "RCC-11307", "RCC-11309", "RCC-11313", "RCC-11317", "RCC-11320", "RCC-11323", "RCC-11326",
        "RCC-11329", "RCC-11332", "RCC-11335", "RCC-11338", "RCC-11342", "RCC-11345", "RCC-11348", "RCC-11351",
        "RCC-11354", "RCC-11358", "RCC-11360", "RCC-11364", "RCC-11367", "RCC-11370", "RCC-11373", "RCC-11377",
        "RCC-11380", "RCC-11383", "RCC-11386", "RCC-11387", "RCC-11389", "RCC-11394", "RCC-11396", "RCC-11400",
        "RCC-11403", "RCC-11405", "RCC-11409", "RCC-11411", "RCC-11415", "RCC-11419", "RCC-11422", "RCC-11425",
        "RCC-11429", "RCC-11431", "RCC-11435", "RCC-11437", "RCC-11441", "RCC-11445", "RCC-11447", "RCC-11451",
        "RCC-11452", "RCC-11455", "RCC-11459", "RCC-11461", "RCC-11465", "RCC-11469", "RCC-11471", "RCC-11475",
        "RCC-11477", "RCC-11481", "RCC-11484", "RCC-11487", "RCC-11490", "RCC-11493", "RCC-11496", "RCC-11499",
        "RCC-11502", "RCC-11506", "RCC-11510", "RCC-11513", "RCC-11516", "RCC-11519", "RCC-11520", "RCC-11523",
        "RCC-11526", "RCC-11530", "RCC-11533", "RCC-11536", "RCC-11539", "RCC-11543", "RCC-11546", "RCC-11549",
        "RCC-11552", "RCC-11555", "RCC-11557", "RCC-11562", "RCC-11564", "RCC-11568", "RCC-11571", "RCC-11574",
        "RCC-11578", "RCC-11580", "RCC-11583", "RCC-11586", "RCC-11589", "RCC-11592", "RCC-11594", "RCC-11596",
        "RCC-11599", "RCC-11602", "RCC-11605", "RCC-11609", "RCC-11612", "RCC-11614", "RCC-11618", "RCC-8027",
        "RCC-8038", "RCC-8049", "RCC-8054", "RCC-8059", "RCC-8069", "RCC-8075", "RCC-8084", "RCC-8094", "RCC-8106",
        "RCC-8111", "RCC-8118", "RCC-8126", "RCC-8135", "RCC-8145", "RCC-8158", "RCC-8164", "RCC-8174", "RCC-8180",
        "RCC-8194", "RCC-8209", "RCC-8223", "RCC-8231", "RCC-8241", "RCC-8251", "RCC-8259", "RCC-8261", "RCC-8272",
        "RCC-8280", "RCC-8291", "RCC-8298", "RCC-8307", "RCC-8317", "RCC-8325", "RCC-8336", "RCC-8344", "RCC-8354",
        "RCC-8365", "RCC-8375", "RCC-8383", "RCC-8393", "RCC-8401", "RCC-8410", "RCC-8417", "RCC-8425", "RCC-8434",
        "RCC-8441", "RCC-8448", "RCC-8451", "RCC-8460", "RCC-8468", "RCC-8478", "RCC-8488", "RCC-8500", "RCC-8508",
        "RCC-8518", "RCC-8531", "RCC-8539", "RCC-8554", "RCC-8565", "RCC-8577", "RCC-8588", "RCC-8600", "RCC-8612",
        "RCC-8624", "RCC-8635", "RCC-8645", "RCC-8655", "RCC-8661", "RCC-8673", "RCC-8682", "RCC-8695", "RCC-8703",
        "RCC-8713", "RCC-8720", "RCC-8730", "RCC-8740", "RCC-8752", "RCC-8764", "RCC-8771", "RCC-8782", "RCC-8795",
        "RCC-8804", "RCC-8816", "RCC-8829", "RCC-8840", "RCC-8853", "RCC-8864", "RCC-8875", "RCC-8885", "RCC-8895",
        "RCC-8903", "RCC-8911", "RCC-8921", "RCC-8931", "RCC-8943", "RCC-8954", "RCC-8964", "RCC-8975", "RCC-8986",
        "RCC-8998", "RCC-9000", "RCC-9012", "RCC-9022", "RCC-9031", "RCC-9043", "RCC-9052", "RCC-9061", "RCC-9069",
        "RCC-9078", "RCC-9089", "RCC-9097", "RCC-9106", "RCC-9115", "RCC-9121", "RCC-9127", "RCC-9137", "RCC-9148",
        "RCC-9156", "RCC-9163", "RCC-9168", "RCC-9174", "RCC-9181", "RCC-9187", "RCC-9192", "RCC-9196", "RCC-9200",
        "RCC-9207", "RCC-9210", "RCC-9217", "RCC-9222", "RCC-9226", "RCC-9233", "RCC-9240", "RCC-9245", "RCC-9251",
        "RCC-9255", "RCC-9261", "RCC-9268", "RCC-9273", "RCC-9278", "RCC-9282", "RCC-9288", "RCC-9292", "RCC-9296",
        "RCC-9301", "RCC-9307", "RCC-9313", "RCC-9317", "RCC-9321", "RCC-9327", "RCC-9331", "RCC-9336", "RCC-9341",
        "RCC-9346", "RCC-9351", "RCC-9355", "RCC-9360", "RCC-9365", "RCC-9371", "RCC-9376", "RCC-9380", "RCC-9387",
        "RCC-9392", "RCC-9396", "RCC-9401", "RCC-9405", "RCC-9410", "RCC-9414", "RCC-9417", "RCC-9422", "RCC-9426",
        "RCC-9431", "RCC-9435", "RCC-9442", "RCC-9446", "RCC-9452", "RCC-9457", "RCC-9462", "RCC-9468", "RCC-9474",
        "RCC-9479", "RCC-9482", "RCC-9489", "RCC-9493", "RCC-9498", "RCC-9504", "RCC-9509", "RCC-9512", "RCC-9515",
        "RCC-9518", "RCC-9521", "RCC-9523", "RCC-9528", "RCC-9531", "RCC-9534", "RCC-9536", "RCC-9539", "RCC-9542",
        "RCC-9546", "RCC-9547", "RCC-9551", "RCC-9553", "RCC-9560", "RCC-9561", "RCC-9564", "RCC-9567", "RCC-9569",
        "RCC-9573", "RCC-9576", "RCC-9580", "RCC-9582", "RCC-9586", "RCC-9588", "RCC-9591", "RCC-9594", "RCC-9597",
        "RCC-9601", "RCC-9604", "RCC-9607", "RCC-9609", "RCC-9613", "RCC-9615", "RCC-9619", "RCC-9622", "RCC-9624",
        "RCC-9628", "RCC-9631", "RCC-9636", "RCC-9639", "RCC-9642", "RCC-9645", "RCC-9647", "RCC-9651", "RCC-9655",
        "RCC-9659", "RCC-9663", "RCC-9665", "RCC-9668", "RCC-9673", "RCC-9676", "RCC-9679", "RCC-9683", "RCC-9687",
        "RCC-9690", "RCC-9693", "RCC-9695", "RCC-9698", "RCC-9701", "RCC-9703", "RCC-9707", "RCC-9711", "RCC-9713",
        "RCC-9716", "RCC-9718", "RCC-9722", "RCC-9724", "RCC-9728", "RCC-9732", "RCC-9736", "RCC-9738", "RCC-9741",
        "RCC-9743", "RCC-9746", "RCC-9749", "RCC-9751", "RCC-9755", "RCC-9758", "RCC-9761", "RCC-9766", "RCC-9767",
        "RCC-9771", "RCC-9773", "RCC-9777", "RCC-9782", "RCC-9784", "RCC-9788", "RCC-9792", "RCC-9795", "RCC-9798",
        "RCC-9802", "RCC-9806", "RCC-9808", "RCC-9811", "RCC-9814", "RCC-9818", "RCC-9821", "RCC-9824", "RCC-9825",
        "RCC-9829", "RCC-9832", "RCC-9835", "RCC-9838", "RCC-9842", "RCC-9845", "RCC-9848", "RCC-9851", "RCC-9854",
        "RCC-9858", "RCC-9862", "RCC-9864", "RCC-9867", "RCC-9871", "RCC-9875", "RCC-9879", "RCC-9882", "RCC-9885",
        "RCC-9887", "RCC-9891", "RCC-9895", "RCC-9897", "RCC-9901", "RCC-9905", "RCC-9908", "RCC-9912", "RCC-9916",
        "RCC-9921", "RCC-9925", "RCC-9928", "RCC-9932", "RCC-9935", "RCC-9937", "RCC-9942", "RCC-9944", "RCC-9948",
        "RCC-9950", "RCC-9954", "RCC-9957", "RCC-9961", "RCC-9964", "RCC-9966", "RCC-9971", "RCC-9973", "RCC-9976",
        "RCC-9978", "RCC-9982", "RCC-9984", "RCC-9988", "RCC-9991", "RCC-9993", "RCC-9998"
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
