# Create table summary and upload to GCP
import optuna
import os
from utils_gcp import upload_file_to_bucket, download_file_from_bucket, \
    check_file_exists_in_bucket, upload_path_to_bucket, get_list_of_files_in_bucket
from misc import save_json
import pandas as pd
import numpy as np
import os

BASE_DIR = '/DATA'
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'
USE_WEBAPP_DB = True
storage_name = 'optuna'
DATA_DIR = f'{BASE_DIR}/data'
GCP_SAVE_LOC = 'March_12_Data'

#######################################
### Analysis of the top trials


def convert_date_columns(df):
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col])
    return df


def convert_duration_to_minutes(df):
    df['duration (mins)'] = df['duration'].apply(lambda x: pd.Timedelta(x).total_seconds() / 60)
    return df


def summarize_params_columns(df):
    result = {}
    for col in df.columns:
        if col.startswith('params_'):
            if pd.api.types.is_numeric_dtype(df[col]):
                result[col] = df[col].median()
            else:
                result[col] = df[col].mode()[0]
    return result


def summarize_user_attrs_columns(df):
    result = {}
    for col in df.columns:
        if col.startswith('user_attrs_'):
            if col == 'user_attrs_DEBUG':
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                result[col] = df[col].mean()
                result[col + ' std'] = df[col].std()
            else:
                result[col] = df[col].mode()[0]
    return result



def process_top_trials(df, top_trial_perc=0.05, direction='maximize'):

    if direction == 'maximize':
        df.sort_values(by='value', ascending=False, inplace=True)
    else:
        df.sort_values(by='value', ascending=True, inplace=True)

    # only look at the completed runs
    df = df[df['state'] == 'COMPLETE'].copy()

    if 'user_attrs_DEBUG' in df.columns:
        df.fillna({'user_attrs_DEBUG': False}, inplace=True)
        df = df[df['user_attrs_DEBUG'] == False].copy()
    print('After removing failed, pruned and debug trials,', df.shape[0], ' remain')

    num_trials = df.shape[0]
    top_trials = int(np.floor(num_trials*top_trial_perc))
    if top_trials < 3:
        top_trials = 3
    print('look at the top', top_trials, 'trials')


    df = df.head(top_trials).copy()
    df = convert_date_columns(df)
    df = convert_duration_to_minutes(df)
    res = summarize_params_columns(df)
    res.update(summarize_user_attrs_columns(df))
    res['total trials'] = num_trials
    res['top trials'] = top_trials
    res['duration (mins)'] = df['duration (mins)'].mean()

    return res


def compile_top_trials_from_all_studies(gcp_save_loc=GCP_SAVE_LOC, data_dir=DATA_DIR):
    if data_dir[-1] != '/':
        data_dir += '/'

    # Get a list of all files in the GCP bucket
    files = get_list_of_files_in_bucket(gcp_save_loc)

    # Filter the files to only include JSON files
    json_files = [file for file in files if file.endswith('toptrials_summary.json')]
    # remove the leading gcp_save_loc
    # json_files = [file.replace(f'{gcp_save_loc}/', '') for file in json_files]
    # print(json_files)

    # Download the JSON files to the data directory
    for file in json_files:
        if not os.path.exists(f'{data_dir}{file}'):
            download_file_from_bucket(gcp_save_loc, file, local_path=data_dir)
        else:
            print(f'{data_dir}{file} already exists')


    all_summary = {}
    for file in json_files:
        study_name = file.split('_toptrials_summary')[0]
        print(study_name)
        summary = pd.read_json(f'{data_dir}{file}', typ='series')
        all_summary[study_name] = summary

    all_summary_df = pd.DataFrame(all_summary).T
    return all_summary_df

#######################################

if __name__ == '__main__':


    study_name = input("Enter the study name: ")

    gcp_save_loc = GCP_SAVE_LOC
    data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    if USE_WEBAPP_DB:
        print('using webapp database')
        storage_name = WEBAPP_DB_LOC
    else:
        print('use local sqlite database downloaded from GCP bucket')
        # save the study in a sqlite database located in result_dir
        storage_loc = f'{data_dir}/{storage_name}.db'
        if not os.path.exists(storage_loc):
            print("checking if study exists on GCP")
            if check_file_exists_in_bucket(gcp_save_loc, f'{storage_name}.db'):
                print("downloading study from GCP")
                download_file_from_bucket(gcp_save_loc, f'{storage_name}.db', local_path=data_dir)

        storage_name = f'sqlite:///{storage_loc}'


    if study_name == 'all':

        study_name_list = ['MSKCC BINARY_study_march13_S_Clas','MSKCC BINARY_study_march13_L_Clas',\
                           'MSKCC BINARY_study_march13_S_Adv','MSKCC BINARY_study_march13_L_Adv',\
                            'MSKCC BINARY_study_march13_S_Reg','MSKCC BINARY_study_march13_L_Reg',\
                            'BINARY_study_march12_S_TGEM',
                            'MSKCC BINARY_study_march12_S_Clas','MSKCC BINARY_study_march12_L_Clas',\
                           'MSKCC BINARY_study_march12_S_Adv','MSKCC BINARY_study_march12_L_Adv',\
                            'MSKCC BINARY_study_march12_S_Reg','MSKCC BINARY_study_march12_L_Reg',
        ]
    else:
        study_name_list = [study_name]


    for study_name in study_name_list:
        study = optuna.create_study(direction="maximize",
                                    study_name=study_name, 
                                    storage=storage_name, 
                                    load_if_exists=True)
        

        study_table_path = f'{data_dir}/{study_name}_table.csv'
        
        # Create a table of the study in csv format
        study_table = study.trials_dataframe()
        # study_table.to_csv('study_table.csv', index=False)

        study_table.to_csv(study_table_path, index=False)


        # Create a summary of the top trials
        try:
            summary = process_top_trials(study_table, top_trial_perc=0.05, direction='maximize')
            save_json(summary, f'{data_dir}/{study_name}_toptrials_summary.json')


            upload_file_to_bucket(study_table_path, gcp_save_loc)
            upload_file_to_bucket(f'{data_dir}/{study_name}_toptrials_summary.json', gcp_save_loc)
        # except KeyError 
        except KeyError as e:
            print(e)
            continue
        


    all_study_summary = compile_top_trials_from_all_studies(gcp_save_loc, data_dir)
    all_study_summary.to_csv(f'{data_dir}/all_study_summary.csv')
    upload_file_to_bucket(f'{data_dir}/all_study_summary.csv', gcp_save_loc)
    # all_study_summary.to_csv('all_study_summary.csv')
