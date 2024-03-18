# Create table summary and upload to GCP
import optuna
import os
from utils_gcp import upload_file_to_bucket, download_file_from_bucket, \
    check_file_exists_in_bucket, upload_path_to_bucket, get_list_of_files_in_bucket
from misc import save_json
import pandas as pd
import numpy as np
import os

from paretoset import paretoset



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
                if isinstance(df[col].mode()[0], dict):
                    print('issue with ', col)
                else:
                    result[col] = df[col].mode()[0]
    return result



def apply_cutoffs(df, min_cutoffs, max_cutoffs):
    for col, min_cutoff in min_cutoffs.items():
        if col in df.columns:
            df = df[df[col] > min_cutoff]
    for col, max_cutoff in max_cutoffs.items():
        if col in df.columns:
            df = df[df[col] < max_cutoff]
    return df


def pareto_reduction(df, sense_list=None, current_set=None, target_size=None, objective_cols=None):
    if target_size is None:
        target_size = np.floor(0.2*df.shape[0])
    if objective_cols is None:
        objective_cols = df.columns

    if df.shape[0] < target_size*1.5:
        return df

    mask = paretoset(df[objective_cols], sense=sense_list)
    current_set = pd.concat([current_set, df[mask]])
    if current_set.shape[0] < target_size:
        return pareto_reduction(df[~mask], sense_list, current_set, target_size, objective_cols)
    else:
        return current_set

def choose_top_trials(df, top_trial_perc=0.1, directions=None, 
                      min_cutoffs={},max_cutoffs={},objective_cols=None):
    
    if objective_cols is None:
        if 'value' in df.columns:
            objective_cols = ['value']
        else:
            objective_cols = [col for col in df.columns if 'values_' in col]

    
    if directions is None:
        directions = ['maximize'] * len(objective_cols)


    # only look at the completed runs
    df = df[df['state'] == 'COMPLETE'].copy()

    if 'user_attrs_DEBUG' in df.columns:
        df.fillna({'user_attrs_DEBUG': False}, inplace=True)
        df = df[df['user_attrs_DEBUG'] == False].copy()
    print('After removing failed, pruned and debug trials,', df.shape[0], ' remain')

    if df.shape[0] < 10:
        print('only', df.shape[0], 'not enough for analysis')
        raise ValueError('not enough trials for analysis')

    # choose how many top trials should be considered
    top_trials_count = int(np.floor(df.shape[0]*top_trial_perc))
    if top_trials_count < 5:
        top_trials_count = 5
    print('look at the top', top_trials_count, 'trials')

    # apply the cutoffs
    df = apply_cutoffs(df, min_cutoffs, max_cutoffs)

    print('After applying cutoffs,', df.shape[0], ' remain')

    if len(objective_cols) == 1:
        if directions[0] == 'maximize':
            df.sort_values(by=objective_cols[0], ascending=False, inplace=True)
        else:
            df.sort_values(by=objective_cols[0], ascending=True, inplace=True)
        top_trials = df.head(top_trials_count).copy()
        return top_trials



    sense_list = ['max' if direction == 'maximize' else 'min' for direction in directions]
    top_trials = pareto_reduction(df, 
                                sense_list=sense_list, 
                                target_size=top_trials_count,
                                objective_cols=objective_cols)

    print('found ', top_trials.shape[0], ' trials near the pareto frontier')


    return top_trials


def process_top_trials(df,top_trial_perc=0.1, 
                       directions=None, objective_cols=None,
                       min_cutoffs={},max_cutoffs={}):

    num_trials = df.shape[0]
    print('total number of trials:', num_trials)


    if objective_cols is None:
        if 'value' in df.columns:
            objective_cols = ['value']
        else:
            objective_cols = [col for col in df.columns if 'values_' in col]

    top_trials = choose_top_trials(df, top_trial_perc=top_trial_perc, directions=directions, 
                      min_cutoffs=min_cutoffs,max_cutoffs=max_cutoffs, objective_cols=objective_cols)

    top_trials = convert_date_columns(top_trials)
    top_trials = convert_duration_to_minutes(top_trials)
    res= {}
    res.update({key +' AVG':top_trials[key].mean() for key in objective_cols})
    res.update({key +' MED':top_trials[key].median() for key in objective_cols})
    res.update({key +' MIN':top_trials[key].min() for key in objective_cols})
    res.update({key +' MAX':top_trials[key].max() for key in objective_cols})
    res.update(summarize_params_columns(top_trials))
    res.update(summarize_user_attrs_columns(top_trials))
    res['total trials'] = num_trials
    res['top trials'] = top_trials.shape[0]
    res['duration (mins)'] = top_trials['duration (mins)'].mean()

    return res



def compile_top_trials_from_all_studies(gcp_save_loc=GCP_SAVE_LOC, data_dir=DATA_DIR, chosen_studies=None):
    if data_dir[-1] != '/':
        data_dir += '/'

    if (chosen_studies is not None) and (len(chosen_studies)==0):
        raise ValueError('chosen_studies was given but it is empty')
    
    # Get a list of all files in the GCP bucket
    files = get_list_of_files_in_bucket(gcp_save_loc)

    # Filter the files to only include JSON files
    json_files = [file for file in files if file.endswith('toptrials_summary.json')]
    json_files = [file[1:] if file[0] == '/' else file for file in json_files]
    
    # remove the leading gcp_save_loc
    # json_files = [file.replace(f'{gcp_save_loc}/', '') for file in json_files]
    # print(json_files)

    # Download the JSON files to the data directory
    for file in json_files:
        study_name = file.split('_toptrials_summary')[0]
        if study_name[0] == '/':
            study_name = study_name[1:]
        
        # print(study_name)
        if (chosen_studies is not None) and (study_name not in chosen_studies):
            continue
        
        if not os.path.exists(f'{data_dir}{file}'):
            download_file_from_bucket(gcp_save_loc, file, local_path=data_dir)
        else:
            print(f'{data_dir}{file} already exists')


    all_summary = {}
    for file in json_files:
        study_name = file.split('_toptrials_summary')[0]
        if study_name[0] == '/':
            study_name = study_name[1:]
        
        if (chosen_studies is not None) and (study_name not in chosen_studies):
            continue
            # print('exclude: ', study_name)

        print(study_name)
        summary = pd.read_json(f'{data_dir}{file}', typ='series')
        all_summary[study_name] = summary

    all_summary_df = pd.DataFrame(all_summary).T
    return all_summary_df

#######################################

if __name__ == '__main__':


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

    # Choose optional cutoffs
    min_cutoffs = {}
    max_cutoffs = {}
    # min_cutoffs['value'] = 0.80
    # min_cutoffs['values_0'] = 0.80
    # max_cutoffs['values_1'] = 0.99
    top_trial_perc = 0.1


    # print out a list of the available studies in the database
    print('Available studies:')
    list_available_studies = optuna.study.get_all_study_names(storage=storage_name)
    print(list_available_studies)

    study_name = input("Enter the study name: ")
    if study_name == 'all':
        study_name_list = list_available_studies
    elif study_name not in list_available_studies:
        print(f'search for all studies that contain {study_name}')
        study_name_list = [x for x in list_available_studies if study_name in x]
    else:
        study_name_list = [study_name]


    print('studies to be processed:', study_name_list)
    for study_name in study_name_list:

        print('processing', study_name)

        study = optuna.study.load_study(study_name=study_name, storage=storage_name)
        

        study_table_path = f'{data_dir}/{study_name}_table.csv'
        
        # Create a table of the study in csv format
        study_table = study.trials_dataframe()
        study_table.to_csv('study_table.csv', index=False)

        study_table.to_csv(study_table_path, index=False)


        # Create a summary of the top trials
        try:

            # get the directions from the study
            directions =  [study.directions[0].name for _ in range(len(study.directions))]
            summary = process_top_trials(study_table, top_trial_perc=top_trial_perc, directions=directions,
                                         min_cutoffs=min_cutoffs,max_cutoffs=max_cutoffs)

            save_json(summary, f'{data_dir}/{study_name}_toptrials_summary.json')


            upload_file_to_bucket(study_table_path, gcp_save_loc)
            upload_file_to_bucket(f'{data_dir}/{study_name}_toptrials_summary.json', gcp_save_loc)
        # except KeyError 
        except KeyError as e:
            print(e)
            continue

        except ValueError as e:
            print(e)
            continue
        


    all_study_summary = compile_top_trials_from_all_studies(gcp_save_loc, data_dir, chosen_studies=study_name_list)
    all_study_summary.to_csv(f'{data_dir}/all_study_summary.csv')
    upload_file_to_bucket(f'{data_dir}/all_study_summary.csv', gcp_save_loc)
    # all_study_summary.to_csv('all_study_summary.csv')
