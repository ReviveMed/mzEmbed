import os
import pandas as pd
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import pickle
import mysql.connector
from mysql.connector.constants import ClientFlag
from google.cloud import storage
import gcsfs


from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

script_path = os.path.dirname(os.path.abspath(__file__))
# dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_path, '..'))

from study_alignment.utils_targets import peaks_to_targets_wrapper, process_targeted_data
from study_alignment.utils_eclipse import align_ms_studies_with_Eclipse
from study_alignment.utils_metabCombiner import align_ms_studies_with_metabCombiner, create_metaCombiner_grid_search
from study_alignment.mspeaks import (create_mspeaks_from_mzlearn_result, MSPeaks, create_mspeaks_from_mzlearn_result,
                                     load_mspeaks_from_pickle)

from study_alignment.align_pair import align_ms_studies
from met_matching.metabolite_name_matching_main import refmet_query

from study_alignment.utils_misc import change_param_freq_threshold, get_method_param_name
from study_alignment.utils_misc import load_json, save_json, unravel_dict

########################################################################################################################
# Helper Functions
########################################################################################################################
def get_synthetic_norm_func(norm_func_vals, base_func=synthetic_normalization_Nov2023_wrapper_repeat):
    other_kwargs = {}
    for key, val in norm_func_vals.items():
        if key == 'norm_method_name':
            continue
        other_kwargs[key] = val

    def my_cycle_synthetic_norm(peak_intensity, peak_info, sample_info):
        norm_df, _ = base_func(peak_intensity, sample_info, peak_info, **other_kwargs)
        return norm_df

    return my_cycle_synthetic_norm


def get_synthetic_norm_func_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return get_synthetic_norm_func(data)


def save_records_to_json(save_path, record_dct, scores_dict=None):
    # save the results to json
    if scores_dict is not None:
        for key in scores_dict.keys():
            record_dct[key] = scores_dict[key]

    # assume record_dct is your dictionary
    record_dct_converted = {}
    for key, value in record_dct.items():
        if isinstance(value, np.float32):
            record_dct_converted[key] = float(value)
        elif isinstance(value, np.int64):
            record_dct_converted[key] = int(value)
        else:
            record_dct_converted[key] = value

    with open(save_path, 'w') as fp:
        json.dump(record_dct_converted, fp)

    return


def get_norm_func(norm_name, data_dir):
    os.makedirs(os.path.join(data_dir, 'norm_settings'), exist_ok=True)
    norm_func_json_file = os.path.join(data_dir, 'norm_settings', f'{norm_name}.json')

    if 'pool' in norm_name:
        # save_records_to_json(norm_func_json_file,{'norm_method_name':'Map Pool'})
        return orig_pool_map_norm
    elif 'raw' in norm_name:
        # save_records_to_json(norm_func_json_file,{'norm_method_name':'Raw'})
        return None
    elif 'TIC' in norm_name:
        # save_records_to_json(norm_func_json_file,{'norm_method_name':'TIC'})
        return compute_TIC_norm

    if os.path.exists(norm_func_json_file):
        norm_func = get_synthetic_norm_func_from_json(norm_func_json_file)
        return norm_func
    else:
        raise ValueError(f'Norm function {norm_name} not found')


def get_list_available_study_ids(data_path):
    study_id_list = []
    for study_id in os.listdir(data_path):
        if '.' in study_id:
            continue
        if 'SKIP' in study_id:
            continue
        if ('ST' in study_id) or ('MTBL' in study_id):
            study_id_list.append(study_id)
    return study_id_list


def min_max_scale(df):
    overall_min = df.min().min()
    overall_max = df.max().max()
    df = (df - overall_min) / (overall_max - overall_min)
    return df


def standardize_across_cohorts(combined_intensity, cohort_labels, method):
    assert len(combined_intensity.columns) == len(cohort_labels)

    if method == 'combat':
        data_corrected = pycombat_norm(combined_intensity, cohort_labels)
    elif method == 'raw':
        data_corrected = combined_intensity.copy()
    elif method == 'min_max':
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(cohort_labels == cohort)[0]
            cohort_data = data_corrected.iloc[:, cohort_idx].copy()
            cohort_data = min_max_scale(cohort_data)
            data_corrected.iloc[:, cohort_idx] = cohort_data

    elif method == 'zscore_0':
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(cohort_labels == cohort)[0]
            cohort_data = data_corrected.iloc[:, cohort_idx].copy()
            cohort_data = (cohort_data - cohort_data.mean(axis=0)) / cohort_data.std(axis=0)
            cohort_data.fillna(0, inplace=True)
            data_corrected.iloc[:, cohort_idx] = cohort_data
    elif method == 'zscore_1':
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(cohort_labels == cohort)[0]
            cohort_data = data_corrected.iloc[:, cohort_idx].copy()
            cohort_data = (cohort_data - cohort_data.mean(axis=1)) / cohort_data.std(axis=1)
            cohort_data.fillna(0, inplace=True)
            data_corrected.iloc[:, cohort_idx] = cohort_data
    else:
        raise ValueError(f'Invalid method: {method}')

    return data_corrected


########################################################################################################################
# min helper functions
########################################################################################################################
def findOccurrences(s, ch):  # to find position of '/' in blob path ,used to create folders in local storage
    return [i for i, letter in enumerate(s) if letter == ch]


def download_from_bucket(bucket_name, blob_path, local_path):
    # Create this folder locally
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=blob_path))

    startloc = 0
    for blob in blobs:
        startloc = 0
        folderloc = findOccurrences(blob.name.replace(blob_path, ''), '/')
        if (not blob.name.endswith("/")):
            if (blob.name.replace(blob_path, '').find("/") == -1):
                downloadpath = local_path + '/' + blob.name.replace(blob_path, '')
                blob.download_to_filename(downloadpath)
            else:
                for folder in folderloc:

                    if not os.path.exists(local_path + '/' + blob.name.replace(blob_path, '')[startloc:folder]):
                        create_folder = local_path + '/' + blob.name.replace(blob_path, '')[
                                                           0:startloc] + '/' + blob.name.replace(blob_path, '')[
                                                                               startloc:folder]
                        startloc = folder + 1
                        os.makedirs(create_folder)

                downloadpath = local_path + '/' + blob.name.replace(blob_path, '')

                blob.download_to_filename(downloadpath)


def get_mspeak_from_job_id(script_path_mspeak, job_id, freq_th):
    # process the reference study
    query = ("SELECT * FROM app_user_job_status WHERE id = " + str(job_id) + ";")
    cursor.execute(query)
    records = cursor.fetchall()
    user_id = records[0][1]
    job_time = records[0][3]
    # job_time = job_time.strftime("%Y-%m-%d %H:%M:%S")
    mzlearn_path = f"mzlearn/{user_id}/{job_time}"

    # download the mzlearn result needed for the combination
    intermediate_folder_path = f"{mzlearn_path}/mzlearn_intermediate_results"
    bucket_name = 'mzlearn-webapp.appspot.com'
    bucket = client.get_bucket(bucket_name)
    dst_path = f"{script_path_mspeak}/{job_id}/result-"  # local path to the folder to be saved
    # try to donwload the folder from gcp to local
    # if gcp patch exists then download
    # mzlearn-webapp.appspot.com/mzlearn/min/2023-09-29 15:43:35/mzlearn_intermediate_results
    full_intermediate_folder_path = f"mzlearn-webapp.appspot.com/{mzlearn_path}/mzlearn_intermediate_results"
    if fs.exists(full_intermediate_folder_path):
        print("try to download mzlearn_intermediate_results to local")
        # when downloading only need the
        # sample_info/sample_info.csv
        # targets.csv do not have?
        # final_peaks folder
        sample_info_folder_path = f"{intermediate_folder_path}/sample_info"
        final_peaks_folder_path = f"{intermediate_folder_path}/final_peaks"
        download_from_bucket(bucket_name, sample_info_folder_path, f"{dst_path}/sample_info")
        download_from_bucket(bucket_name, final_peaks_folder_path, f"{dst_path}/final_peaks")
    else:
        print("Did not download mzlearn_intermediate_results to local")

    # create mspeaks object for the reference study
    # define input and output paths
    job_result_folder = f"{script_path_mspeak}/{job_id}"
    result_path = f"{script_path_mspeak}/{job_id}/result-"
    metadata_file = f"{script_path_mspeak}/{job_id}/result-/sample_info/sample_info.csv"
    study_mspeak = create_mspeaks_from_mzlearn_result(result_path)

    # apply freq filtering and remove outlier
    # TODO: use removed outliers to build sample_subset
    study_mspeak.apply_freq_th_on_peaks(freq_th=freq_th,
                                        inplace=True,
                                        )

    # use pool_map normalization if possible, else use synthetic normalization
    pool_map_intensity_path = f"{script_path}/{job_id}/result-/final_peaks/intensity_max_pool_map_norm.csv"
    synthetic_norm_intensity_path = f"{script_path}/{job_id}/result-/final_peaks/intensity_max_synthetic_map_norm.csv"
    normalized_intensity_df = pd.DataFrame()
    if os.path.exists(pool_map_intensity_path):
        normalized_intensity_df = pd.read_csv(pool_map_intensity_path, index_col=0)
    else:  # use synthetic normalization if pool_map normalization does not exist
        normalized_intensity_df = pd.read_csv(synthetic_norm_intensity_path, index_col=0)
    # if normalized_intensity_df is not empty, then use it to update the intensity of the origin_study
    if not normalized_intensity_df.empty:
        print("peak intensity added")
        study_mspeak.add_peak_intensity(normalized_intensity_df)

    return study_mspeak


########################################################################################################################
# data base and GCP bucket connection
########################################################################################################################
config = {
    'user': 'mz',
    'password': 'zm6148mz',
    'host': '34.134.200.45',
    'client_flags': [ClientFlag.SSL],
    'ssl_ca': 'server-ca.pem',
    'ssl_cert': 'client-cert.pem',
    'ssl_key': 'client-key.pem',
    'database': 'mzlearn_webapp_DB'
}
cnxn = mysql.connector.connect(**config)
cursor = cnxn.cursor()
# bucket connection
client = storage.Client()
fs = gcsfs.GCSFileSystem()

# TODO: read from a new peak combination job table to retrive the job ids
# TODO: reference_job_id is the job id of the reference study
# TODO: other_job_ids is the job ids of the other studies

# read database job entry
# establish connection
# query database to find the latest open job
# flag 1: open; 2: running; 3: done
query = (
    "SELECT * FROM app_pretrain_peak_combine_job_status WHERE job_status = 1 ORDER BY id DESC LIMIT 1")
cursor.execute(query)
records = cursor.fetchall()

# if found 1 record
if len(records) > 0:
    ####################################################################################################################
    # read the job entry to get job detail
    ####################################################################################################################
    peak_combine_job_id = records[0][0]
    user_id = records[0][2]
    reference_job_id = records[0][3]
    other_job_ids = records[0][4]
    reference_job_freq_th = records[0][5]
    other_job_freq_th = records[0][6]
    alignment_methods = records[0][8]

    alignment_methods = alignment_methods.split(',')
    other_job_ids = other_job_ids.split(',')
    reference_job_freq_th = reference_job_freq_th.split(',')
    other_job_freq_th = other_job_freq_th.split(',')
    other_job_freq_th = [float(i)/100 for i in other_job_freq_th]
    reference_job_freq_th = [float(i)/100for i in reference_job_freq_th]

    # change the job status to 2 meaning running
    job_status = 2
    query = (
        f"UPDATE app_pretrain_peak_combine_job_status SET job_status = %s WHERE id = %s")
    val = (job_status, peak_combine_job_id)
    cursor.execute(query, val)
    cnxn.commit()

    ####################################################################################################################
    # begin the combination between refrerence and other studies
    ####################################################################################################################
    # build ms object settings
    origin_freq_th = reference_job_freq_th
    other_study_freq_th = other_job_freq_th
    fill_na_strat = 'min'
    num_cohorts_thresh = 0.5
    align_save_dir = f"{script_path}/alignment_results"
    cohort_correction_method = 'combat'

    # if no alignment folder exists, create one
    if not os.path.exists(align_save_dir):
        os.makedirs(align_save_dir)

    # # get mspeak object for the reference study
    # reference_job_id = str(reference_job_id)
    # origin_study = get_mspeak_from_job_id(script_path, reference_job_id, origin_freq_th)
    #
    # # fill missing values
    # origin_study.fill_missing_values(method=fill_na_strat)
    # origin_study.add_study_id(reference_job_id, rename_samples=False)
    # origin_study.save_to_pickle(os.path.join(f"{script_path}/{reference_job_id}", f"{reference_job_id}.pkl"))

    ####################################################################################################################
    # TODO move the download and build mspeak object outside of the loop
    ####################################################################################################################

    ####################################################################################################################
    # TODO: add a new loop there to loop through all the other studies frequency threshold
    # TODO: clearup existing code and get read to add new method
    ####################################################################################################################
    grid_search_summary_df = pd.DataFrame()
    for alignment_method in alignment_methods:
        for reference_freq_th in origin_freq_th:
            for freq_th in other_study_freq_th:

                # frequency sub dir
                freq_grid_search_dir_name = f'reference_freq_th_{reference_freq_th}_freq_th_{freq_th}'

                # get mspeak object for the reference study
                reference_job_id = str(reference_job_id)
                origin_study = get_mspeak_from_job_id(script_path, reference_job_id, reference_freq_th)

                # fill missing values
                origin_study.fill_missing_values(method=fill_na_strat)
                origin_study.add_study_id(reference_job_id, rename_samples=False)
                if not os.path.exists(f"{script_path}/{reference_job_id}/{freq_grid_search_dir_name}"):
                    os.makedirs(f"{script_path}/{reference_job_id}/{freq_grid_search_dir_name}")
                origin_study.save_to_pickle(os.path.join(f"{script_path}/{reference_job_id}/{freq_grid_search_dir_name}", f"{reference_job_id}.pkl"))

                # save the alignment results different folder depends on the frequency threshold
                align_save_dir_freq_th = os.path.join(align_save_dir, f'{freq_grid_search_dir_name}')
                # if no alignment folder exists, create one
                if not os.path.exists(align_save_dir_freq_th):
                    os.makedirs(align_save_dir_freq_th)
                for study_id in other_job_ids:
                    # process rest of the jobs and aligin to origin_study
                    # other_study_freq_th = 0.2
                    study_id = str(study_id)
                    # get mspeak object for the study_id study
                    new_study = get_mspeak_from_job_id(script_path, study_id, freq_th)
                    # fill missing values
                    new_study.fill_missing_values(method=fill_na_strat)
                    new_study.add_study_id(study_id, rename_samples=False)
                    if not os.path.exists(f"{script_path}/{study_id}/{freq_grid_search_dir_name}"):
                        os.makedirs(f"{script_path}/{study_id}/{freq_grid_search_dir_name}")
                    new_study.save_to_pickle(os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}", f"{study_id}.pkl"))

                    ############################################################################################################
                    # alignment happens here
                    ############################################################################################################
                    try:
                        align_ms_studies(origin_study,
                                         new_study,
                                         origin_name=reference_job_id,
                                         input_name=study_id,
                                         save_dir=align_save_dir_freq_th)
                    except Exception as e:
                        print(f'Alignment failed for {study_id} with error {e}')

                alignment_df = combine_alignments_in_dir(align_save_dir_freq_th, origin_name=reference_job_id)
                alignment_df.to_csv(os.path.join(align_save_dir_freq_th, 'alignment_df.csv'), index=True)

                # based on alignment_df build the grid_search_summary_df
                # this summary df has the following columns:
                # 1. freq_th
                # 2. peaks from origin study
                # 3. peaks from other study (each study has a column)
                # 4. common peaks found in all studies
                print(alignment_df.columns)
                dummy_row = {'method': alignment_method,
                             'reference_cohort_freq_threshold': f"{int(reference_freq_th*100)}%",
                             'remaining_cohorts_freq_threshold': f"{int(freq_th*100)}%",
                             'common_peaks': (~alignment_df.isna()).all(axis=1).sum(), # common peaks is the count of rows from alignment_df that has no nan
                             'reference_cohort_peaks': alignment_df.shape[0]}
                for study_id in other_job_ids:
                    study_id = str(study_id)
                    dummy_row[f"remaining_cohort_{study_id}_peaks"] = alignment_df['Compound_ID_' + str(study_id)].count()

                dummy_row['common_peaks'] = (~alignment_df.isna()).all(axis=1).sum()
                grid_search_summary_df = grid_search_summary_df.append(dummy_row, ignore_index=True)

                ################################################################################################################
                # rename the features in each study to correspond to the origin, remove the features that do not align to origin
                ################################################################################################################
                for study_id in other_job_ids:
                    result_path = os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}", f"{study_id}.pkl")
                    input_study_pkl_file = os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}", f"{study_id}.pkl")
                    renamed_study_pkl_file = os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}", f"{study_id}_renamed.pkl")
                    if os.path.exists(input_study_pkl_file):
                        input_study = load_mspeaks_from_pickle(input_study_pkl_file)
                        input_alignment = alignment_df['Compound_ID_' + str(study_id)].copy()
                        input_alignment.dropna(inplace=True)

                        current_ids = input_alignment.values
                        new_ids = input_alignment.index
                        input_study.rename_selected_peaks(current_ids, new_ids, verbose=False)
                        input_study.save_to_pickle(renamed_study_pkl_file)

                ############################################################################################################
                # do stats on alignment results
                ############################################################################################################
                # save the alignment results to a csv on gcp bucket
                # mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{job_id}/alignment_df.csv
                # TODO: save the alignment results to a csv on gcp bucket based on threshold as well
                # peak_combine_job_id = 1

                gcp_file_path = f"mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/{freq_grid_search_dir_name}/alignment_df.csv"
                with fs.open(gcp_file_path, 'w') as f:
                    alignment_df.to_csv(f, index=True)

                num_studies = alignment_df.shape[1]
                print(f'Number of studies: {num_studies}')
                num_origin_feats = alignment_df.shape[0]
                print(f'Number of Origin features: {num_origin_feats}')

                feat_count = (~alignment_df.isna()).sum(axis=1)
                plt.hist(feat_count)
                plt.xlabel('Number of studies')
                plt.ylabel('Number of features')
                plt.title('Number of features detected in each study')
                plt.savefig(os.path.join(align_save_dir_freq_th, 'num_features_matched_across_studies.png'))
                print(f'Number of features detected in all studies: {np.sum(feat_count == num_studies)}')

                ########################################################################################################################
                # based on the alignment results, do peak combination
                ########################################################################################################################
                # TODO: add save here based on other job frequency threshold
                feat_thresh_name = f'num_cohorts_thresh_{num_cohorts_thresh}'
                chosen_feats = alignment_df.index[feat_count >= num_studies * num_cohorts_thresh].tolist()
                print(f'Number of features selected: {len(chosen_feats)}')

                select_feats_dir = os.path.join(align_save_dir_freq_th, feat_thresh_name)
                os.makedirs(select_feats_dir, exist_ok=True)

                # load the reference study first and do log2 transformation
                combined_study = origin_study.peak_intensity.loc[chosen_feats, :].copy()
                combined_study = np.log2(combined_study)
                # for each other study, load the peak intensity and do log2 transformation
                for study_id in other_job_ids:
                    print(study_id)
                    print(freq_grid_search_dir_name)
                    # input_study_pkl_file = os.path.join(output_dir, f'{result_name}_renamed.pkl')
                    input_study_pkl_file = os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}", f"{study_id}_renamed.pkl")
                    if os.path.exists(input_study_pkl_file):
                        input_study = load_mspeaks_from_pickle(input_study_pkl_file)
                        subset_chosen = [i for i in chosen_feats if i in input_study.peak_intensity.index]
                        input_peaks = input_study.peak_intensity.loc[subset_chosen, :].copy()
                        input_peaks = np.log2(input_peaks)
                        # input_peaks = min_max_scale(input_peaks)
                        # combined_study = combined_study.join(input_peaks, lsuffix=reference_job_id, rsuffix=study_id, how='outer')
                        combined_study = combined_study.join(input_peaks, how='outer')

                # Verify that the sample names are the same in the combined study as they are in the combined metadata
                combined_study.fillna(combined_study.mean(), inplace=True)
                combined_study.to_csv(os.path.join(align_save_dir_freq_th, 'combined_study.csv'))

                # correct for cohort effects
                # cohort_labels = combined_meta_data['file_name'].to_list()
                # data_corrected = standardize_across_cohorts(combined_study, cohort_labels, method=cohort_correction_method)

                # save the combined peak intensity to gcp bucket
                # mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{job_id}/alignment_df.csv
                gcp_file_path = f"mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/{freq_grid_search_dir_name}/combined_study.csv"
                with fs.open(gcp_file_path, 'w') as f:
                    combined_study.to_csv(f, index=True)
                print("combined peak intensity saved to gcp bucket")

    # save the grid search summary to gcp bucket
    print(grid_search_summary_df)
    grid_search_summary_df.to_csv(os.path.join(align_save_dir, 'grid_search_summary_df.csv'), index=False)
    gcp_file_path = f"mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/grid_search_summary_df.csv"
    with fs.open(gcp_file_path, 'w') as f:
        grid_search_summary_df.to_csv(f, index=True)

    ####################################################################################################################
    # update the database
    ####################################################################################################################
    job_status = 3
    query = (
        f"UPDATE app_pretrain_peak_combine_job_status SET job_status = %s WHERE id = %s")
    val = (job_status, peak_combine_job_id)
    cursor.execute(query, val)
    cnxn.commit()