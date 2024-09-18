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
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import plotly.express as px
from plotly.offline import plot
import itertools

from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, '..'))

from study_alignment.utils_targets import peaks_to_targets_wrapper, process_targeted_data
from study_alignment.utils_eclipse import align_ms_studies_with_Eclipse
from study_alignment.utils_metabCombiner import align_ms_studies_with_metabCombiner, create_metaCombiner_grid_search
from study_alignment.mspeaks import create_mspeaks_from_mzlearn_result, MSPeaks, load_mspeaks_from_pickle
from study_alignment.align_multi import *
from study_alignment.standardize import standardize_across_cohorts
from study_alignment.align_pair import align_ms_studies
# from met_matching.metabolite_name_matching_main import refmet_query

from study_alignment.utils_misc import change_param_freq_threshold, get_method_param_name
from study_alignment.utils_misc import load_json, save_json, unravel_dict


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


def get_mspeak_from_job_id(script_path_mspeak, job_id, freq_th, max_samples_th):
    # process the reference study
    query = ("SELECT * FROM app_user_job_status WHERE id = " + str(job_id) + ";")
    cursor.execute(query)
    records = cursor.fetchall()
    user_id = records[0][1]
    job_time = records[0][3]
    user_selected_normalization_method = records[0][48]
    # job_time = job_time.strftime("%Y-%m-%d %H:%M:%S")
    mzlearn_path = f"mzlearn/{user_id}/{job_time}"

    # download the mzlearn result needed for the combination
    intermediate_folder_path = f"{mzlearn_path}"
    bucket_name = 'mzlearn-webapp.appspot.com'
    bucket = client.get_bucket(bucket_name)
    dst_path = f"{script_path_mspeak}/{job_id}/result-"  # local path to the folder to be saved
    # try to donwload the folder from gcp to local
    # if gcp patch exists then download
    # mzlearn-webapp.appspot.com/mzlearn/min/2023-09-29 15:43:35/mzlearn_intermediate_results
    full_intermediate_folder_path = f"mzlearn-webapp.appspot.com/{mzlearn_path}"
    if fs.exists(full_intermediate_folder_path):
        print("try to download files to build mspeak object")
        # when downloading only need the
        # sample_info/sample_info.csv
        # targets.csv do not have?
        # final_peaks folder
        sample_info_folder_path = f"{intermediate_folder_path}/sample_info"
        final_peaks_folder_path = f"{intermediate_folder_path}/final_peaks"
        params_overall_fild_path = f"{intermediate_folder_path}/params_overall.csv"
        # mzlearn-webapp.appspot.com/mzlearn/min/2023-12-08 15:41:42/params_overall.csv
        # download this file to local as well
        # check if local file eixts, if not then download
        if not os.path.exists(f"{dst_path}/params_overall.csv"):
            download_from_bucket(bucket_name, sample_info_folder_path, f"{dst_path}/sample_info")
            download_from_bucket(bucket_name, final_peaks_folder_path, f"{dst_path}/final_peaks")
            blob = bucket.blob(params_overall_fild_path)
            blob.download_to_filename(f"{dst_path}/params_overall.csv")
        print("downloaded files to build mspeak object")
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
                                        max_samples_th=max_samples_th, #default None
                                        )

    # use pool_map normalization if possible, else use synthetic normalization
    # use user selected normalization method
    # normalization_methods: hall_mark_normalization,pool_map_normalization,synthetic_map_normalization
    # pool_map_intensity_path = f"{script_path}/{job_id}/result-/final_peaks/intensity_max_pool_map_norm.csv"
    # synthetic_norm_intensity_path = f"{script_path}/{job_id}/result-/final_peaks/intensity_max_synthetic_map_norm.csv"
    if user_selected_normalization_method == 'pool_map_normalization':
        normalized_intensity_peak_list_path = f"{script_path}/{job_id}/result-/final_peaks/intensity_max_pool_map_norm.csv"
    elif user_selected_normalization_method == 'synthetic_map_normalization':
        normalized_intensity_peak_list_path = f"{script_path}/{job_id}/result-/final_peaks/intensity_max_synthetic_map_norm.csv"
    else:
        normalized_intensity_peak_list_path = f"{script_path}/{job_id}/result-/final_peaks/intensity_max_norm.csv"

    normalized_intensity_df = pd.read_csv(normalized_intensity_peak_list_path, index_col=0)
    if not normalized_intensity_df.empty:
        print("peak intensity added")
        study_mspeak.add_peak_intensity(normalized_intensity_df)

    return study_mspeak


def upload_to_gcs(local_directory, bucket_name, gcs_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_directory)
            gcs_file_path = os.path.join(gcs_path, relative_path)

            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file_path)


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
    max_num_samples = records[0][10]

    if max_num_samples:
        max_num_samples = max_num_samples.split(',')
        max_num_samples = [int(i) for i in max_num_samples]
    else:
        max_num_samples = [None]

    alignment_methods = alignment_methods.split(',')
    other_job_ids = other_job_ids.split(',')
    reference_job_freq_th = reference_job_freq_th.split(',')
    other_job_freq_th = other_job_freq_th.split(',')
    other_job_freq_th = [float(i) / 100 for i in other_job_freq_th]
    reference_job_freq_th = [float(i) / 100 for i in reference_job_freq_th]

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

    ####### Add these as new user-defined options ######
    # NOTE: To save time, all of these options can be looped through AFTER the alignment has been computed
    # NOTE (2): the fill na method is currently applied very early in the process before alignment, but it can be moved to after alignment
    fill_na_strat = 'min'
    # fill_na_strat_list = ['min',''mean'','knn','min/2','median','log_mean','log_knn']
    num_cohorts_thresh = 0
    # num_cohorts_thresh_list = [0,0.25,0.5,0.75]
    cohort_correction_method = 'zscore'
    # cohort_correction_method_list = ['combat','zscore','min_max','raw']
    ###### end of new user-defined options ######

    align_save_dir = f"{script_path}/alignment_results"

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
    # get run order mapped to file name
    ####################################################################################################################
    # need to add the run order into the pca_df as well based on file name\
    all_job_sample_info = pd.DataFrame()
    all_job_ids = [reference_job_id] + other_job_ids
    for job_id in all_job_ids:
        # read job detail based on job id
        query = f"SELECT * FROM app_user_job_status WHERE id = {job_id}"
        cursor.execute(query)
        records = cursor.fetchall()

        job_creation_time = records[0][3]
        user_id = records[0][1]

        # mzlearn-webapp.appspot.com/mzlearn/min/2023-10-20 13:55:12/sample_info/sample_info.csv
        job_creation_time = job_creation_time.strftime("%Y-%m-%d %H:%M:%S")
        mzlearn_path = f'mzlearn/{user_id}/{job_creation_time}'
        sample_info_path = f"mzlearn-webapp.appspot.com/{mzlearn_path}/sample_info/sample_info.csv"
        with fs.open(sample_info_path) as f:
            sample_info_df = pd.read_csv(f, sep=",")
        # add a new column to the sample_info_df called "mzlearn_cohort_id", which is the job_id
        sample_info_df["mzlearn_cohort_id"] = job_id
        # append the sample_info_df to all_job_sample_info
        all_job_sample_info = pd.concat([all_job_sample_info, sample_info_df], sort=False,
                                        ignore_index=True)
        # mzlearn-webapp.appspot.com/mzlearn/min/2023-10-20 13:55:12/targets_peaks/name_matched_targets.csv
        # save this file to local storage if it exists
        targets_path = f"mzlearn-webapp.appspot.com/{mzlearn_path}/targets_peaks/name_matched_targets.csv"
        if fs.exists(targets_path):
            if not os.path.exists(f"{script_path}/{job_id}"):
                os.makedirs(f"{script_path}/{job_id}")
            with fs.open(targets_path) as f:
                targets_df = pd.read_csv(f, sep=",")
            targets_df.to_csv(f"{script_path}/{job_id}/targets.csv", index=False)
    print(all_job_sample_info)

    ####################################################################################################################
    # TODO: add a new loop there to loop through all the other studies frequency threshold
    # TODO: clearup existing code and get read to add new method
    ####################################################################################################################
    # new way to to grid search, build the grid search parameters list first
    # then loop through the list to do the alignment and peak combination
    gridsearch_params_combinations = list(itertools.product(alignment_methods,
                                                            origin_freq_th,
                                                            other_study_freq_th,
                                                            max_num_samples,
                                                            ))
    gridsearch_params_combinations = pd.DataFrame(gridsearch_params_combinations,
                                                  columns=['method',
                                                           'reference_cohort_freq_threshold',
                                                           'remaining_cohorts_freq_threshold',
                                                           'max_num_samples',
                                                           ])

    print(f"gridsearch_params_combinations: {gridsearch_params_combinations}")

    for index, row in gridsearch_params_combinations.iterrows():
        alignment_method = row['method']
        reference_freq_th = row['reference_cohort_freq_threshold']
        freq_th = row['remaining_cohorts_freq_threshold']
        max_num_samples = row['max_num_samples']

        # grid search sub dir
        freq_grid_search_dir_name = f"grid_search_index_{index}"

        # get mspeak object for the reference study
        print("get original study data")
        origin_study = get_mspeak_from_job_id(script_path, str(reference_job_id), reference_freq_th, None)

        # fill missing values
        origin_study.fill_missing_values(method=fill_na_strat)
        origin_study.add_study_id(reference_job_id, rename_samples=False)
        if not os.path.exists(f"{script_path}/{reference_job_id}/{freq_grid_search_dir_name}"):
            os.makedirs(f"{script_path}/{reference_job_id}/{freq_grid_search_dir_name}")
        origin_study.save_to_pickle(
            os.path.join(f"{script_path}/{reference_job_id}/{freq_grid_search_dir_name}",
                         f"{reference_job_id}.pkl"))
        origin_peak_obj_path = os.path.join(f"{script_path}/{reference_job_id}/{freq_grid_search_dir_name}",
                                            f"{reference_job_id}.pkl")

        # save the alignment results different folder depends on the frequency threshold
        align_save_dir_freq_th = os.path.join(align_save_dir, f'{freq_grid_search_dir_name}')
        # if no alignment folder exists, create one
        if not os.path.exists(align_save_dir_freq_th):
            os.makedirs(align_save_dir_freq_th)
        # save other studies to folder
        input_peak_obj_path_list = []
        print("get other study data")
        for study_id in other_job_ids:
            print(study_id)
            # process rest of the jobs and aligin to origin_study
            # get mspeak object for the study_id study
            new_study = get_mspeak_from_job_id(script_path, str(study_id), freq_th, max_num_samples)
            # fill missing values
            new_study.fill_missing_values(method=fill_na_strat)
            new_study.add_study_id(study_id, rename_samples=False)
            if not os.path.exists(f"{script_path}/{study_id}/{freq_grid_search_dir_name}"):
                os.makedirs(f"{script_path}/{study_id}/{freq_grid_search_dir_name}")
            new_study.save_to_pickle(
                os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}", f"{study_id}.pkl"))
            input_peak_obj_path_list.append(
                os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}", f"{study_id}.pkl"))

        ########################################################################################################
        # alignment happens here using
        ########################################################################################################
        other_job_ids = [str(i) for i in other_job_ids]
        alignment_df = align_multiple_ms_studies(origin_peak_obj_path, input_peak_obj_path_list,
                                                 align_save_dir_freq_th,
                                                 origin_name=str(reference_job_id),
                                                 input_name_list=other_job_ids,
                                                 alignment_method=alignment_method)
        alignment_df.to_csv(os.path.join(align_save_dir_freq_th, 'alignment_df.csv'), index=True)

        # based on alignment_df add columns to row in the gridsearch_params_combinations
        # this summary df has the following columns:
        # 1. freq_th
        # 2. peaks from origin study
        # 3. peaks from other study (each study has a column)
        # 4. max_num_of_samples
        # 5. common peaks found in all studies
        # edit the row in gridsearch_params_combinations to add new columns
        gridsearch_params_combinations.at[index, 'common_peaks'] = (~alignment_df.isna()).all(axis=1).sum()
        gridsearch_params_combinations.at[index, 'reference_cohort_peaks'] = alignment_df.shape[0]
        for study_id in other_job_ids:
            gridsearch_params_combinations.at[index, f"remaining_cohort_{study_id}_peaks"] = alignment_df[
                str(study_id)].count()

        ########################################################################################################
        # rename the features in each study to correspond to the origin, remove the features that
        # do not align to origin
        ########################################################################################################
        rename_inputs_to_origin(align_save_dir_freq_th,
                                multi_alignment_df=None,
                                input_name_list=None,
                                load_dir=None,
                                input_peak_obj_path_list=input_peak_obj_path_list)

        ########################################################################################################
        # do stats on alignment results
        ########################################################################################################
        # save the alignment results to a csv on gcp bucket
        # mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{job_id}/alignment_df.csv
        # TODO: save the alignment results to a csv on gcp bucket based on threshold as well
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

        ################################################################################################################
        # based on the alignment results, do peak combination
        ################################################################################################################
        feat_thresh_name = f'num_cohorts_thresh_{num_cohorts_thresh}'
        chosen_feats = alignment_df.index[feat_count >= num_studies * num_cohorts_thresh].tolist()
        print(f'Number of features selected: {len(chosen_feats)}')

        # remove nan from chosen_feats
        chosen_feats = [i for i in chosen_feats if i in origin_study.peak_intensity.index]

        select_feats_dir = os.path.join(align_save_dir_freq_th, feat_thresh_name)
        os.makedirs(select_feats_dir, exist_ok=True)

        # load the reference study first and do log2 transformation
        combined_study = origin_study.peak_intensity.loc[chosen_feats, :].copy()
        combined_study_nan_mask = origin_study.missing_val_mask.loc[chosen_feats, :].copy()
        combined_study = np.log2(combined_study)

        # get all the file names for the origin_study
        origin_study_file_list = origin_study.peak_intensity.columns.to_list()
        study_id2file_name = {str(reference_job_id): origin_study_file_list}

        # get the cohort labels for the origin_study
        # read the database entry for the origin_study based on the job_id
        study_id2cohort_label = {}
        try:
            query = f"select cohort_label from app_user_job_status where id = {reference_job_id}"
            cursor.execute(query)
            records = cursor.fetchall()
            study_id2cohort_label[str(reference_job_id)] = records[0][0]
        except Exception as e:
            print(f'Error in reading cohort labels: {e}')
            study_id2cohort_label[str(reference_job_id)] = "NA"

        # for each other study, load the peak intensity and do log2 transformation
        for study_id in other_job_ids:
            print(study_id)
            print(freq_grid_search_dir_name)
            # input_study_pkl_file = os.path.join(output_dir, f'{result_name}_renamed.pkl')
            input_study_pkl_file = os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}",
                                                f"{study_id}_renamed.pkl")
            if os.path.exists(input_study_pkl_file):
                # input_study = load_mspeaks_from_pickle(input_study_pkl_file)
                input_study = MSPeaks()
                input_study.load(input_study_pkl_file)
                input_study_file_list = input_study.peak_intensity.columns.to_list()
                # add a key to study_id2file_name
                study_id2file_name[str(study_id)] = input_study_file_list
                # add the key to study_id2cohort_label
                try:
                    query = f"select cohort_label from app_user_job_status where id = {study_id}"
                    cursor.execute(query)
                    records = cursor.fetchall()
                    study_id2cohort_label[str(study_id)] = records[0][0]
                except Exception as e:
                    print(f'Error in reading cohort labels: {e}')
                    study_id2cohort_label[str(study_id)] = "NA"
                subset_chosen = [i for i in chosen_feats if i in input_study.peak_intensity.index]
                input_peaks = input_study.peak_intensity.loc[subset_chosen, :].copy()
                input_study_nan_mask = input_study.missing_val_mask.loc[subset_chosen, :].copy()
                input_peaks = np.log2(input_peaks)
                # input_peaks = min_max_scale(input_peaks)
                # combined_study = combined_study.join(input_peaks, lsuffix=reference_job_id, rsuffix=study_id, how='outer')
                combined_study = combined_study.join(input_peaks, how='outer')
                combined_study_nan_mask = combined_study_nan_mask.join(input_study_nan_mask, how='outer')

        # Verify that the sample names are the same in the combined study as they are in the combined metadata
        combined_study.fillna(combined_study.mean(), inplace=True)
        combined_study.to_csv(os.path.join(align_save_dir_freq_th, 'combined_study.csv'))

        print(combined_study_nan_mask)
        combined_study_nan_mask.fillna(True, inplace=True)
        combined_study_nan_mask.to_csv(os.path.join(align_save_dir_freq_th, 'combined_study_nan_mask.csv'))
        # combined_study_nan_mask do a transpose to get a new dataframe with file_name as index and peak intensity as columns
        combined_study_nan_mask = combined_study_nan_mask.T
        # calculate the number of missing values for each file (missing value for each row)
        # that is count the true from each row and divide by the number of columns
        combined_study_nan_mask['missing_values'] = combined_study_nan_mask.sum(axis=1) / \
                                                    combined_study_nan_mask.shape[1]
        # reformt the combined_study_nan_mask['missing_values'] to save only first 2 decimal points and convert to percentage
        combined_study_nan_mask['missing_values'] = combined_study_nan_mask['missing_values'].apply(
            lambda x: round(x * 100, 2))
        # add % to combined_study_nan_mask['missing_values'] value
        combined_study_nan_mask['missing_values'] = combined_study_nan_mask['missing_values'].astype(str) + '%'

        # check if combined_study is empty
        if combined_study.empty:
            print("combined_study is empty")
            continue

        # get the study_id2file_name
        print(study_id2file_name.keys)
        # build a metadata_df that has file_name, study_id
        metadata_df = pd.DataFrame()
        # use the columns from combined_study to build the metadata_df file_name column
        metadata_df['file_name'] = combined_study.columns
        # create a new column study_id, that is the study_id of the file_name from study_id2file_name
        metadata_df['mzlearn_cohort_id'] = metadata_df['file_name'].apply(
            lambda x: [k for k, v in study_id2file_name.items() if x in v][0])
        print(metadata_df)
        metadata_df.to_csv(os.path.join(align_save_dir_freq_th, 'combined_study_cohort_ids.csv'))

        # need to know the study id for each study used here use job id for each file
        try:
            cohort_labels = metadata_df['mzlearn_cohort_id'].to_list()
            data_corrected = standardize_across_cohorts(combined_study, cohort_labels,
                                                        method=cohort_correction_method)
            # data_corrected = combined_study
            # calculate number of peaks and number of files from data_corrected
            num_peaks = data_corrected.shape[0]
            num_files = data_corrected.shape[1]
        except Exception as e:
            print(f'Error in standardization: {e}')
            continue

        ########################################################################################################
        # Look at the PCA of the combined study
        ########################################################################################################
        # fill nan with  0
        try:
            data_corrected = data_corrected.fillna(0)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data_corrected.T)
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=data_corrected.columns)
            # edit the metadata_df['mzlearn_cohort_id'].to_list() list to inclucde cohort label
            mzlearn_cohort_ids = metadata_df['mzlearn_cohort_id'].to_list()
            new_mzlearn_cohort_ids = []
            for mzlearn_cohort_id in mzlearn_cohort_ids:
                new_mzlearn_cohort_ids.append(
                    f"{mzlearn_cohort_id} ({study_id2cohort_label[mzlearn_cohort_id]})")
            pca_df['mzlearn_cohort_id'] = new_mzlearn_cohort_ids
            pca_df['file_name'] = metadata_df['file_name'].to_list()
            pca_df['MV percentage'] = combined_study_nan_mask['missing_values'].to_list()
            # based on the file_name, get the run order from all_job_sample_info
            run_order = []
            for file_name in pca_df['file_name']:
                run_order.append(
                    all_job_sample_info[all_job_sample_info['file_name'] == file_name]['run_order'].values[0])
            pca_df['run_order'] = run_order
            pca_df.to_csv(os.path.join(align_save_dir_freq_th, f'pca_df.csv'))

            ########################################################################################################
            # %% Look at the UMAP of the combined study
            reducer = umap.UMAP()
            umap_result = reducer.fit_transform(data_corrected.T)
            umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'], index=data_corrected.columns)
            umap_df['mzlearn_cohort_id'] = new_mzlearn_cohort_ids
            umap_df['file_name'] = metadata_df['file_name'].to_list()
            umap_df['MV percentage'] = combined_study_nan_mask['missing_values'].to_list()
            # umap_df['Study_num'] = metadata_df['Study_num']
            # umap_df['label'] = metadata_df[pretrain_label_col]
            # based on the file_name, get the run order from all_job_sample_info
            run_order = []
            for file_name in umap_df['file_name']:
                run_order.append(
                    all_job_sample_info[all_job_sample_info['file_name'] == file_name]['run_order'].values[0])
            umap_df['run_order'] = run_order
            umap_df.to_csv(os.path.join(align_save_dir_freq_th, f'umap_df.csv'))
        except Exception as e:
            print(f'Error in PCA and UMAP: {e}')

        ########################################################################################################
        # look at the targed peak info from the combined study and plot bar plots based on alignment_df
        ########################################################################################################
        # based on the alignment_df, for each study build a dictionary of reference_study_peak_id2targets
        # print(f"alignment_df is {alignment_df}")
        # drop the rows that have nan in all columns
        alignment_df_path = f"{align_save_dir_freq_th}/alignment_df.csv"
        alignment_df = pd.read_csv(alignment_df_path)
        alignment_df = alignment_df.dropna(how='all')
        print(f"alignment_df is {alignment_df}")

        # based on the column names of alignment_df, get the study_id
        # loop through each column of alignment_df and find the corresponding study_id
        study_id2list_of_peak_matches = {}
        for column in alignment_df.columns:
            # for each study_id, create a list of list of peak matched to a peak id
            list_of_peak_target_matches = []
            # loop through each row of this column
            # get the study_id from the column name
            study_id = column
            # based on the study_id, get the targeted data info from mzlearn run that was saved before
            # local target file path f"{script_path}/{job_id}/targets.csv"
            target_file_path = f"{script_path}/{study_id}/targets.csv"
            if os.path.exists(target_file_path):
                targets_df = pd.read_csv(target_file_path)
                # loop through each row of this column
                for row in alignment_df[column]:
                    peak_id = row
                    # print(f"peak_id is {peak_id}")
                    # find the HMDB_match value in the targets_df from the row with feats == peak_id
                    # if found, then add the HMDB_match value to list_of_peak_target_matches
                    matched_target = "NA"
                    if not pd.isna(peak_id):
                        target_row = targets_df[targets_df['feats'] == peak_id]
                        if not target_row.empty:
                            matched_target = target_row['HMDB_match'].values[0]
                    list_of_peak_target_matches.append(matched_target)

            # read the target file if it exists
            study_id2list_of_peak_matches[study_id] = list_of_peak_target_matches

        # how long each key of study_id2list_of_peak_matches is
        for key, value in study_id2list_of_peak_matches.items():
            print(f"study_id: {key} has {len(value)} peaks matched")

        # start comparison between the studies, each other_job_ids compared to reference_job_id once
        # get the study_id2list_of_peak_matches for the reference_job_id
        reference_study_id_list_of_peak_matches = study_id2list_of_peak_matches[str(reference_job_id)]
        # loop through each study_id in other_job_ids
        alignment_targets_tracking_results = pd.DataFrame()
        for study_id in other_job_ids:
            print(f"other study_id: {study_id}")
            other_study_id_list_of_peak_matches = study_id2list_of_peak_matches[str(study_id)]

            if len(other_study_id_list_of_peak_matches) > 0:
                print(f"this study has targeted peaks")

                # combine reference_study_id_list_of_peak_matches and other_study_id_list_of_peak_matches
                all_peak_matches = reference_study_id_list_of_peak_matches + other_study_id_list_of_peak_matches
                # drop all "NA" and None from all_peak_matches
                all_peak_matches = [i for i in all_peak_matches if i != "NA"]
                # count the number of unique peaks in all_peak_matches
                unique_peaks = len(set(all_peak_matches))

                # compare the reference_study_id_list_of_peak_matches and other_study_id_list_of_peak_matches
                # to see how many peaks are common
                correct_alignment_peaks = 0
                miss_aligned_peaks = 0
                for ref_peak_matches, other_peak_matches in zip(reference_study_id_list_of_peak_matches,
                                                                other_study_id_list_of_peak_matches):
                    # if both ref_peak_matches and other_peak_matches are not "NA" and not None
                    if ref_peak_matches != "NA" and other_peak_matches != "NA":
                        if ref_peak_matches == other_peak_matches:
                            correct_alignment_peaks += 1
                        else:
                            miss_aligned_peaks += 1

                # build a row to add to alignment_targets_tracking_results
                row = {'reference_study_id': str(int(reference_job_id)),
                       'other_study_id': str(int(study_id)),
                       'total_possible_aligned_targets': unique_peaks,
                       'correct_alignment_targets': correct_alignment_peaks,
                       'miss_aligned_peaks': miss_aligned_peaks,
                       'correct_alignment_percentage': correct_alignment_peaks / unique_peaks * 100}
                alignment_targets_tracking_results = alignment_targets_tracking_results.append(row, ignore_index=True)

        # save the alignment_targets_tracking_results to the freq_grid_search_dir_name folder
        alignment_targets_tracking_results.to_csv(os.path.join(align_save_dir_freq_th, f'alignment_targets_tracking_results.csv'))

        ########################################################################################################
        # save the combined peak intensity to gcp bucket
        ########################################################################################################
        # gcp_file_path = f"mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/{freq_grid_search_dir_name}
        # save all results to gcp as well
        gcp_file_path = f"mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/{freq_grid_search_dir_name}"
        # upload all files from folders: keras_autoencoder_models0 and classical_models to all_resuls_file_folder_gcp_path
        bucket_name = 'mzlearn-webapp.appspot.com'
        upload_to_gcs(align_save_dir_freq_th, bucket_name, gcp_file_path)

    ####################################################################################################################
    # save the grid search summary to gcp bucket
    ####################################################################################################################
    print(gridsearch_params_combinations)
    # edit the gridsearch_params_combinations
    #reference_freq_th = row['reference_cohort_freq_threshold']
    # freq_th = row['remaining_cohorts_freq_threshold']
    # convert the values from reference_cohort_freq_threshold and remaining_cohorts_freq_threshold to percentage and with % sign
    gridsearch_params_combinations['reference_cohort_freq_threshold'] = gridsearch_params_combinations[
        'reference_cohort_freq_threshold'].apply(lambda x: f"{x * 100}%")
    gridsearch_params_combinations['remaining_cohorts_freq_threshold'] = gridsearch_params_combinations[
        'remaining_cohorts_freq_threshold'].apply(lambda x: f"{x * 100}%")
    gridsearch_params_combinations.to_csv(os.path.join(align_save_dir, 'grid_search_summary_df.csv'), index=False)
    gcp_file_path = f"mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/grid_search_summary_df.csv"
    with fs.open(gcp_file_path, 'w') as f:
        gridsearch_params_combinations.to_csv(f, index=True)

    ####################################################################################################################
    # update the database
    ####################################################################################################################
    job_status = 3
    query = (
        f"UPDATE app_pretrain_peak_combine_job_status SET job_status = %s WHERE id = %s")
    val = (job_status, peak_combine_job_id)
    cursor.execute(query, val)
    cnxn.commit()
