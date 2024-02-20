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
    intermediate_folder_path = f"{mzlearn_path}"
    bucket_name = 'mzlearn-webapp.appspot.com'
    bucket = client.get_bucket(bucket_name)
    dst_path = f"{script_path_mspeak}/{job_id}/result-"  # local path to the folder to be saved
    # try to donwload the folder from gcp to local
    # if gcp patch exists then download
    # mzlearn-webapp.appspot.com/mzlearn/min/2023-09-29 15:43:35/mzlearn_intermediate_results
    full_intermediate_folder_path = f"mzlearn-webapp.appspot.com/{mzlearn_path}"
    if fs.exists(full_intermediate_folder_path):
        print("try to download mzlearn_intermediate_results to local")
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
    num_cohorts_thresh = 0.5
    # num_cohorts_thresh_list = [0,0.25,0.5,0.75]
    cohort_correction_method = 'combat'
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
    # TODO: add a new loop there to loop through all the other studies frequency threshold
    # TODO: clearup existing code and get read to add new method
    ####################################################################################################################
    grid_search_summary_df = pd.DataFrame()
    for alignment_method in alignment_methods:
        for reference_freq_th in origin_freq_th:
            for freq_th in other_study_freq_th:

                # frequency sub dir
                freq_grid_search_dir_name = f'{alignment_method}_reference_freq_th_{reference_freq_th}_freq_th_{freq_th}'

                # get mspeak object for the reference study
                print("get original study data")
                origin_study = get_mspeak_from_job_id(script_path, str(reference_job_id), reference_freq_th)

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
                    new_study = get_mspeak_from_job_id(script_path, str(study_id), freq_th)
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
                alignment_df.to_csv(os.path.join(align_save_dir_freq_th, 'alignment_df.csv'), index=True)\

                # try:
                #     align_ms_studies(origin_study,
                #                      new_study,
                #                      origin_name=reference_job_id,
                #                      input_name=study_id,
                #                      save_dir=align_save_dir_freq_th)
                # except Exception as e:
                #     print(f'Alignment failed for {study_id} with error {e}')

                # alignment_df = combine_alignments_in_dir(align_save_dir_freq_th, origin_name=reference_job_id)
                # alignment_df.to_csv(os.path.join(align_save_dir_freq_th, 'alignment_df.csv'), index=True)

                # based on alignment_df build the grid_search_summary_df
                # this summary df has the following columns:
                # 1. freq_th
                # 2. peaks from origin study
                # 3. peaks from other study (each study has a column)
                # 4. common peaks found in all studies
                print(alignment_df.columns)
                dummy_row = {'method': alignment_method,
                             'reference_cohort_freq_threshold': f"{int(reference_freq_th * 100)}%",
                             'remaining_cohorts_freq_threshold': f"{int(freq_th * 100)}%",
                             'common_peaks': (~alignment_df.isna()).all(axis=1).sum(),
                             # common peaks is the count of rows from alignment_df that has no nan
                             'reference_cohort_peaks': alignment_df.shape[0]}
                for study_id in other_job_ids:
                    dummy_row[f"remaining_cohort_{study_id}_peaks"] = alignment_df[str(study_id)].count()

                dummy_row['common_peaks'] = (~alignment_df.isna()).all(axis=1).sum()
                grid_search_summary_df = grid_search_summary_df.append(dummy_row, ignore_index=True)

                ########################################################################################################
                # rename the features in each study to correspond to the origin, remove the features that
                # do not align to origin
                ########################################################################################################
                rename_inputs_to_origin(align_save_dir_freq_th,
                                        multi_alignment_df=None,
                                        input_name_list=None,
                                        load_dir=None,
                                        input_peak_obj_path_list=input_peak_obj_path_list)

                # for study_id in other_job_ids:
                #     result_path = os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}",
                #                                f"{study_id}.pkl")
                #     input_study_pkl_file = os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}",
                #                                         f"{study_id}.pkl")
                #     renamed_study_pkl_file = os.path.join(f"{script_path}/{study_id}/{freq_grid_search_dir_name}",
                #                                           f"{study_id}_renamed.pkl")
                #     if os.path.exists(input_study_pkl_file):
                #         input_study = load_mspeaks_from_pickle(input_study_pkl_file)
                #         input_alignment = alignment_df['Compound_ID_' + str(study_id)].copy()
                #         input_alignment.dropna(inplace=True)
                #
                #         current_ids = input_alignment.values
                #         new_ids = input_alignment.index
                #         input_study.rename_selected_peaks(current_ids, new_ids, verbose=False)
                #         input_study.save_to_pickle(renamed_study_pkl_file)

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
                combined_study_nan_mask = origin_study.missing_val_mask.loc[chosen_feats, :].copy()
                combined_study = np.log2(combined_study)

                # get all the file names for the origin_study
                origin_study_file_list = origin_study.peak_intensity.columns.to_list()
                study_id2file_name = {str(reference_job_id): origin_study_file_list}
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
                        subset_chosen = [i for i in chosen_feats if i in input_study.peak_intensity.index]
                        input_peaks = input_study.peak_intensity.loc[subset_chosen, :].copy()
                        input_study_nan_mask = input_study.missing_val_mask.loc[subset_chosen, :].copy()
                        input_peaks = np.log2(input_peaks)
                        # input_peaks = min_max_scale(input_peaks)
                        # combined_study = combined_study.join(input_peaks, lsuffix=reference_job_id, rsuffix=study_id, how='outer')
                        combined_study = combined_study.join(input_peaks, how='outer')
                        combined_study_nan_mask = combined_study_nan_mask.join(input_study_nan_mask,  how='outer')

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
                combined_study_nan_mask['missing_values'] = combined_study_nan_mask.sum(axis=1) / combined_study_nan_mask.shape[1]
                # reformt the combined_study_nan_mask['missing_values'] to save only first 2 decimal points and convert to percentage
                combined_study_nan_mask['missing_values'] = combined_study_nan_mask['missing_values'].apply(lambda x: round(x * 100, 2))
                # add % to combined_study_nan_mask['missing_values'] value
                combined_study_nan_mask['missing_values'] = combined_study_nan_mask['missing_values'].astype(str) + '%'

                # check if combined_study is empty
                if combined_study.empty:
                    print("combined_study is empty")
                    continue

                # TODO: correct for cohort effects using cohort_labels (study id)
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
                    data_corrected = standardize_across_cohorts(combined_study, cohort_labels, method=cohort_correction_method)
                    # data_corrected = combined_study
                    # calculate number of peaks and number of files from data_corrected
                    num_peaks = data_corrected.shape[0]
                    num_files = data_corrected.shape[1]
                except Exception as e:
                    print(f'Error in standardization: {e}')
                    continue

                # TODO: add save here to plot UMAP and PCA
                ########################################################################################################
                # %% Look at the PCA of the combined study
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_corrected.T)
                pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=data_corrected.columns)
                pca_df['mzlearn_cohort_id'] = metadata_df['mzlearn_cohort_id'].to_list()
                pca_df['file_name'] = metadata_df['file_name'].to_list()
                pca_df['MV percentage'] = combined_study_nan_mask['missing_values'].to_list()
                # pca_df['Study_num'] = metadata_df['Study_num']
                # pca_df['label'] = metadata_df[pretrain_label_col]
                pca_df.to_csv(os.path.join(align_save_dir_freq_th, f'pca_df_{cohort_correction_method}.csv'))
                hover_data = ['MV percentage']
                # plot the PCA and add file_name as hover and add title to show num_peaks and num_files
                graph = px.scatter(pca_df, x="PC1", y="PC2", color='mzlearn_cohort_id',
                                   color_discrete_sequence=px.colors.qualitative.Plotly,
                                   title=f'PCA with {num_peaks} peaks and {num_files} files',
                                   hover_name='file_name',
                                   hover_data=hover_data)

                graph.update_traces(marker={'size': 4.5})
                graph_div = plot({'data': graph}, output_type='div')
                pickle.dump(graph_div, open(f'{align_save_dir_freq_th}/pca_plot.pkl', 'wb'))

                # plot the PCA
                # sns.scatterplot(x='PC1', y='PC2', hue='study_id', data=pca_df)
                # plt.savefig(os.path.join(select_feats_dir, f'pca_plot_{cohort_correction_method}.png'),
                #             bbox_inches='tight')
                # plt.title(cohort_correction_method)
                # plt.close()

                ########################################################################################################
                # %% Look at the UMAP of the combined study
                reducer = umap.UMAP()
                umap_result = reducer.fit_transform(data_corrected.T)

                umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'], index=data_corrected.columns)
                umap_df['mzlearn_cohort_id'] = metadata_df['mzlearn_cohort_id'].to_list()
                umap_df['file_name'] = metadata_df['file_name'].to_list()
                umap_df['MV percentage'] = combined_study_nan_mask['missing_values'].to_list()
                # umap_df['Study_num'] = metadata_df['Study_num']
                # umap_df['label'] = metadata_df[pretrain_label_col]
                umap_df.to_csv(os.path.join(align_save_dir_freq_th, f'umap_df_{cohort_correction_method}.csv'))
                hover_data = ['MV percentage']
                graph = px.scatter(umap_df, x="UMAP1", y="UMAP2", color='mzlearn_cohort_id',
                                   color_discrete_sequence=px.colors.qualitative.Plotly,
                                   title=f'UMAP with {num_peaks} peaks and {num_files} files',
                                   hover_name='file_name',
                                   hover_data=hover_data)
                graph.update_traces(marker={'size': 4.5})
                graph_div = plot({'data': graph}, output_type='div')
                pickle.dump(graph_div, open(f'{align_save_dir_freq_th}/umap_plot.pkl', 'wb'))
                # plot the UMAP
                # sns.scatterplot(x='UMAP1', y='UMAP2', hue='study_id', data=umap_df)
                # plt.savefig(os.path.join(select_feats_dir, f'umap_plot_{cohort_correction_method}'),
                #             bbox_inches='tight')
                # plt.title(cohort_correction_method)
                # plt.close()

                ########################################################################################################
                # save the combined peak intensity to gcp bucket
                # gcp_file_path = f"mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/{freq_grid_search_dir_name}
                # save all results to gcp as well
                gcp_file_path = f"mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/{freq_grid_search_dir_name}"
                # upload all files from folders: keras_autoencoder_models0 and classical_models to all_resuls_file_folder_gcp_path
                bucket_name = 'mzlearn-webapp.appspot.com'
                upload_to_gcs(align_save_dir_freq_th, bucket_name, gcp_file_path)

                # mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{job_id}/alignment_df.csv
                # gcp_file_path = f"mzlearn-webapp.appspot.com/mzlearn_pretraining/peak_combine_results/{peak_combine_job_id}/{freq_grid_search_dir_name}/combined_study.csv"
                # with fs.open(gcp_file_path, 'w') as f:
                #     combined_study.to_csv(f, index=True)
                # print("combined peak intensity saved to gcp bucket")

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
