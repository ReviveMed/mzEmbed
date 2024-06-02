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

########################################################################################################################
# read the file_info from selected projects
# match the file name to gathered meta-data
########################################################################################################################
gather_meta_data = False
if gather_meta_data:
    # priority projects
    projects = ['ST000909', 'ST001918', 'ST001428', 'ST002331', 'ST001931', 'ST001932', 'ST001849', 'ST001422', 'ST002027',
                'ST000422', 'ST002244', 'ST002251', 'ST001423', 'ST002112', 'ST001408', 'ST000388', 'ST001519']

    mzlearn_run_ids = [605, 606, 581, 522, 539, 507, 526, 555, 557, 550, 558, 502, 504, 579, 584, 585, 586, 587, 588, 505, 509,
                       503, 559, 556]
    # print(f"unique mzlearn_run_id: {len(set(mzlearn_run_ids))}")

    # get the file_info from the selected projects
    # have to do it one by one from 0 to 23
    mzlearn_run_ids = [mzlearn_run_ids[22]]
    for mzlearn_run in mzlearn_run_ids:
        print(f"mzlearn_run: {mzlearn_run}")
        # based on the mzlearn_run_id get job detail from the database
        cursor.execute(f"SELECT * FROM app_user_job_status WHERE id = {mzlearn_run}")
        job_detail = cursor.fetchall()
        job_detail = job_detail[0]
        # print(job_detail)
        user_id = job_detail[1]
        creation_time = job_detail[3]
        public_dataset_code = job_detail[5]
        if mzlearn_run == 581:
            public_dataset_code = 'ST000388'
        # print(user_id, creation_time)
        # convert the creation time to string
        creation_time = creation_time.strftime("%Y-%m-%d %H:%M:%S")
        mzlearn_path = f"mzlearn/{user_id}/{creation_time}"
        print(f"mzlearn_path: {mzlearn_path}; public_dataset_code: {public_dataset_code}")

        # read the sample_info.csv that every mzlearn run has
        # mzlearn-webapp.appspot.com/mzlearn/min/2024-02-22 23:52:04/sample_info/sample_info.csv
        sample_info_path = f"mzlearn-webapp.appspot.com/{mzlearn_path}/sample_info/sample_info.csv"
        with fs.open(sample_info_path) as f:
            sample_info = pd.read_csv(f)
        print(sample_info)

        # read the gathered meta-data based on the public_dataset_code
        # mzlearn-webapp.appspot.com/Data-engine/ST001519/meta_and_target/auto_built_meta/all_metadata.csv
        meta_data_path = f"mzlearn-webapp.appspot.com/Data-engine/{public_dataset_code}/meta_and_target/auto_built_meta/all_metadata.csv"
        with fs.open(meta_data_path) as f:
            meta_data = pd.read_csv(f)
        # sort meta_data by Subject ID
        # meta_data = meta_data.sort_values(by=['Raw_files'])
        print(meta_data)

        # print(dfsadfsad)

        # loop through the sample_info df and try to find the matching rows from meta_data
        # by checking if the file_name from sample_info is in the meta_data Raw_files column
        for index, row in sample_info.iterrows():
            file_name = row['file_name']
            # if there is "/" in the file_name, then only take the last part
            if "/" in file_name:
                file_name = file_name.split("/")[-1]
            # if there is "." in the file_name, then only take the first part
            if "." in file_name:
                file_name = file_name.split(".")[0]

            # file name only take the first 3 part separated by "_"
            # file_name = "_".join(file_name.split("_")[:3])

            # extract the last part of the file_name separated by "_"
            file_name_first_half = "_".join(file_name.split("_")[:-1])
            file_name_last_half = file_name.split("_")[-1]

            # the file_name_last_half is in the format 001, 002, 003, to 269 tec
            # add 1 to the file_name_last_half file_name, such that 001 become 002, 260 become 261
            file_name_last_half = f"{int(file_name_last_half) + 1:03d}"
            file_name = f"{file_name_first_half}_{file_name_last_half}"

            # # extract subject_id from the file_name
            # subject_id = f'{file_name.split("_")[2]}'
            # # print(subject_id)
            # subject_id = f'{subject_id.split("-")[0]}-{subject_id.split("-")[1]}'
            # print(f"file_name: {file_name}; subject_id: {subject_id}")

            # find the matching row from meta_data by matching the subject_id to meta-data Subject ID column
            # there may be multiple matching rows, select the first one
            column_to_use_for_matching = 'Raw_files'
            # matching_row = meta_data[meta_data[column_to_use_for_matching] == file_name]
            matching_row = meta_data[meta_data[column_to_use_for_matching].str.contains(file_name)]
            # check if the matching_row is empty or not
            age = 'NA'
            age_at_symptom_onset = 'NA'
            gender = 'NA'
            treatment = 'NA'
            if matching_row.empty:
                print(f"file_name: {file_name} not found in meta_data")
            else:
                print(f"file_name: {file_name} found in {matching_row['Raw_files']}")
                # select the first matching row
                matching_row = matching_row.iloc[0]
                print(matching_row)
                try:
                    age = matching_row['Age']
                except Exception as e:
                    print(f"error: {e}")

                try:
                    age_at_symptom_onset = matching_row['Age at Symptom onset (years)']
                except Exception as e:
                    print(f"error: {e}")

                try:
                    gender = matching_row['Factors.Sex']
                except Exception as e:
                    print(f"error: {e}")

            # add the meta-data to the sample_info df
            sample_info.loc[index, 'age'] = age
            sample_info.loc[index, 'age_at_symptom_onset'] = age_at_symptom_onset
            sample_info.loc[index, 'gender'] = gender

        # save the sample_info df with the added meta-data to the GCP bucket
        # save the sample_info df to the GCP bucket
        sample_info_path = f"mzlearn-webapp.appspot.com/{mzlearn_path}/sample_info/sample_info_with_meta_data.csv"
        with fs.open(sample_info_path, 'w') as f:
            sample_info.to_csv(f, index=False)
        print(f"sample_info with meta-data saved to: {sample_info_path}")


########################################################################################################################
# combine gathered into one file
########################################################################################################################
# the files are saved in {script_path}/mzlearn_pretraining_metadata
# read each file as panda df and seelct only file_name, age, gender, and combine into one
combine_into_one = True
if combine_into_one:
    gathered_files_path = f"{os.path.dirname(os.path.realpath(__file__))}/mzlearn_pretraining_metadata"
    # print(gathered_files_path)
    combined_aga_gender_df = pd.DataFrame()
    for file in os.listdir(gathered_files_path):
        if file.endswith(".csv"):
            file_path = f"{gathered_files_path}/{file}"
            # print(file_path)
            with open(file_path) as f:
                df = pd.read_csv(f)
            # print(df)
            # make the column names of the df all into lower case
            df.columns = map(str.lower, df.columns)
            # if age not in the columns, then add it as NA
            if 'age' not in df.columns:
                df['age'] = 'NA'
            if 'gender' not in df.columns:
                df['gender'] = 'NA'

            # select only file_name, age, gender
            df = df[['file_name', 'age', 'gender']]

            # combine the df
            combined_aga_gender_df = pd.concat([combined_aga_gender_df, df], ignore_index=True)

    # save the combined df to the local as csv file
    # fill Nan from combined_aga_gender_df as 'NA'
    combined_aga_gender_df = combined_aga_gender_df.fillna('NA')
    # change "-" to "NA" in age column and gender column
    combined_aga_gender_df = combined_aga_gender_df.replace('-', 'NA')
    # if there is "/" in collumn file_name, split it and take the last part
    combined_aga_gender_df['file_name'] = combined_aga_gender_df['file_name'].apply(lambda x: x.split("/")[-1])
    # save the combined df to the GCP bucket
    combined_aga_gender_df.to_csv('combined_aga_gender.csv', index=False)

