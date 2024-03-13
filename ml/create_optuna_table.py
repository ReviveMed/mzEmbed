# Create table summary and upload to GCP
import optuna
import os
from utils_gcp import upload_file_to_bucket, download_file_from_bucket, check_file_exists_in_bucket, upload_path_to_bucket



BASE_DIR = '/DATA'
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'
USE_WEBAPP_DB = True
storage_name = 'optuna'
DATA_DIR = f'{BASE_DIR}/data'
gcp_save_loc = 'March_12_Data'

if __name__ == '__main__':


    study_name = input("Enter the study name: ")

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



    study = optuna.create_study(direction="maximize",
                                study_name=study_name, 
                                storage=storage_name, 
                                load_if_exists=True)
    

    study_table_path = f'{data_dir}/{study_name}_table.csv'
    
    # Create a table of the study in csv format
    study_table = study.trials_dataframe()
    study_table.to_csv('study_table.csv', index=False)

    study_table.to_csv(study_table_path, index=False)

    upload_file_to_bucket(study_table_path, gcp_save_loc)