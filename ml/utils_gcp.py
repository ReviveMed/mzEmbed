# GCP related functions

import os
from google.cloud import storage
import gcsfs

fs = gcsfs.GCSFileSystem()
# client = storage.Client()
client = storage.Client(project='mzlearn-webapp')

gcp_root_path = f'Data-engine/JonahData'
# gcp_root_path = f'Data-engine'
client_id = 'mzlearn-webapp.appspot.com'
gcp_root_path_with_client = f'gs://{client_id}/{gcp_root_path}'


##########################################
####### What is on the GCP bucket? #######

def check_file_exists_in_bucket(project_subdir, file_path):
    """
    Check if a file exists in the specified Google Cloud Storage bucket.

    Args:
        project_subdir (str): The subdirectory within the bucket where the file is located.
        file_path (str): The path to the file within the bucket.

    Returns:
        bool: True if the file exists in the bucket, False otherwise.
    """
    gcp_project_dir = os.path.join(gcp_root_path_with_client, project_subdir, file_path)
    return fs.exists(gcp_project_dir)



def get_list_of_files_in_bucket(project_subdir):
    """
    Retrieves a list of files in a Google Cloud Storage bucket.

    Args:
        project_subdir (str): The subdirectory within the bucket to search for files.

    Returns:
        list: A list of file names found in the specified subdirectory.

    """
    gcp_project_dir = os.path.join(gcp_root_path, project_subdir)
    # print(gcp_project_dir)

    blobs = client.list_blobs(client_id, prefix=gcp_project_dir)
    file_list = []
    for blob in blobs:
        file_list.append(blob.name.replace(gcp_project_dir, ''))
    return file_list



#################################################
####### Downloading from the GCP bucket #######

def download_file_from_bucket(project_subdir, file_path, local_path=None, verbose=True):
    """
    Downloads a file from a Google Cloud Storage bucket.

    Args:
        project_subdir (str): The subdirectory within the bucket where the file is located.
        file_path (str): The path of the file within the subdirectory.
        local_path (str, optional): The local directory where the file should be saved. If not provided, the file will be saved in the current directory. Defaults to None.
        verbose (bool, optional): Whether to print verbose output during the download process. Defaults to True.

    Returns:
        float: The total size of the downloaded file in megabytes.
    """
    tot_download_size = 0
    blobs = client.list_blobs(client_id, prefix=f'{gcp_root_path}/{project_subdir}/{file_path}')
    
    if local_path is None:
        # save to current directory
        save_path = file_path
    else:
        save_path = os.path.join(local_path, file_path)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for blob in blobs:
        if verbose: 
            print(blob.name)
        blob.download_to_filename(save_path)
        file_sz = os.path.getsize(save_path)
        tot_download_size += file_sz

    tot_download_size = tot_download_size / 1e6
    if verbose: 
        print(f'Downloaded {tot_download_size:.2f} MB')
    
    return tot_download_size


def download_dir_from_bucket(project_subdir, dir_path, local_path, verbose=True):
    """
    Downloads a directory from a Google Cloud Storage bucket to a local directory.

    Args:
        project_subdir (str): The subdirectory within the Google Cloud Storage bucket where the directory is located.
        dir_path (str): The path of the directory within the project_subdir to download.
        local_path (str): The local directory path where the downloaded files will be saved.
        verbose (bool, optional): Whether to print verbose output during the download process. Defaults to True.

    Returns:
        float: The total size of the downloaded files in megabytes.

    Raises:
        <ExceptionType>: <Description of the exception raised, if applicable>

    """
    gcp_prefix = os.path.join(gcp_root_path, project_subdir, dir_path)
    tot_download_size = 0

    blobs = client.list_blobs(client_id, prefix=gcp_prefix)

    files_to_download = [blob.name.replace(gcp_prefix, '') for blob in blobs]
    if verbose: 
        print(f'Found {len(files_to_download)} files to download')

    save_dir = os.path.join(local_path, dir_path)
    os.makedirs(save_dir, exist_ok=True)

    for blob in blobs:
        blob_file_path = blob.name.replace(gcp_prefix, '')
        save_path = os.path.join(save_dir, blob_file_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        blob.download_to_filename(save_path)
        file_sz = os.path.getsize(save_path)
        tot_download_size += file_sz

    tot_download_size = tot_download_size / 1e6
    if verbose: 
        print(f'Downloaded {tot_download_size:.2f} MB to {local_path}')
    
    return tot_download_size


#################################################
####### Uploading to the GCP bucket #######


def upload_path_to_bucket(local_path, project_subdir, verbose=True, save_subdir=None):
    """
    Uploads files from a local path to a Google Cloud Storage bucket.

    Args:
        local_path (str): The local path of the files to be uploaded.
        project_subdir (str): The subdirectory within the Google Cloud Storage bucket where the files will be uploaded.
        verbose (bool, optional): If True, prints progress messages during the upload process. Defaults to True.
        save_subdir (str, optional): The subdirectory within the project_subdir where the files will be saved. Defaults to None.

    Returns:
        float: The total size of the uploaded files in megabytes.

    Raises:
        None

    """

    if not os.path.exists(local_path):
        if verbose: 
            print(f'Local path {local_path} does not exist')
        return 0

    bucket = client.get_bucket(client_id)
    gcp_project_dir = os.path.join(gcp_root_path, project_subdir, save_subdir or '')
    tot_upload_size = 0

    if verbose:
        print(f'Uploading {local_path} to {gcp_project_dir}')

    paths = [local_path] if os.path.isfile(local_path) else os.walk(local_path)

    for root, dirs, files in paths:
        for file in files:
            local_file_path = os.path.join(root, file)
            file_sz = os.path.getsize(local_file_path)
            gcp_file_path = os.path.join(gcp_project_dir, os.path.relpath(local_file_path, local_path))
            blob = bucket.blob(gcp_file_path)
            blob.upload_from_filename(local_file_path)
            tot_upload_size += file_sz
            if verbose: 
                print(f'Uploaded {file_sz / 1e6:.2f} MB to {gcp_file_path}')

    if verbose: 
        print(f'Uploaded {tot_upload_size / 1e6:.2f} MB to {gcp_project_dir}')
    
    return tot_upload_size / 1e6


def upload_file_to_bucket(local_path, project_subdir, save_subdir=None, verbose=True):
    """
    Uploads a file to a Google Cloud Storage bucket.

    Args:
        local_path (str): The local path of the file to be uploaded.
        project_subdir (str): The subdirectory within the project directory where the file will be saved.
        save_subdir (str, optional): The subdirectory within the project subdirectory where the file will be saved. Defaults to None.
        verbose (bool, optional): If True, prints information about the upload process. Defaults to True.

    Returns:
        float: The size of the uploaded file in megabytes.

    Raises:
        None

    """

    if not os.path.exists(local_path):
        if verbose: 
            print(f'Local path {local_path} does not exist')
        return 0

    bucket = client.get_bucket(client_id)
    gcp_project_dir = os.path.join(gcp_root_path, project_subdir, save_subdir or '')
    file_sz = os.path.getsize(local_path)
    gcp_file_path = os.path.join(gcp_project_dir, os.path.basename(local_path))
    blob = bucket.blob(gcp_file_path)
    blob.upload_from_filename(local_path)

    if verbose: 
        print(f'Uploaded {file_sz / 1e6:.2f} MB to {gcp_file_path}')
    
    return file_sz / 1e6



### main function
if __name__ == '__main__':
    upload_file_to_bucket('main.py', 'test', verbose=True)