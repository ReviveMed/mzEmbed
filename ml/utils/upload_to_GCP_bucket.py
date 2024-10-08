from google.cloud import storage
import os

def upload_to_gcs(local_directory, bucket_name, gcs_path):
    """
        Uploads all files from a local directory to a specified Google Cloud Storage (GCS) bucket.

        Args:
            local_directory (str): The local directory path from where files will be uploaded.
            bucket_name (str): The name of the GCS bucket where files will be uploaded.
            gcs_path (str): The path in the GCS bucket where the files will be stored. 
                            This acts as a prefix to organize the uploaded files in GCS.

        Returns:
            None
    """
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
# example usage
# local directory to upload
local_directory = '/home/example_folder_name'
# gcs directory to upload to
gcp_file_path = f"mzlearn_pretraining/example_folder_name"
# upload all files from folders: keras_autoencoder_models0 and classical_models to all_resuls_file_folder_gcp_path
bucket_name = 'mzlearn-webapp.appspot.com'
upload_to_gcs(local_directory, bucket_name, gcp_file_path)