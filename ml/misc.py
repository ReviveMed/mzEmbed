import os
import numpy as np
import pandas as pd
import torch
import requests
import zipfile
import json

###################
## Basic File I/O Functions
###################
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def clean_for_json(data):
    if isinstance(data, list):
        data = [clean_for_json(d) for d in data]

    if isinstance(data, dict):
        data = {k: clean_for_json(v) for k, v in data.items() if k[0] != '_'}

    if isinstance(data, np.int64):
        data = int(data)

    if callable(data):
        # convert function to string
        data = str(data)
    
    # check if data is a tensor
    if isinstance(data, torch.Tensor):
        data = data.tolist()

    return data

def save_json(data, file_path):
    data = clean_for_json(data)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


####################################################################################
# Download data
def download_data_file(dropbox_url, save_dir='data'):
    # Parse the file name from the URL
    file_name = dropbox_url.split("/")[-1]
    if '?' in file_name:
        file_name = file_name.split('?')[0]

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Send a GET request to the Dropbox URL
    response = requests.get(dropbox_url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the contents of the response to a file
        with open(os.path.join(save_dir, file_name), 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        print(f"Failed to download data from {dropbox_url}. Status code: {response.status_code}")

    return


def download_data_dir(dropbox_url, save_dir='data'):
    # Parse the file name from the URL
    file_name = dropbox_url.split("/")[-1]

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Send a GET request to the Dropbox URL
    response = requests.get(dropbox_url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        zip_path = os.path.join(save_dir, file_name)
        # Write the contents of the response to a file
        with open(zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
    else:
        print(f"Failed to download data from {dropbox_url}. Status code: {response.status_code}")

    # delete the zip file
    os.remove(zip_path)
    return

def get_dropbox_dir():
    my_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton'
    if not os.path.exists(my_dir):
        my_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton'
    if not os.path.exists(my_dir):
        raise ValueError('Dropbox directory not found')

    return my_dir

####################################################################################

def encode_df_col(df, col, val_mapper=None, suffix='_encoded'):
    # Encode text features as integers
    if val_mapper is None:
        vals = df[col].unique()
        # remove the nan values
        vals = vals[~pd.isnull(vals)]
        sorted_vals = np.sort(vals)
        val_mapper = {val: i for i, val in enumerate(sorted_vals)}
    df[col + suffix] = df[col].map(val_mapper)
    return df, val_mapper


####################################################################################
def normalize_loss(current_loss, loss_avg=1, beta=0, current_epoch=1):
    # normalize the loss by its average, useful when creating multi-output models
    if current_epoch < 0:
        new_loss = current_loss / loss_avg
        return new_loss, loss_avg
    if beta  == 0:
        loss_avg = (loss_avg*current_epoch + current_loss.item()) / (current_epoch + 1)
    elif beta < 0:
        loss_avg = 1
    else:
        loss_avg = beta * loss_avg + (1 - beta) * current_loss.item()

    new_loss = current_loss / loss_avg
    return new_loss, loss_avg


# round to significant digits
def round_to_sig(x, sig_figs=2):
    return round(x, sig_figs-int(np.floor(np.log10(abs(x))))-1)

def round_to_even(n):
    if n % 2 == 0:
        return n
    else:
        return n + 1

def get_clean_batch_sz(len_dataset, org_batch_sz):
    # due to batch normalization, we want the batches to be as clean as possible
    curr_remainder = len_dataset % org_batch_sz
    max_iter = 100
    if org_batch_sz >= len_dataset:
        return org_batch_sz
    if (curr_remainder == 0) or (curr_remainder > org_batch_sz/2):
        return org_batch_sz
    else:
        batch_sz = org_batch_sz
        iter = 0
        while (curr_remainder != 0) and (curr_remainder < batch_sz/2) and (iter < max_iter):
            iter += 1
            if batch_sz < org_batch_sz/2:
                batch_sz = 2*org_batch_sz
            batch_sz -= 1
            curr_remainder = len_dataset % batch_sz
        if iter >= max_iter:
            print('Warning: Could not find a clean batch size')
        # print('old batch size:', org_batch_sz, 'new batch size:', batch_sz, 'remainder:', curr_remainder)
        return batch_sz