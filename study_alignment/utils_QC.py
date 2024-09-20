'''

Utility pots to perform quality control on the data


'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import requests
import zipfile
from google.cloud import storage
import gcsfs
# import mysql.connector
# from mysql.connector.constants import ClientFlag
from google.cloud import storage



def generate_pca_embedding(matrix, n_components=2):
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(matrix.T)
    if isinstance(matrix, pd.DataFrame):
        embedding = pd.DataFrame(embedding, index=matrix.columns, columns=[f'PCA{i+1}' for i in range(n_components)])
    return embedding




def generate_umap_embedding(matrix, n_components=2):

    reducer = umap.UMAP(n_components=n_components)
    embedding = reducer.fit_transform(matrix.T)
    if isinstance(matrix, pd.DataFrame):
        embedding = pd.DataFrame(embedding, index=matrix.columns, columns=[f'UMAP{i+1}' for i in range(n_components)])
    return embedding




def plot_pca(mzlearn_run_folder_name, embedding,metadata,col_name,yes_umap=False):
    plt.figure()  # Create a new figure
    if yes_umap:
        xvar = 'UMAP1'
        yvar = 'UMAP2'
    else:
        xvar = 'PCA1'
        yvar = 'PCA2'
    if metadata[col_name].nunique() < 10:
        palette = sns.color_palette("tab10", metadata[col_name].nunique())
        sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=metadata[col_name], palette=palette)
    else:
        sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=metadata[col_name])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=col_name)
    plt.xlabel(xvar)
    plt.ylabel(yvar)

    # add counts to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if metadata[col_name].nunique() < 15:
        labels = [f'{x} ({metadata[metadata[col_name]==x].shape[0]})' for x in labels]
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=col_name,
            ncol=2)  # ncol=2 makes the legend have 2 columns

    # add the number of samples to the title and the mzlearn run project folder name
    plt.title(f'mzlearn run: {mzlearn_run_folder_name} | N samples = {metadata[~metadata[col_name].isna()].shape[0]}')




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
        
        # delete the zip file
        os.remove(zip_path)
    else:
        print(f"Failed to download data from {dropbox_url}. Status code: {response.status_code}")
    return



def assign_color_map(unique_vals):
    unique_vals = [str(val) for val in unique_vals]  # Ensure all values are strings
    my_colors = get_color_map(len(unique_vals))
    color_map = dict(zip(np.sort(unique_vals), my_colors))
    return color_map




def get_color_map(n):
    my_32_colors = plt.cm.Accent.colors + plt.cm.Dark2.colors + plt.cm.Set2.colors + plt.cm.Pastel2.colors
    my_10_colors = plt.cm.tab10.colors
    my_20_colors = plt.cm.tab20.colors 
    my_42_colors = my_10_colors + my_32_colors
    my_52_colors = my_20_colors + my_32_colors

    if n <= 10:
        return my_10_colors
    elif n <= 20:
        return my_20_colors
    elif n <= 32:
        return my_32_colors
    elif n <= 42:
        return my_42_colors
    elif n <= 52:
        return my_52_colors
    else:
        # create a colormap from turbo
        return plt.cm.turbo(np.linspace(0, 1, n))



def generate_umap_embedding(Z_data, **kwargs):
    """Generate UMAP plot of data"""
    reducer = umap.UMAP(**kwargs)
    embedding = reducer.fit_transform(Z_data)
    embedding = pd.DataFrame(embedding, index=Z_data.index)
    return embedding




def create_selected_data(input_data_dir, sample_selection_col, 
                         selections_df = None,
                        metadata_df=None,subdir_col='Study ID',
                        output_dir=None, metadata_cols=[], 
                        save_nan=False, use_anndata=False):
    """
    Creates selected data based on the given parameters.

    Parameters:
    - input_data_dir (str): The directory path where the input data is located.
    - sample_selection_col (str): The column name in the metadata dataframe used for sample selection.
    - selections_df (pandas.DataFrame, optional): The dataframe containing the sample selections. If not provided, it will assume metadata_df contains selections dataframe.
    - metadata_df (pandas.DataFrame, optional): The metadata dataframe. If not provided, it will be read from 'metadata.csv' in the input_data_dir.
    - subdir_col (str, optional): The column name in the metadata dataframe used for subdirectory identification. Default is 'Study ID'.
    - output_dir (str, optional): The directory path where the output data will be saved. If not provided, it will be created as 'formatted_data' in the input_data_dir.
    - metadata_cols (list, optional): The list of column names to include in the output metadata. If not provided, all columns will be included.
    - save_nan (bool, optional): Whether to save NaN values in the output files. Default is False.
    - use_anndata (bool, optional): Whether to use AnnData format for saving the output data. Default is False.

    Returns:
    - output_dir (str): The directory path where the output data is saved.
    - save_file_id (str): The identifier used in the output file names.

    Raises:
    - ValueError: If the number of lost samples is too many compared to the metadata.

    """

    if output_dir is None:
        output_dir = os.path.join(input_data_dir, 'formatted_data')
        os.makedirs(output_dir, exist_ok=True)

    if metadata_df is None:
        if os.path.exists(f'{input_data_dir}/metadata.csv'):
            metadata_df = pd.read_csv(f'{input_data_dir}/metadata.csv', index_col=0)
        else:
            print(f'complete metadata file not found, generating using data in {input_data_dir}')
            create_full_metadata(input_data_dir)
            metadata_df = pd.read_csv(f'{input_data_dir}/metadata.csv', index_col=0)

    if selections_df is None:
        if 'Set' not in metadata_df.columns:
            selections_df = assign_sets(metadata_df,return_only_sets=True)
        else:
            selections_df = get_selection_df(metadata_df)

    if subdir_col not in selections_df.columns:
        selections_df[subdir_col] = metadata_df[subdir_col]

    subdir_list = selections_df[subdir_col].unique()

    select_ids = selections_df[selections_df[sample_selection_col]].index.to_list()
    print(f'Number of samples selected: {len(select_ids)}')

    if len(metadata_cols) == 0:
        metadata_cols = metadata_df.columns.to_list()

    save_file_id = sample_selection_col.replace(' ', '_')
    y_file = f'{output_dir}/y_{save_file_id}.csv'
    if save_nan:
        # first create the X_file, before creating the nan_file
        _, _ = create_selected_data(input_data_dir, sample_selection_col, 
                                    selections_df=selections_df,
                                    metadata_df=metadata_df,subdir_col=subdir_col,
                                    output_dir=output_dir, metadata_cols=metadata_cols, 
                                    save_nan=False, use_anndata=False)
        print('Saving the mask of NaN values')

    if save_nan:
        X_file = f'{output_dir}/nan_{save_file_id}.csv'
    else:
        X_file = f'{output_dir}/X_{save_file_id}.csv'
    h5ad_file = f'{output_dir}/{save_file_id}.h5ad'

    if use_anndata and os.path.exists(h5ad_file):
        print(f'Files already exist at {output_dir}')
        return output_dir, save_file_id
    if not use_anndata and os.path.exists(y_file) and os.path.exists(X_file):
        print(f'Files already exist at {output_dir}')
        return output_dir, save_file_id


    X_list = []
    obs_list = []
    y_list = []

    for subdir in subdir_list:
        if save_nan:
            intensity_file = f'{input_data_dir}/{subdir}/nan_matrix.csv'
        else:
            intensity_file = f'{input_data_dir}/{subdir}/scaled_intensity_matrix.csv'
        sub_metadata_file = f'{input_data_dir}/{subdir}/metadata.csv'

        if os.path.exists(intensity_file):
            subset_select_ids = selections_df[selections_df[subdir_col] == subdir].index.to_list()
            subset_select_ids = list(set(select_ids).intersection(subset_select_ids))
            if len(subset_select_ids) > 0:
                intensity_df = pd.read_csv(intensity_file, index_col=0)
                intensity_df = intensity_df.loc[subset_select_ids].copy()
                X_list.append(intensity_df)
                obs_list.extend(subset_select_ids)
                
                # if os.path.exists(sub_metadata_file):
                #     sub_metadata_df = pd.read_csv(sub_metadata_file, index_col=0)
                #     sub_metadata_df[subdir_col] = subdir
                #     y_list.append(sub_metadata_df.loc[subset_select_ids, metadata_cols].copy())
        else:
            print(f'{subdir} is missing')
            continue

    X = pd.concat(X_list, axis=0)
    obs = metadata_df.loc[obs_list, metadata_cols]

    if len(obs) != X.shape[0]:
        print('Warning, the number of samples in the metadata and intensity matrix do not match')
        print(f'Number of samples in metadata: {len(obs)}')
        print(f'Number of samples in intensity matrix: {X.shape[0]}')
        common_samples = list(set(X.index).intersection(obs.index))
        print(f'Number of common samples: {len(common_samples)}')
        if len(common_samples) < 0.9 * len(obs):
            raise ValueError('The number of lost samples is too many, check the data')
        X = X.loc[common_samples, :].copy()
        obs = obs.loc[common_samples, :].copy()

    if use_anndata:
        adata = ad.AnnData(X=X, obs=obs)
        adata.write_h5ad(h5ad_file)
        print(f'Anndata file saved at {h5ad_file}')
    else:
        obs.to_csv(f'{output_dir}/y_{save_file_id}.csv')
        X.to_csv(f'{output_dir}/X_{save_file_id}.csv')
        print('CSV files saved')

    return output_dir, save_file_id