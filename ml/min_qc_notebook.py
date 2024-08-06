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


def generate_pca_embedding(matrix, n_components=2):
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(matrix.T)
    if isinstance(matrix, pd.DataFrame):
        embedding = pd.DataFrame(embedding, index=matrix.columns, columns=[f'PCA{i + 1}' for i in range(n_components)])
    return embedding


def generate_umap_embedding(matrix, n_components=2):
    reducer = umap.UMAP(n_components=n_components)
    embedding = reducer.fit_transform(matrix.T)
    if isinstance(matrix, pd.DataFrame):
        embedding = pd.DataFrame(embedding, index=matrix.columns, columns=[f'UMAP{i + 1}' for i in range(n_components)])
    return embedding


def plot_pca(mzlearn_run_folder_name, embedding, metadata, col_name, yes_umap=False):
    plt.figure()  # Create a new figure
    if yes_umap:
        xvar = 'UMAP1'
        yvar = 'UMAP2'
    else:
        xvar = 'PCA1'
        yvar = 'PCA2'
    if metadata[col_name].nunique() < 10:
        palette = sns.color_palette("tab10", metadata[col_name].nunique())
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=metadata[col_name], palette=palette)
    else:
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=metadata[col_name])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=col_name)
    plt.xlabel(xvar)
    plt.ylabel(yvar)

    # add counts to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if metadata[col_name].nunique() < 15:
        labels = [f'{x} ({metadata[metadata[col_name] == x].shape[0]})' for x in labels]
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=col_name,
                   ncol=2)  # ncol=2 makes the legend have 2 columns

    # add the number of samples to the title and the mzlearn run project folder name
    plt.title(f'mzlearn run: {mzlearn_run_folder_name} | N samples = {metadata[~metadata[col_name].isna()].shape[0]}')


    # plt.title(f'N samples = {metadata[~metadata[col_name].isna()].shape[0]}')


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

# create input and output data dir
homedir = "/app"
input_data_dir = f'{homedir}/INPUT_DATA'
os.makedirs(input_data_dir, exist_ok=True)
output_dir = f'{homedir}/PROCESSED_DATA'
os.makedirs(output_dir, exist_ok=True)

# download datasets and metadata from dropbox to input_data_dir
# https://www.dropbox.com/scl/fo/9xtdsndm39mwxw3p3hrjs/AKjY7SDKdjmu1Ne1oYviLpk?rlkey=xc2xwzj5ky3jzm4q7ec8t6rd9&st=d7berzpw&dl=0
drop_box_data_url = "https://www.dropbox.com/scl/fo/9xtdsndm39mwxw3p3hrjs/AKjY7SDKdjmu1Ne1oYviLpk?rlkey=qq4s1e3s20rdbd1e78rsk04n6&dl=1"
download_data_dir(drop_box_data_url, save_dir=input_data_dir)

########################################################################################################################
# plot PCA for all mzlearn runs
mzlearn_run_input_folder_names2mzlearn_run_id = {'ST000388': 581,
                                                 'ST000422': 550,
                                                 'ST000601': 547,
                                                 'ST000909': 556,
                                                 'ST001236': 589,
                                                 'ST001237': 590,
                                                 'ST001408': 522,
                                                 'ST001422': 502,
                                                 'ST001423': 526,
                                                 'ST001428': 503,
                                                 'ST001519': 605,
                                                 'ST001849': 504,
                                                 'ST001918': 559,
                                                 'ST001931': 505,
                                                 'ST001932': [579,584,585,586,587,588]
                                                 'ST002027': 558,
                                                 'ST002112': 507,
                                                 'ST002244': 557,
                                                 'ST002251': 555,
                                                 'ST002331': 509,
                                                 'stanford-hmp2': 631
                                                }
# for keys in the mzlearn_run_input_folder_names2mzlearn_run_id dictionary
for mzlearn_run_folder_name, mzlearn_run_id in mzlearn_run_input_folder_names2mzlearn_run_id.items():
    base_dir = f"{input_data_dir}/{mzlearn_run_folder_name}"
    intensity_df_path = f"{base_dir}/scaled_intensity_matrix.csv"
    metadata_df_path = f"{base_dir}/metadata.csv"

    # Check if files exist
    if not os.path.exists(intensity_df_path):
        print(f"File not found: {intensity_df_path}")
    if not os.path.exists(metadata_df_path):
        print(f"File not found: {metadata_df_path}")

    # read them as panda df
    intensity_df = pd.read_csv(intensity_df_path, index_col=0)
    metadata_df = pd.read_csv(metadata_df_path, index_col=0)

    # if there is no run_order column in metadata_df
    # create one that is based on the alphabetical order of the index
    if 'run_order' not in metadata_df.columns:
        metadata_df['run_order'] = pd.Categorical(metadata_df.index).codes

    # plot PCA for all mzlearn runs each has its own plot
    embedding0 = generate_pca_embedding(intensity_df.T).values
    plot_pca(embedding0, metadata_df, col_name='run_order', yes_umap=False)

########################################################################################################################
# calcualte peak robustness
# calcualte robustness for all peaks from all all mzlearn runs

# read in all study metadata and selection df
metadata_df = pd.read_csv(f'{input_data_dir}/metadata.csv', index_col=0)
selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv', index_col=0)
metadata_df['Set'] = selections_df['Set']
subdir_col = 'Study ID'

# calcualte peak robustness for all peaks
group_sz_dict = {}
freq_of_peaks_by_study = {}
for mzlearn_run_folder_name, mzlearn_run_id in mzlearn_run_input_folder_names2mzlearn_run_id.items():
    nan_file = f'{input_data_dir}/{mzlearn_run_folder_name}/nan_matrix.csv'
    samples_in_cohort = metadata_df[metadata_df[subdir_col] == mzlearn_run_folder_name].index
    nan_matrix = pd.read_csv(nan_file, index_col=0)
    nan_matrix = nan_matrix.loc[samples_in_cohort, :].copy()
    num_samples = nan_matrix.shape[0]

    freq_of_peaks = 1 - nan_matrix.mean(axis=0)
    freq_of_samples = 1 - nan_matrix.mean(axis=1)

    group_sz_dict[mzlearn_run_folder_name] = num_samples
    freq_of_peaks_by_study[mzlearn_run_folder_name] = freq_of_peaks
group_sz_df = pd.DataFrame(group_sz_dict, index=['N']).T
group_sz_df = group_sz_df.astype(int)
group_freq_df = pd.DataFrame(freq_of_peaks_by_study)
group_log_sz_df = np.log(1 + group_sz_df)
peak_robustness = group_freq_df.mul(group_log_sz_df['N'], axis=1).sum(axis=1) / group_log_sz_df['N'].sum()
# print(peak_robustness)

# create color map for studies and cohort
study_id_to_label = metadata_df.groupby('Study ID')['Cohort Label v0'].first().to_dict()
cohort_label_to_color = assign_color_map(metadata_df['Cohort Label v0'].unique())
study_id_to_color = {k: cohort_label_to_color[v] for k, v in study_id_to_label.items()}
study_id_to_uniq_color = assign_color_map(metadata_df['Study ID'].unique())
group_peak_counts = {}
for mzlearn_run_folder_name, mzlearn_run_id in mzlearn_run_input_folder_names2mzlearn_run_id.items():
    study_id = mzlearn_run_folder_name
    group_peak_counts[study_id] = np.sum(group_freq_df[study_id] > 0)

feat_info_3 = group_freq_df

col_cluster = False
row_cluster = False
subdir_list = mzlearn_run_input_folder_names2mzlearn_run_id.keys()
plot_df = feat_info_3[subdir_list].T
col_colors = pd.DataFrame(index=plot_df.columns)
for col in feat_info_3.columns:
    if col in subdir_list:
        continue
    col_colors[col] = feat_info_3[col].map({True: 'black', False: 'white'})
col_colors[f'Peak Robustness'] = peak_robustness.map(lambda x: plt.cm.plasma(x))
if col_cluster == False:
    col_val_order = peak_robustness
    col_val_order.sort_values(ascending=False, inplace=True)
    plot_df = plot_df[col_val_order.index]
if row_cluster == False:
    row_val_order = group_sz_df['N'].copy()
    row_val_order.sort_values(ascending=False, inplace=True)
    plot_df = plot_df.loc[row_val_order.index]

# create row colors
row_colors = pd.DataFrame(index=plot_df.index)
row_colors['Cohort Label'] = [study_id_to_color[c] for c in plot_df.index]
row_colors['Cohort Size'] = [group_sz_df.loc[c, 'N'] for c in plot_df.index]
row_colors['Cohort Size'] = row_colors['Cohort Size'] / row_colors['Cohort Size'].max()
row_colors['Cohort Size'] = row_colors['Cohort Size'].map(lambda x: plt.cm.copper_r(x))
new_row_names = [f'{c} (Nfiles={group_sz_df.loc[c].values[0]}, Npeaks={group_peak_counts[c]:.0f})'
                 for c in plot_df.index]

plot_df.index = new_row_names
row_colors.index = new_row_names
print(plot_df)

g = sns.clustermap(plot_df, cmap='viridis', figsize=(20, 10),
                   row_colors=row_colors, col_colors=col_colors,
                   col_cluster=col_cluster, row_cluster=row_cluster,
                   dendrogram_ratio=0.04)

plt.setp(g.ax_heatmap.get_xticklabels(), visible=False)

# for each row of the plot_df, plot a histogram of the peak robustness
for i, row in enumerate(plot_df.index):
    ax = g.ax_row_dendrogram[i]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)
    ax.set_title(row, fontsize=8)
    ax.hist(feat_info_3.loc[row], bins=20, color='black', orientation='horizontal')
    ax.set_xlim(0, 1)

########################################################################################################################
# umap
# plot the UMAP for all inputs

# read in all study metadata and selection df
metadata_df = pd.read_csv(f'{input_data_dir}/metadata.csv', index_col=0)
selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv',index_col=0)
metadata_df['Set'] = selections_df['Set']
subdir_col = 'Study ID'

hue_col = 'Cohort Label v0'
eval_name = 'Discovery Train'
setup_id = 'Raw Data'

pretrain_metadata = metadata[metadata['Set'].str.contains('Pretrain')]
cohort_order = pretrain_metadata['Cohort Label v0'].value_counts().sort_index().index.tolist()

# based on eval_name build X_data and y_data from

# X_data = pd.read_csv(f'{output_dir}/X_Pretrain_Discovery_Train.csv', index_col=0)
# y_data = pd.read_csv(f'{output_dir}/y_Pretrain_Discovery_Train.csv', index_col=0)

Z_umap = generate_umap_embedding(X_data)

Z_umap.columns = ['UMAP1', 'UMAP2']
Z_embed = Z_umap.join(y_data)



 # palette = get_color_map(Z_embed[hue_col].nunique())
# Get the counts for each instance of the hue column, and the corresponding colormap
Z_count_sum = (~Z_embed[hue_col].isnull()).sum()
print(f'Number of samples in {eval_name}: {Z_count_sum}')
if Z_count_sum < 10:
    print('too few to plot')


if Z_embed[hue_col].nunique() > 30:
    # if more than 30 unique values, then assume its continuous
    palette = 'flare'
    Z_counts = None
else:
    # if fewer than 30 unique values, then assume its categorical
    # palette = get_color_map(Z_embed[hue_col].nunique())
    palette = assign_color_map(Z_embed[hue_col].unique())
    Z_counts = Z_embed[hue_col].value_counts()

plot_title = f'{setup_id} Latent Space of {eval_name} (N={Z_count_sum})'
# choose the marker size based on the number of nonnan values
# marker_sz = 10/(1+np.log(Z_count_sum))
marker_sz = 100/np.sqrt(Z_count_sum)

## PCA ##
## UMAP ##
fig = sns.scatterplot(data=Z_embed, x='UMAP1', y='UMAP2', hue=hue_col, palette=palette,s=marker_sz,hue_order=cohort_order)
# place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# edit the legend to include the number of samples in each cohort
handles, labels = fig.get_legend_handles_labels()

# Add the counts to the legend if hue_col is categorical
if Z_counts is not None:
    # new_labels = [f'{label} ({Z_embed[Z_embed[hue_col]==label].shape[0]})' for label in labels]
    new_labels = []
    for label in labels:
        # new_labels.append(f'{label} ({Z_counts.loc[eval(label)]})')
        try:
            new_labels.append(f'{label} ({Z_counts.loc[label]})')
        except KeyError:
            new_labels.append(f'{label} ({Z_counts.loc[eval(label)]})')
else:
    new_labels = labels

# make the size of the markers in the handles larger
for handle in handles:
    # print(dir(handle))
    handle.set_markersize(10)
    # handle._sizes = [100]

plt.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue_col)

plt.title(plot_title)


########################################################################################################################
# code to use old RCC 2925 data but with latest code and latest meta-data
# load from /PROCESSED_DATA folder and load in each .csv file
homedir = os.path.expanduser("~")
PROCESSED_DATA_dir = f'{homedir}/PROCESSED_DATA'
# load the latest_metadata
meta_data_df = pd.read_csv(f'{homedir}/latest_metadata.csv', index_col=0)
print(meta_data_df)

# loop through all files to update
failed_match_files = []
for file in os.listdir(f'{PROCESSED_DATA_dir}/old_data'):
    new_y_df = pd.DataFrame()
    # if the file is a .csv file and start with y
    if file.endswith(".csv") and file.startswith('y'):
        print(f'file name is {file}')
        # load the data
        df = pd.read_csv(f'{PROCESSED_DATA_dir}/old_data/{file}', index_col=0)
        # go through each row of the the df
        for index, row in df.iterrows():
            # get the index of the row
            idx = row.name
            # check if the row is in the index of the meta_data_df
            if idx in meta_data_df.index:
                print("there is a match in latest metadata")
                # save this new row from meta_data_df to the new_y_df
                new_y_df = pd.concat([new_y_df, meta_data_df.loc[[idx]]])
            # else printout the idx
            else:
                print(f'idx {idx} not in meta_data_df')
                failed_match_files.append(file)
        # save the updated df
        new_y_df.to_csv(f'{PROCESSED_DATA_dir}/{file}')

# plot sns umap
# remove 'nan' from palette
if 'nan' in palette:
    palette.remove('nan')

print(f"palette is {palette}")
fig = sns.scatterplot(data=Z_embed, x='UMAP1', y='UMAP2', hue=hue_col, palette=palette,s=marker_sz)
# place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# build meta data for RCC all data with shanghai data removed and with train/test/val split changed
# remove all rows from shanghai study (ST002773)
# remove all rows from original_metadata with 'Study ID' == 'ST002773'
original_metadata = pd.read_csv(f'{homedir}/original_metadata.csv', index_col=0)
original_metadata = original_metadata[original_metadata['Study ID'] != 'ST002773']

original_metadata['Pretrain All'] = True
original_metadata['Finetune All'] = False

rcc3_baseline = original_metadata[(original_metadata['Study ID'] == 'ST001237') & (original_metadata['Timepoint']=='baseline')].index.to_list()
original_metadata.loc[rcc3_baseline,'Pretrain All'] = False
original_metadata.loc[rcc3_baseline,'Finetune All'] = True

qc_samples = original_metadata[(original_metadata['Sample_Class']=='NIST1950') |
                            (original_metadata['Sample_Class']=='NIST_1950') |
                            (original_metadata['Sample_Class']=='Study_QC_Sample') |
                            (original_metadata['Sample_Class']=='Study_QAQC')].index.to_list()

original_metadata.loc[qc_samples,'Pretrain All'] = False
original_metadata.loc[qc_samples,'Finetune All'] = False

new_metadata = assign_sets(original_metadata)


########################################################################################################################
metadata_path = "/home/min/INPUT_DATA/metadata.csv"
sample_info = pd.read_csv(metadata_path, index_col=0)
# add a index column that is the same as the index
sample_info['index'] = sample_info.index

########################################################################################################################
# compare old selection df and new selection df
old_selection_df_path = f'{homedir}/INPUT_DATA/selection_df.csv'
new_selection_df_path = f'{homedir}/data_rcc_all_studies/selection_df.csv'

# find the difference beteween the two selection dfs
old_selection_df = pd.read_csv(old_selection_df_path, index_col=0)
new_selection_df = pd.read_csv(new_selection_df_path, index_col=0)

# drop Study ID and Job ID from new_selection_df
new_selection_df.drop(['Study ID', 'Job ID'], axis=1, inplace=True)

# find the difference in Finetune test column between the old and the new selection dfs
diff = old_selection_df['Finetune Test'] != new_selection_df['Finetune Test']
print(f'Number of differences in Finetune Test column: {diff.sum()}')

########################################################################################################################
metadata_path = "/home/min/INPUT_DATA/metadata.csv"
selection_df_path = "/home/min/INPUT_DATA/selection_df.csv"
sample_info = pd.read_csv(metadata_path, index_col=0)
selection_df = pd.read_csv(selection_df_path, index_col=0)
# add a index column that is the same as the index
sample_info['index'] = sample_info.index
print(f"all samples {len(sample_info)}")
# only select a subset from sample_info that has index from selection_df with Pretrain All == True
sample_info = sample_info[sample_info['index'].isin(selection_df[selection_df['Pretrain All'] == True].index)]
# fill ["Cohort Label v0", "is Pediatric", "Sex", "Smoking Status", "IMDC BINARY"] columns from sample_info NaN with 'NA'
# Define the columns to fill NaN values
columns_to_fill = ["Cohort Label v0", "is Pediatric", "Sex", "Smoking Status", "IMDC BINARY"]
# Fill NaN values in the specified columns with 'NA'
sample_info[columns_to_fill] = sample_info[columns_to_fill].fillna('NA')
# Optionally, you can check if the NaN values have been filled
print(sample_info[columns_to_fill].isna().sum())
print(f"pretrain samples {len(sample_info)}")

########################################################################################################################
bs, ss = optimize_splits(
    sample_info=sample_info,
    vars_to_balance=["Cohort Label v0", "is Pediatric", "Sex", "Smoking Status", "IMDC BINARY"],
    num_cols =  ["OS", "Age", "BMI"],
    subject_id_col='index',
    tt_frac = 0.15,
    tv_frac = 0.176
)

# save bs to file
bs.to_csv('bs.csv')

########################################################################################################################
# edit the existing selection_df to use the updated split from splited_metadata.csv

# loop through all rows from splited_metadata.csv
splited_metadata = pd.read_csv('splited_metadata.csv', index_col=0)
selection_df = pd.read_csv('selection_df.csv', index_col=0)

# loop through all rows from splited_metadata.csv
for index, row in splited_metadata.iterrows():
    # get the index of the row
    idx = row.name
    # get the assignment from the row
    assignment = row['tvt_split']
    # check if the row is in the index of the selection_df
    if idx in selection_df.index:
        print("there is a match in selection_df, edit the row")
        # update the selection_df with the new values
        # if the assignment is 'test'
        if assignment == 'test':
            selection_df.loc[idx, 'Pretrain Discovery'] = False
            selection_df.loc[idx, 'Pretrain Test'] = True
            selection_df.loc[idx, 'Pretrain Discovery Train'] = False
            selection_df.loc[idx, 'Pretrain Discovery Val'] = False
            selection_df.loc[idx, 'Set'] = 'Pretrain Test'
        if assignment == 'val':
            selection_df.loc[idx, 'Pretrain Discovery'] = True
            selection_df.loc[idx, 'Pretrain Test'] = False
            selection_df.loc[idx, 'Pretrain Discovery Train'] = False
            selection_df.loc[idx, 'Pretrain Discovery Val'] = True
            selection_df.loc[idx, 'Set'] = 'Pretrain Discovery Val'
        if assignment == 'train':
            selection_df.loc[idx, 'Pretrain Discovery'] = True
            selection_df.loc[idx, 'Pretrain Test'] = False
            selection_df.loc[idx, 'Pretrain Discovery Train'] = True
            selection_df.loc[idx, 'Pretrain Discovery Val'] = False
            selection_df.loc[idx, 'Set'] = 'Pretrain Discovery Train'
    # else printout the idx
    else:
        print(f'idx {idx} not in selection_df')

# save selection_df to file
selection_df.to_csv('selection_df.csv')