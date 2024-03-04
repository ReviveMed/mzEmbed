# %%
import os
import pandas as pd
import numpy as np
# ! pip install supervenn
from supervenn import supervenn

import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_dropbox_dir
from standardize import fill_na_by_cohort, standardize_across_cohorts
from sklearn.decomposition import PCA
import umap

# %% [markdown]
# #### TODO List
#  - save additional files that give information about the total number of peaks in each cohort id after frequency threshold but Before alignment

# %% [markdown]
# ### Specify the directories and files

# %%
dropbox_dir = get_dropbox_dir()
base_dir = os.path.join(dropbox_dir, 'development_CohortCombination','alignment_RCC_2024_Feb_27')

ref_freq = 0.4
input_freq = 0.2
matt_ft_dir = os.path.join(base_dir, 'matt_top_fts')
data_dir = os.path.join(base_dir, 'alignment_id_31', f'merge_reference_freq_th_{ref_freq}_freq_th_{input_freq}')

cohort_ids_to_labels_file = os.path.join(base_dir, 'cohort_ids_to_labels.xlsx')
save_dir = os.path.join(data_dir,'plots')
os.makedirs(save_dir, exist_ok=True)
savefig_params = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.05}

# %% [markdown]
# ## Import Matt's top Features

# %%
matt_ft_files = os.listdir(matt_ft_dir)
matt_ft_files = [f for f in matt_ft_files if f.endswith('.txt')]

matt_ft_dict = {}
for f in matt_ft_files:
    ft_name = f.split('_feats')[0]
    # with open(os.path.join(matt_ft_dir, f), 'r') as file:
    #     ft = file.read().split(', ')
    # if len(ft) == 1:
    with open(os.path.join(matt_ft_dir, f), 'r') as file:
        ft = file.read().splitlines()
        # print(file.read()
    # remove all of the ', and commas from the strings in the list
    ft = [x.strip(',').strip(' ').strip('"').strip("'").strip('\n').strip('\t') for x in ft]
    matt_ft_dict[ft_name] = ft
    # break
    print(ft_name + ': ' + str(len(ft)))


# %% [markdown]
# ## Import the RCC Results

# %%
rcc_peak_info_file = os.path.join(base_dir, 'rcc_result', 'peak_info.csv')
rcc_peak_info_df = pd.read_csv(rcc_peak_info_file, index_col=0)

rcc_peak_info_df = rcc_peak_info_df[rcc_peak_info_df['freq'] >= ref_freq].copy()

print(f'Number of peaks in the reference cohort after {ref_freq} filter: ', rcc_peak_info_df.shape[0])


# %% RCC Targets
    
rcc_matched_targets_file = os.path.join(base_dir,'rcc_result', 'matched_targets HILIC POSITIVE ION MODE.csv')
rcc_matched_targets_df = pd.read_csv(rcc_matched_targets_file, index_col=0)
rcc_matched_targets_df.loc[rcc_peak_info_df.index]

potential_feats = rcc_matched_targets_df[rcc_matched_targets_df['potential_target']].index.to_list()
print('Number of features that potentially match to a target metabolite: ', len(potential_feats))

double_match_ids = rcc_matched_targets_df[rcc_matched_targets_df['potential_target_count'] > 1]
num_double_match = double_match_ids.shape[0]
print('Number of features that potentially match to more than one target metabolite: ', double_match_ids.shape[0])
print(rcc_matched_targets_df.loc[double_match_ids.index, 'potential_target_id'])

# here are the double matches in RCC, two are the same metabolite (but different adducts?)
# FT3202                           tryptophanTryptophan_μM
# FT3237                           kynurenineKynurenine_μM
# FT8451    C18:1 LPC plasmalogen_AC18:1 LPC plasmalogen_B


potential_feat_names = rcc_matched_targets_df.loc[potential_feats]['potential_target_id'].unique()
# print('Number of potential feature names: ', len(potential_feat_names))
print(potential_feat_names)

print('Number of target metabolite captured: ', len(potential_feat_names))

# for now don't remove the double counts, since they are NOT actually double counts
num_rcc_targets_found =  len(potential_feat_names)
rcc_target_feats = potential_feats

# add to the matt ft dictionary
matt_ft_dict['rcc_targets'] = rcc_target_feats

# %%
# Helper functions to finding the number and percentage of captured features
def get_captured_fts(matt_ft_list, align_ft_list):
    captured_fts = [ft for ft in matt_ft_list if ft in align_ft_list]
    return captured_fts

def get_captured_perc(matt_ft_list, align_ft_list):
    captured_fts = get_captured_fts(matt_ft_list, align_ft_list)
    matt_capture_perc = len(captured_fts) / len(matt_ft_list)
    align_capture_perc = len(captured_fts) / len(align_ft_list)
    return matt_capture_perc, align_capture_perc


# check that the Matt feats are found in the RCC peaks
print('In the original RCC dataset at the {} frequency threshold:'.format(ref_freq))
for matt_ft_name, matt_ft_list in matt_ft_dict.items():
    captured_peaks = get_captured_fts(matt_ft_list, rcc_peak_info_df.index)
    print(f'Number of {matt_ft_name} captured: {len(captured_peaks)} out of {len(matt_ft_list)}')

# %% [markdown]
# ## Import the Alignment Results

# %%
cohort_id_file = os.path.join(data_dir, 'combined_study_cohort_ids.csv')
nan_mask_file = os.path.join(data_dir, 'combined_study_nan_mask.csv')
combined_study_file = os.path.join(data_dir, 'combined_study.csv')
align_score_file = os.path.join(data_dir, 'align_score_df.csv')
align_feats_file = os.path.join(data_dir, 'alignment_df.csv')


umap_file = os.path.join(data_dir, 'umap_df_zscore.csv')
pca_file = os.path.join(data_dir, 'pca_df_zscore.csv')

# %%
# Create the pretraining metadata
if '.csv' in cohort_ids_to_labels_file:
    cohort_ids_to_labels_df = pd.read_csv(cohort_ids_to_labels_file, index_col=0)
elif '.xlsx' in cohort_ids_to_labels_file:
    cohort_ids_to_labels_df = pd.read_excel(cohort_ids_to_labels_file, index_col=0)
else:
    raise ValueError('cohort_ids_to_labels_file must be a csv or xlsx file')

cohort_id_df = pd.read_csv(cohort_id_file, index_col=0)
cohort_id_df.set_index('file_name', inplace=True)
cohort_id_df.columns = ['cohort_id']
metadata_df = cohort_id_df.join(cohort_ids_to_labels_df, on='cohort_id')
# turn the cohort id into a string
metadata_df['cohort_id'] = metadata_df['cohort_id'].astype(str)

study_id_to_label = metadata_df.groupby('Study ID')['Cohort Label'].first().to_dict()
cohort_id_to_label = metadata_df.groupby('cohort_id')['Cohort Label'].first().to_dict()
cohort_id_to_study_id = metadata_df.groupby('cohort_id')['Study ID'].first().to_dict()

# %%
# information about the reference cohort
ref_cohort_id = '541'
num_ref_cohort_peaks = rcc_peak_info_df.shape[0]
#TODO Save the number of peaks in the reference cohort used for alignment

# %%
# Get the alignment scores
align_scores = pd.read_csv(align_score_file, index_col=0)
align_scores.index = align_scores.index.astype(str)
cohort_id_to_align_score = align_scores.iloc[:,0].to_dict()
cohort_id_to_align_score[ref_cohort_id] = 1.0

# %%
# Create the unique color maps to consistent plotting across the different plots

# join the following colormaps Accent, Dark2, Set2, Pastel2
my_32_colors = plt.cm.Accent.colors + plt.cm.Dark2.colors + plt.cm.Set2.colors + plt.cm.Pastel2.colors
my_10_colors = plt.cm.tab10.colors
my_20_colors = plt.cm.tab20.colors 
my_42_colors = my_10_colors + my_32_colors
my_52_colors = my_20_colors + my_32_colors

def get_color_map(n):
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
    
def assign_color_map(unique_vals):
    my_colors = get_color_map(len(unique_vals))
    color_map = dict(zip(np.sort(unique_vals), my_colors))
    return color_map

# %%
# Create cohort ids for the RCC subsets
rcc1_files = [f for f in metadata_df.index if 'RCC_HP' in f]
print(len(rcc1_files))
rcc3_files = [f for f in metadata_df[metadata_df['cohort_id']==ref_cohort_id].index if 'RCC_HP' not in f]
print(len(rcc3_files))

metadata_df['Cohort ID Expanded'] = metadata_df['cohort_id']
metadata_df['Study ID Expanded'] = metadata_df['Study ID']
metadata_df.loc[rcc1_files, 'Cohort ID Expanded'] = 'RCC1'
metadata_df.loc[rcc3_files, 'Cohort ID Expanded'] = 'RCC3'
metadata_df.loc[rcc1_files, 'Study ID Expanded'] = 'ST001236'
metadata_df.loc[rcc3_files, 'Study ID Expanded'] = 'ST001237'


# %%
cohort_label_to_color = assign_color_map(metadata_df['Cohort Label'].unique())
study_id_to_color = {k:cohort_label_to_color[v] for k,v in study_id_to_label.items()}
cohort_id_to_color = {k:cohort_label_to_color[v] for k,v in cohort_id_to_label.items()}

study_id_to_uniq_color = assign_color_map(np.sort(metadata_df['Study ID'].unique()))
cohort_id_to_uniq_color = assign_color_map(np.sort(metadata_df['cohort_id'].unique()))

# %%
# create a plot that displays the color along with the cohort label
fig, ax = plt.subplots(figsize=(4,3))
for label, color in cohort_label_to_color.items():
    ax.scatter([], [], c=color, label=label)
ax.legend()
plt.savefig(os.path.join(save_dir, 'cohort_label_color_map.png'), **savefig_params)
plt.close()

# %%
metadata_df.head()

# %%
metadata_summary = pd.DataFrame()
metadata_summary['Cohort ID'] = metadata_df['cohort_id'].unique()
metadata_summary['Cohort ID'] = metadata_summary['Cohort ID'].astype(str)
metadata_summary['Cohort Label'] = metadata_summary['Cohort ID'].map(cohort_id_to_label)
metadata_summary['Study ID'] = metadata_summary['Cohort ID'].map(cohort_id_to_study_id)
metadata_summary['Number of Samples'] = metadata_summary['Cohort ID'].map(metadata_df['cohort_id'].value_counts())
metadata_summary['Alignment Score'] = metadata_summary['Cohort ID'].map(cohort_id_to_align_score)

# %%
print('Number of samples: ' + str(metadata_summary['Number of Samples'].sum()))
print('Number of cohorts: ' + str(metadata_summary.shape[0]))
print('Number of Study IDs: ' + str(metadata_summary['Study ID'].nunique()))
print('Number of Cohort Labels: ' + str(metadata_summary['Cohort Label'].nunique()))

# %%
num_aligned_list = []
matt_captured_peaks_dct = {}
for cohort_id in metadata_summary['Cohort ID']:
    if cohort_id == ref_cohort_id:
        num_aligned_list.append(np.nan)
        for matt_ft_name, matt_ft_list in matt_ft_dict.items():
            # matt_captured_peaks_dct[(cohort_id, matt_ft_name)] = np.nan
            matt_captured_peaks_dct[(cohort_id, matt_ft_name)] = len(matt_ft_list)
        continue
    
    pair_alignment_file = os.path.join(data_dir,f'{cohort_id}_aligned_to_{ref_cohort_id}_with_merge.csv')
    pair_alignment_df = pd.read_csv(pair_alignment_file, index_col=0)
    pair_alignment_df.columns = pair_alignment_df.columns.astype(str)
    pair_alignment_df.dropna(inplace=True)
    num_aligned_fts = pair_alignment_df.shape[0]
    num_aligned_list.append(num_aligned_fts)

    rcc_ft_ids = pair_alignment_df[ref_cohort_id].to_list()
    for matt_ft_name, matt_ft_list in matt_ft_dict.items():
        matt_captured_peaks_dct[(cohort_id, matt_ft_name)] = len(get_captured_fts(matt_ft_list, rcc_ft_ids))

metadata_summary['Number of Initial Aligned Peaks'] = num_aligned_list

# not the correct value of the the original number of peaks, we need other files to extract this information
# metadata_summary['Number of Original Peaks'] = round(metadata_summary['Number of Initial Aligned Peaks']/metadata_summary['Alignment Score'])

for matt_ft_name, matt_ft_list in matt_ft_dict.items():
    metadata_summary[matt_ft_name + ' Captured # (Initial)'] = metadata_summary['Cohort ID'].map(lambda x: matt_captured_peaks_dct[(x, matt_ft_name)])
    metadata_summary[matt_ft_name + ' Captured % (Initial)'] = metadata_summary[matt_ft_name + ' Captured # (Initial)']/len(matt_ft_list)


# %%
metadata_summary.to_csv(os.path.join(data_dir, 'metadata_summary.csv'))
metadata_summary.head()

# %%
# create a plot of the number of samples in each cohort Label
fig, ax = plt.subplots(figsize=(8,6))
temp = metadata_summary.groupby('Cohort Label')['Number of Samples'].sum().sort_values(ascending=False)
temp.plot(kind='bar', ax=ax, color=[cohort_label_to_color[label] for label in temp.index])
ax.set_ylabel('Number of Samples')
ax.set_title('Number of Samples in Each Cohort Label')
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.savefig(os.path.join(save_dir, 'num_samples_per_cohort_label.png'), **savefig_params)
plt.close()


# %%
# create a plot of the number of samples in each Study ID
fig, ax = plt.subplots(figsize=(8,6))
temp = metadata_summary.groupby('Study ID')['Number of Samples'].sum().sort_values(ascending=False)
temp.plot(kind='bar', ax=ax, color=[study_id_to_color[label] for label in temp.index])
ax.set_ylabel('Number of Samples')
ax.set_title('Number of Samples in Each Study ID')
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.savefig(os.path.join(save_dir, 'num_samples_per_study_id.png'), **savefig_params)
plt.close()

# %%
# create a plot of the number of samples in each cohort
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=metadata_summary, x='Cohort ID', y='Number of Samples', palette=cohort_label_to_color, hue='Cohort Label')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Number of Samples')
ax.set_title('Number of Samples in Each Cohort')
# place legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join(save_dir, 'num_samples_per_cohort.png'), **savefig_params)
plt.close()


# %%
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=metadata_summary, x='Cohort ID', y='Alignment Score', palette=cohort_label_to_color, hue='Cohort Label')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Alignment Score')
ax.set_title('Alignment Score in Each Cohort')
# place legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join(save_dir, 'alignment_score_per_cohort.png'), **savefig_params)
plt.close()

# %%
# fig, ax = plt.subplots(figsize=(6,4))
# sns.barplot(data=metadata_summary, x='Cohort ID', y='Number of Original Peaks', palette=cohort_label_to_color, hue='Cohort Label')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# # ax.set_ylabel('Alignment Score')
# ax.set_title('Number of Original Peaks in Each Cohort')
# # place legend outside of the plot
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
if False:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=metadata_summary, x='Cohort ID', y='top_25 Captured % (Initial)', palette=cohort_label_to_color, hue='Cohort Label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylabel('Alignment Score')
    # ax.set_title('Number of Original Peaks in Each Cohort')
    # place legend outside of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(save_dir, 'top_25_feats_captured_per_cohort.png'), **savefig_params)
    plt.close()

    # %%
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=metadata_summary, x='Cohort ID', y='168_os_pfs Captured % (Initial)', palette=cohort_label_to_color, hue='Cohort Label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylabel('Alignment Score')
    # ax.set_title('Number of Original Peaks in Each Cohort')
    # place legend outside of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(save_dir, '168_os_pfs_feats_captured_per_cohort.png'), **savefig_params)
    plt.close()

    # %%
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=metadata_summary, x='Cohort ID', y='net_matched Captured % (Initial)', palette=cohort_label_to_color, hue='Cohort Label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylabel('Alignment Score')
    # ax.set_title('Number of Original Peaks in Each Cohort')
    # place legend outside of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(save_dir, 'net_matched_feats_captured_per_cohort.png'), **savefig_params)

    # %%
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=metadata_summary, x='Cohort ID', y='Number of Initial Aligned Peaks', palette=cohort_label_to_color, hue='Cohort Label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylabel('Alignment Score')
    # ax.set_title('Number of Original Peaks in Each Cohort')
    # place legend outside of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(save_dir, 'num_initial_aligned_peaks_per_cohort.png'), **savefig_params)

# %%
# metadata_summary

# %% [markdown]
# ## Original Peak Intensity Analysis

# %%
# Helper Functions

def create_plot(plot_df, hue_col, palette_dict,include_MV=True,sz=None):
    if 'UMAP1' in plot_df.columns:
        x_col = 'UMAP1'
        y_col = 'UMAP2'
    else:
        x_col = 'PC1'
        y_col = 'PC2'

    if sz is None:
        sz = 10/np.log2(plot_df.shape[0])
        print('marker size: ', sz)

    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, palette=palette_dict, ax=ax, s=sz)
    
        # place the legend outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # edit the legend to include the number of samples in each cohort
    handles, labels = ax.get_legend_handles_labels()
    if include_MV:
        labels = [f'{label} (N={plot_df[plot_df[hue_col]==label].shape[0]}, MV%={plot_df[plot_df[hue_col]==label]["MV"].mean():.0f})' for label in labels]
    else:
        labels = [f'{label} (N={plot_df[plot_df[hue_col]==label].shape[0]})' for label in labels]

    # make the size of the markers in the handles larger
    for handle in handles:
        # print(dir(handle))
        handle.set_markersize(10)
        # handle._sizes = [100]
    
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return fig, ax

# %% [markdown]
# ### UMAP plot

# %%
if False:
    umap_df = pd.read_csv(umap_file, index_col=0)
    umap_df = umap_df.loc[metadata_df.index]
    umap_df['Cohort Label'] = metadata_df['Cohort Label']
    umap_df['Study ID'] = metadata_df['Study ID']
    umap_df['Cohort ID'] = metadata_df['cohort_id']
    umap_df['MV'] = umap_df['MV percentage'].apply(lambda x: x.strip('%')).astype(float)

    # %%
    create_plot(umap_df, 'Cohort Label', cohort_label_to_color)
    plt.savefig(os.path.join(save_dir, 'umap_cohort_label.png'), **savefig_params)
    plt.close()

    # %%
    create_plot(umap_df, 'Cohort ID', cohort_id_to_color)
    plt.savefig(os.path.join(save_dir, 'umap_cohort_id.png'), **savefig_params)
    plt.close()

    # %%
    create_plot(umap_df, 'Cohort ID', cohort_id_to_uniq_color)
    plt.savefig(os.path.join(save_dir, 'umap_cohort_id_uniq_color.png'), **savefig_params)
    plt.close()

    # %%
    create_plot(umap_df, 'Study ID', study_id_to_uniq_color)
    plt.savefig(os.path.join(save_dir, 'umap_study_id_uniq_color.png'), **savefig_params)
    plt.close()

    # %% [markdown]
    # ### PCA Plot

    # %%
    pca_df = pd.read_csv(pca_file, index_col=0)
    pca_df = pca_df.loc[metadata_df.index]
    pca_df['Cohort Label'] = metadata_df['Cohort Label'] #+ '(N=' + metadata_df['cohort_id'].map(metadata_df['cohort_id'].value_counts().astype(str)) + ')'
    pca_df['Study ID'] = metadata_df['Study ID']
    pca_df['Cohort ID'] = metadata_df['cohort_id']
    pca_df['MV'] = pca_df['MV percentage'].apply(lambda x: x.strip('%')).astype(float)


    # %%
    create_plot(pca_df, 'Cohort Label', cohort_label_to_color)
    plt.savefig(os.path.join(save_dir, 'pca_cohort_label.png'), **savefig_params)
    plt.close()


    # %%
    create_plot(pca_df, 'Cohort ID', cohort_id_to_uniq_color)
    plt.savefig(os.path.join(save_dir, 'pca_cohort_id_uniq_color.png'), **savefig_params)
    plt.close()

    # %%
    create_plot(pca_df, 'Study ID', study_id_to_uniq_color)
    plt.savefig(os.path.join(save_dir, 'pca_study_id_uniq_color.png'), **savefig_params)
    plt.close()

# %% [markdown]
# ## Choose a Subset of the Aligned Features

# %%
## Helper Functions

def compute_peak_robustness(alignment_df, nan_mask, metadata_summary,metadata_df,
                            ref_id=ref_cohort_id,group_col='Study ID Expanded'):

    # num_samples = metadata_df[group_col].value_counts()

    num_samples = metadata_summary[['Cohort ID','Number of Samples']].copy()
    num_samples.set_index('Cohort ID', inplace=True)
    total_num_samples = num_samples['Number of Samples'].sum()
    align_summary =  alignment_df.copy()
    align_summary.index = align_summary[ref_id]
    # align_summary.set_index(ref_id, inplace=True)

    nan_locs = align_summary.isna()
    ref_fts = align_summary.index

    for col in align_summary.columns:
        align_summary[col] = ref_fts

    align_summary[nan_locs] = None
    align_summary = align_summary.astype(bool).astype(int).T
    align_summary.index.name = 'Cohort ID'

    align_summary_cohortsize_weighted = align_summary.mul(num_samples['Number of Samples'], axis=0)

    # weight is by the size of the cohort
    peak_robustness_1 = align_summary_cohortsize_weighted.sum(axis=0)/total_num_samples

    # each cohort is weighted equally
    peak_robustness_2 = align_summary.sum(axis=0)/align_summary.shape[0]

    # each weight is by the log(size) of the cohort
    align_summary_logcohortsize_weighted = np.log(1 + align_summary_cohortsize_weighted)
    log_tot_num_samples = np.sum(1+ np.log(num_samples['Number of Samples']))
    peak_robustness_3 = align_summary_logcohortsize_weighted.sum(axis=0)/log_tot_num_samples


    # the peak frequency across the cohorts
    # cohort_peak_frequency = nan_mask.sum(axis=1)

    group_freq_dct = {}
    group_sz_dct = {}
    for group_id in metadata_df[group_col].unique():
        cohort_samples = metadata_df[metadata_df[group_col]==group_id].index
        val = 1 - nan_mask[cohort_samples].mean(axis=1)
        group_freq_dct[group_id] = val
        group_sz_dct[group_id] = len(cohort_samples)

    group_freq_df = pd.DataFrame(group_freq_dct)
    group_sz_df = pd.DataFrame(group_sz_dct, index=['Number of Samples']).T

    # the peak frequency across the group, each group weighted by size
    peak_robustness_4 = group_freq_df.mul(group_sz_df['Number of Samples'], axis=1).sum(axis=1)/group_sz_df['Number of Samples'].sum()
    

    # the peak frequency across the group, each group weighted by log(size)
    group_log_sz_df = np.log(1 + group_sz_df)
    log_group_sz_df = group_freq_df.mul(group_log_sz_df['Number of Samples'], axis=1).sum(axis=1)/group_log_sz_df['Number of Samples'].sum()
    peak_robustness_5 = log_group_sz_df


    peak_robustness_dct = {
        'Found, Cohort Equal Weighted': peak_robustness_2,
        'Found, Cohort Size Weighted': peak_robustness_1,
        'Found, Cohort Log Size Weighted': peak_robustness_3,
        'Freq, Cohort Size Weighted': peak_robustness_4,
        'Freq, Cohort Log Size Weighted': peak_robustness_5
    }

    return peak_robustness_dct

# %%
# Load the original alignment summary
alignment_df = pd.read_csv(align_feats_file)
alignment_df.columns = alignment_df.columns.astype(str)
alignment_df.dropna(axis=0, how='all', inplace=True)

# %% Compute different peak robustness scores
nan_mask = pd.read_csv(nan_mask_file, index_col=0)
nan_mask.columns = nan_mask.columns.astype(str)

# %%
peak_robustness_dct = compute_peak_robustness(alignment_df, nan_mask, metadata_summary, metadata_df)

# %%
v = peak_robustness_dct['Found, Cohort Equal Weighted']
sns.histplot(v, bins=np.linspace(0,1,31))


# %%


# for k,v in peak_robustness_dct.items():

#     fig, ax = plt.subplots(figsize=(6,4))
#     sns.histplot(v, bins=np.linspace(0,1,31), ax=ax)
#     for matt_ft_name, matt_ft_list in matt_ft_dict.items():
#         captured_peaks = get_captured_fts(matt_ft_list, v.index)
#         sns.histplot(v.loc[captured_peaks], bins=np.linspace(0,1,31), ax=ax, label=matt_ft_name, alpha=0.5)

#     plt.title(k)
#     # add a text box with the total number of peaks in the top right corner
#     plt.text(0.95, 0.95, f'Total Peaks: {v.shape[0]}', ha='right', va='top', transform=plt.gca().transAxes)
#     plt.xlabel('Peak Robustness')
#     plt.ylabel('Number of Peaks')
#     plt.savefig(os.path.join(save_dir, f'peak_robustness_{k}.png'), **savefig_params)
#     plt.close()


# for k,v in peak_robustness_dct.items():

#     fig, ax = plt.subplots(figsize=(6,4))
#     sns.histplot(v, bins=np.linspace(0,1,31), ax=ax, stat='density')
#     for matt_ft_name, matt_ft_list in matt_ft_dict.items():
#         captured_peaks = get_captured_fts(matt_ft_list, v.index)
#         sns.histplot(v.loc[captured_peaks], bins=np.linspace(0,1,31), ax=ax, label=matt_ft_name, alpha=0.5,stat='density')

#     plt.title(k)
#     # add a text box with the total number of peaks in the top right corner
#     plt.text(0.95, 0.95, f'Total Peaks: {v.shape[0]}', ha='right', va='top', transform=plt.gca().transAxes)
#     plt.legend()
#     plt.xlabel('Peak Robustness')
#     plt.ylabel('Number of Peaks')
#     plt.savefig(os.path.join(save_dir, f'peak_robustness_{k}_density.png'), **savefig_params)
#     plt.close()

# for k,v in peak_robustness_dct.items():

#     fig, ax = plt.subplots(figsize=(6,4))
#     sns.histplot(v, bins=np.linspace(0,1,31), ax=ax, stat='percent', kde=True)
#     for matt_ft_name, matt_ft_list in matt_ft_dict.items():
#         if len(matt_ft_list) < 27:
#             continue
#         captured_peaks = get_captured_fts(matt_ft_list, v.index)
#         sns.histplot(v.loc[captured_peaks], bins=np.linspace(0,1,31), ax=ax, label=matt_ft_name, alpha=0.5,stat='percent', kde=True)

#     plt.title(k)
#     # add a text box with the total number of peaks in the top right corner
#     plt.text(0.95, 0.95, f'Total Peaks: {v.shape[0]}', ha='right', va='top', transform=plt.gca().transAxes)
#     plt.legend()
#     plt.xlabel('Peak Robustness')
#     plt.ylabel('Number of Peaks')
#     plt.savefig(os.path.join(save_dir, f'peak_robustness_{k}_percent.png'), **savefig_params)
#     plt.close()


for k,v in peak_robustness_dct.items():

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(v, bins=np.linspace(0,1,31), ax=ax, stat='density', kde=True, color='gray')
    for matt_ft_name, matt_ft_list in matt_ft_dict.items():
        captured_peaks = get_captured_fts(matt_ft_list, v.index)
        # KDE of the captured peaks
        sns.kdeplot(v.loc[captured_peaks], ax=ax, label=matt_ft_name)

    plt.title(k)
    # add a text box with the total number of peaks in the top right corner
    plt.text(0.05, 0.95, f'Total Peaks: {v.shape[0]}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.legend()
    plt.xlabel('Peak Robustness')
    plt.ylabel('Number of Peaks')
    plt.savefig(os.path.join(save_dir, f'peak_robustness_{k}_percent_2.png'), **savefig_params)
    plt.close()

# exit()
# %%

# function to get the sets of aligned features across each cohort id
def get_align_summary(alignment_df, ref_id, filter_th=0, incl_ref_col=True):

    # ref_fts = alignment_df[ref_id]
    align_summary = alignment_df.copy()
    align_summary.set_index(ref_id, inplace=True)
    nan_locs = align_summary.isna()
    ref_fts = align_summary.index

    for col in align_summary.columns:
        align_summary[col] = ref_fts

    align_summary[nan_locs] = None  

    if isinstance(filter_th, int) or isinstance(filter_th, float):
        if filter_th > len(align_summary.columns):
            raise ValueError('filter_th must be less than the number of columns in the alignment_df')
        if filter_th > 0:
            align_summary = align_summary[align_summary.count(axis=1) >= filter_th].copy()
        elif filter_th < 0:
            raise ValueError('filter_th must be a positive integer')
        
    elif isinstance(filter_th, list):
        align_summary = align_summary.loc[filter_th,:].copy()
    else:
        raise ValueError('filter_th must be an int or a list')

    if incl_ref_col:
        align_summary[ref_id] = align_summary.index
        align_summary = align_summary[[ref_id] + align_summary.columns.tolist()[:-1]].copy()

    return align_summary

def convert_align_summary_to_sets(align_summary):
    sets = [set(align_summary[col].dropna()) for col in align_summary.columns]
    set_names = align_summary.columns
    return sets, set_names


# %%
align_summary = get_align_summary(alignment_df, ref_cohort_id, filter_th=0)

cohort_id_to_sample_num = metadata_summary[['Cohort ID','Number of Samples']].copy()
cohort_id_to_sample_num.set_index('Cohort ID', inplace=True)

align_summary_bool = align_summary.astype(bool).astype(int).T
align_summary_bool.index.name = 'Cohort ID'

number_samples_by_ft = align_summary_bool.mul(cohort_id_to_sample_num['Number of Samples'], axis=0)
tot_num_samples = cohort_id_to_sample_num['Number of Samples'].sum()
print('Total number of samples: ', tot_num_samples)





# %%
# not_nan_mask = ~pd.read_csv(nan_mask_file, index_col=0)
not_nan_mask = ~nan_mask

output_dct = {}
org_min_num_cohort_ids = -1

for min_num_cohort_ids in range(0,20):
    align_summary = get_align_summary(alignment_df, ref_cohort_id, filter_th=min_num_cohort_ids)

    chosen_feats = align_summary[ref_cohort_id].dropna().to_list()
# print(f'Number of chosen features found in {min_num_cohort_ids+1}/{tot_num_cohort_ids} cohort ids: ', len(chosen_feats))

    sets, set_names = convert_align_summary_to_sets(align_summary)

    subset_summary = metadata_summary.copy()
    subset_summary.set_index('Cohort ID', inplace=True)
    subset_summary['Number of Subset Aligned Peaks'] = align_summary.count(axis=0)
    # subset_summary['Estimated Avg Peak Frequency'] = (number_samples_by_ft[chosen_feats].sum(axis=0)/tot_num_samples).mean()
    
    estimate_avg_peak_freq = (number_samples_by_ft[chosen_feats].sum(axis=0)/tot_num_samples).mean()

    if all([c in not_nan_mask.index for c in chosen_feats]):
        if org_min_num_cohort_ids == -1:
            org_min_num_cohort_ids = min_num_cohort_ids
        avg_peak_freq = (not_nan_mask.loc[chosen_feats].sum(axis=1)/not_nan_mask.shape[1]).mean()
    else:
        avg_peak_freq = np.nan

    matt_captured_peaks = {}
    num_of_matt_fts = {}

    for matt_ft_name, matt_ft_list in matt_ft_dict.items():
        # captured_peaks = [len(get_captured_fts(matt_ft_list, s)) for s in sets]
        captured_peaks = [len(get_captured_fts(matt_ft_list, s)) for s in sets]
        matt_captured_peaks[matt_ft_name] = captured_peaks
        num_of_matt_fts[matt_ft_name] = len(matt_ft_list)
        subset_summary[matt_ft_name + ' Captured # (Subset)'] = captured_peaks
        subset_summary[matt_ft_name + ' Captured % (Subset)'] = round(subset_summary[matt_ft_name + ' Captured # (Subset)'] / num_of_matt_fts[matt_ft_name],3)

    output_dct[min_num_cohort_ids] = {
        'Number of Aligned Peaks': align_summary.shape[0],
        'Avg Peak Frequency': avg_peak_freq,
        'Avg Peak Frequency Estimate' : estimate_avg_peak_freq}


    for matt_ft_name in matt_ft_dict.keys():
        output_dct[min_num_cohort_ids][matt_ft_name + ' Captured # (Subset)'] = subset_summary[matt_ft_name + ' Captured # (Subset)'].max()
        output_dct[min_num_cohort_ids][matt_ft_name + ' Captured % (Subset)'] = subset_summary[matt_ft_name + ' Captured # (Subset)'].max() / num_of_matt_fts[matt_ft_name]
        

output = pd.DataFrame(output_dct).T        

# %%
fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(data=output, x='Number of Aligned Peaks', y='top_10 Captured % (Subset)', label='top_10', ax=ax)
sns.lineplot(data=output, x='Number of Aligned Peaks', y='top_25 Captured % (Subset)', label='top_25', ax=ax)
sns.lineplot(data=output, x='Number of Aligned Peaks', y='net_matched Captured % (Subset)', label='net_matched', ax=ax)
sns.lineplot(data=output, x='Number of Aligned Peaks', y='Avg Peak Frequency Estimate', label='Avg Peak Frequency Estimate', ax=ax)
sns.lineplot(data=output, x='Number of Aligned Peaks', y='Avg Peak Frequency', label='Avg Peak Frequency', ax=ax)
plt.ylabel('Percentage')
plt.xlabel('Number of Aligned Peaks')
# move legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join(save_dir, 'alignment_summary.png'), **savefig_params)
plt.close()


# %% [markdown]
# ### Choose a subset of the aligned peaks

# %%




align_summary_bool = align_summary.astype(bool)
align_counts = align_summary_bool.sum(axis=0)


# %%
# Cluster map of the aligned peaks

# add extra information to the Align Summary columns (such as the number of samples in each cohort, and peak score)
align_summary_plot = align_summary.copy()
align_summary_bool = align_summary.astype(bool)
align_counts = align_summary_bool.sum(axis=0)
align_summary_plot.columns = [
    f'{c} (Nfiles={cohort_id_to_sample_num.loc[c].values[0]}, Npeaks={align_counts[c]:.0f}, Align%={cohort_id_to_align_score[c]:.2f})' 
    for c in align_summary.columns]



row_colors = pd.Series([cohort_id_to_color[c] for c in align_summary.columns])
row_colors.index = align_summary_plot.columns
row_colors.name = 'Cohort Label'


g = sns.clustermap(align_summary_plot.astype(bool).astype(int).T, cmap='viridis', figsize=(10,10), 
            row_colors=row_colors)

plt.setp(g.ax_heatmap.get_xticklabels(), visible=False)
plt.xlabel('Aligned Features')





# %%
for min_num_cohort_ids in range(org_min_num_cohort_ids,16,5):
    # min_num_cohort_ids = 6 #6 was chosen originally
    tot_num_cohort_ids = len(alignment_df.columns)
    align_summary = get_align_summary(alignment_df, ref_cohort_id, filter_th=min_num_cohort_ids)

    chosen_feats = align_summary[ref_cohort_id].dropna().to_list()
    print(f'Number of chosen features found in {min_num_cohort_ids+1}/{tot_num_cohort_ids} cohort ids: ', len(chosen_feats))

    sets, set_names = convert_align_summary_to_sets(align_summary)

    subset_summary = metadata_summary.copy()
    subset_summary.set_index('Cohort ID', inplace=True)
    subset_summary['Number of Subset Aligned Peaks'] = align_summary.count(axis=0)

    # %%
    # find out how many of matt's top features are captured in the algin_summary
    matt_captured_peaks = {}
    num_of_matt_fts = {}

    for matt_ft_name, matt_ft_list in matt_ft_dict.items():
        # captured_peaks = [len(get_captured_fts(matt_ft_list, s)) for s in sets]
        captured_peaks = [len(get_captured_fts(matt_ft_list, s)) for s in sets]
        matt_captured_peaks[matt_ft_name] = captured_peaks
        num_of_matt_fts[matt_ft_name] = len(matt_ft_list)
        subset_summary[matt_ft_name + ' Captured # (Subset)'] = captured_peaks
        subset_summary[matt_ft_name + ' Captured % (Subset)'] = round(subset_summary[matt_ft_name + ' Captured # (Subset)'] / num_of_matt_fts[matt_ft_name],3)

    for matt_ft_name in matt_ft_dict.keys():
        print(f'{matt_ft_name} captured % (Initial): {metadata_summary[matt_ft_name + " Captured % (Initial)"].max()}')
        print(f'{matt_ft_name} captured % (Subset): {subset_summary[matt_ft_name + " Captured % (Subset)"].max()}')

    # %%
    subset_summary.to_csv(os.path.join(data_dir, f'alignment_subset_{min_num_cohort_ids}_summary.csv'))

    # %%
    # print summary
    # print number of features captured in the reference cohort
    # print the nu

    # %%
    align_summary_bool = align_summary.astype(bool)
    align_counts = align_summary_bool.sum(axis=0)


    # %%

    # add extra information to the Align Summary columns (such as the number of samples in each cohort, and peak score)
    align_summary_plot = align_summary.copy()
    align_summary_bool = align_summary.astype(bool)
    align_counts = align_summary_bool.sum(axis=0)
    align_summary_plot.columns = [
        f'{c} (Nfiles={cohort_id_to_sample_num.loc[c].values[0]}, Npeaks={align_counts[c]:.0f}, Align%={cohort_id_to_align_score[c]:.2f})' 
        for c in align_summary.columns]



    row_colors = pd.Series([cohort_id_to_color[c] for c in align_summary.columns])
    row_colors.index = align_summary_plot.columns
    row_colors.name = 'Cohort Label'


    g = sns.clustermap(align_summary_plot.astype(bool).astype(int).T, cmap='viridis', figsize=(10,10), 
                row_colors=row_colors)

    plt.setp(g.ax_heatmap.get_xticklabels(), visible=False)
    plt.xlabel('Aligned Features')

    # remove the color bar
    plt.gcf().axes[-1].remove()
    plt.savefig(os.path.join(save_dir, f'align_summary_H-plot_subst_{min_num_cohort_ids}.png'), **savefig_params)
    plt.close()

    # %%
    # # takes over 7 minutes
    # plt.figure(figsize=(20, 10))

    # # supervenn(sets, set_names,min_width_for_annotation=20,side_plots='right') #color_cycle
    # supervenn(sets, set_names, min_width_for_annotation=20, side_plots='right',color_cycle=[cohort_id_to_color[col] for col in set_names])

    # %% [markdown]
    # 

    # %% [markdown]
    # ## Redo Peak Intensity Analysis


    # %%
    try:
        combined_study = pd.read_csv(combined_study_file, index_col=0)
        combined_study.columns = combined_study.columns.astype(str)

        nan_mask = pd.read_csv(nan_mask_file, index_col=0)
        nan_mask.columns = nan_mask.columns.astype(str)

        # %%
        print('combined_study shape: ', combined_study.shape)
        combined_study = combined_study[metadata_df.index]
        nan_mask = nan_mask[metadata_df.index]

        # cohort_id_list = metadata_df['Study ID Expanded'].tolist() # is it better to use the study id?

        print('combined_study shape after metadata match: ', combined_study.shape)

        # %%
        combined_study_ft_list = combined_study.index.to_list()
        if len(combined_study_ft_list) < len(chosen_feats):
            print('There are fewer features in the combined study than in the chosen features')
            print('select the subset of chosen features')
            chosen_feats = [ft for ft in chosen_feats if ft in combined_study_ft_list]
            print('number of chosen features: ', len(chosen_feats))


        combined_study = combined_study.loc[chosen_feats,:]
        nan_mask = nan_mask.loc[chosen_feats,:]


        # combined_study[nan_mask] = np.nan
        subset_metadata = metadata_df.copy()
        subset_metadata['MV'] = 100*nan_mask.sum(axis=0)/nan_mask.shape[0]



        # %%
        num_missing_vals = nan_mask.sum(axis=0).sort_values(ascending=False)
        num_missing_frac = num_missing_vals/combined_study.shape[0]

        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(num_missing_frac, bins=np.linspace(0,1,21), ax=ax)
        plt.xlabel('Missing Value Fraction')
        plt.ylabel('Number of Samples')
        plt.title('Histogram of Missing Value Fraction')
        plt.savefig(os.path.join(save_dir, f'missing_value_fraction_hist_subset_{min_num_cohort_ids}.png'), **savefig_params)
        plt.close()

        # %%
        # Remove samples with more than 95% missing values
        mis_val_frac_th = 0.95
        print(f'number of samples with more than {100*mis_val_frac_th:.0f}% missing values: ', (num_missing_frac > mis_val_frac_th).sum())
        kept_samples = num_missing_frac[num_missing_frac < mis_val_frac_th].index
        nan_mask = nan_mask.loc[:,kept_samples]
        combined_study = combined_study.loc[:,kept_samples]
        subset_metadata = subset_metadata.loc[kept_samples]
        cohort_id_list = subset_metadata['Cohort ID Expanded'].tolist()

        print('combined_study shape after removing samples with more than 95% missing values: ', combined_study.shape)

        # %%
        # If we want to fill the NA values with something else
        combined_study[nan_mask] = np.nan
        # we want to split up the RCC cohorts into its subsets
        combined_study = fill_na_by_cohort(combined_study, cohort_id_list, method= 'mean_1th')

        # %%
        data_corrected = standardize_across_cohorts(combined_study, cohort_id_list, method='zscore')
        print('data_corrected shape: ', data_corrected.shape)
        # which columns have been removed?
        rem_cols = data_corrected.isna().sum() > 0
        rem_cols = rem_cols[rem_cols].index
        print('Number of columns removed: ', rem_cols.shape[0])
        print('Columns removed: ', rem_cols)
        data_corrected.dropna(axis=1, inplace=True)
        print('data_corrected shape after dropping NA columns: ', data_corrected.shape)
        subset_metadata = subset_metadata.loc[data_corrected.columns]

        # %% [markdown]
        ### PCA Plot

        # %%
        # Generate the PCA plot

        pca_file_subset = os.path.join(data_dir, f'pca_df_zscore_subset_{min_num_cohort_ids}.csv')

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_corrected.T)
        pca_df = pd.DataFrame(pca_result, index=combined_study.columns, columns=['PC1', 'PC2'])
        pca_df['MV'] = subset_metadata['MV']
        pca_df['Cohort Label'] = subset_metadata['Cohort Label']
        pca_df['Study ID'] = subset_metadata['Study ID']
        pca_df['Cohort ID'] = subset_metadata['cohort_id']

        # save the pca_df
        pca_df.to_csv(pca_file_subset)



        # %%
        create_plot(pca_df, 'Cohort Label', cohort_label_to_color)
        plt.savefig(os.path.join(save_dir, f'pca_cohort_label_subset_{min_num_cohort_ids}.png'), **savefig_params)
        plt.close()

        # error occurs here?
        create_plot(pca_df, 'Study ID', study_id_to_uniq_color)
        plt.savefig(os.path.join(save_dir, f'pca_study_id_uniq_color_subset_{min_num_cohort_ids}.png'), **savefig_params)
        plt.close()

        # %% [markdown]
        # ### UMAP Plot
        umap_file_subset = os.path.join(data_dir, f'umap_df_zscore_subset_{min_num_cohort_ids}.csv')


        # %%
        umap_model = umap.UMAP(n_components=2)
        umap_result = umap_model.fit_transform(data_corrected.T)
        umap_df = pd.DataFrame(umap_result, index=combined_study.columns, columns=['UMAP1', 'UMAP2'])
        umap_df['MV'] = subset_metadata['MV']
        umap_df['Cohort Label'] = subset_metadata['Cohort Label']
        umap_df['Study ID'] = subset_metadata['Study ID']
        umap_df['Cohort ID'] = subset_metadata['cohort_id']

        # save the umap_df
        umap_df.to_csv(umap_file_subset)



        # %%
        create_plot(umap_df, 'Cohort Label', cohort_label_to_color)
        plt.savefig(os.path.join(save_dir, f'umap_cohort_label_subset_{min_num_cohort_ids}.png'), **savefig_params)
        plt.close()


        create_plot(umap_df, 'Study ID', study_id_to_uniq_color)
        plt.savefig(os.path.join(save_dir, f'umap_study_id_uniq_color_subset_{min_num_cohort_ids}.png'), **savefig_params)
        plt.close()

        # %%

    except ValueError as e:
        print(e)
        print('PCA and UMAP plots were not generated due to error')
    except FileNotFoundError as e:
        print(e)
        print('file not found?')

