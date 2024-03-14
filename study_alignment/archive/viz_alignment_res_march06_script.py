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
import umap
from sklearn.decomposition import PCA
import pickle

# %% [markdown]
# ## Load Important Stuff



# %% [markdown]
# ### Directories

# %%
# %%
dropbox_dir = get_dropbox_dir()
base_dir = os.path.join(dropbox_dir, 'development_CohortCombination','alignment_RCC_2024_Feb_27')

ref_freq = 0.6
# input_freq = 0.1
grid_id = 1
matt_ft_dir = os.path.join(base_dir, 'matt_top_fts')
# data_dir = os.path.join(base_dir, 'alignment_id_31', f'merge_reference_freq_th_{ref_freq}_freq_th_{input_freq}')
data_dir = os.path.join(base_dir, 'alignment_id_36', f'grid_search_index_{grid_id}')


cohort_ids_to_labels_file = os.path.join(base_dir, 'cohort_ids_to_labels.xlsx')
savefig_params = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.05}

# %% [markdown]
# ## Key Peaks


# %%
savefig_params = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.05}


# %%
# Create the unique color maps to consistent plotting across the different plots

# join the following colormaps Accent, Dark2, Set2, Pastel2

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
    
def assign_color_map(unique_vals):
    my_colors = get_color_map(len(unique_vals))
    color_map = dict(zip(np.sort(unique_vals), my_colors))
    return color_map


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
# ### Matt's Features

# %%
# %%
def get_key_ft_dct():

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
    # ### RCC Target Metabolites

    # %%
    # %%
    rcc_peak_info_file = os.path.join(base_dir, 'rcc_result', 'peak_info.csv')
    rcc_peak_info_df = pd.read_csv(rcc_peak_info_file, index_col=0)

    rcc_peak_info_df = rcc_peak_info_df[rcc_peak_info_df['freq'] >= ref_freq].copy()

    print(f'Number of peaks in the reference cohort after {ref_freq} filter: ', rcc_peak_info_df.shape[0])

        
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

    return matt_ft_dict, rcc_peak_info_df

# %% [markdown]
# ### Helper functions for dealing with the key features

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



# def prep_analysis(data_dir):


save_dir = os.path.join(data_dir,'plots')
os.makedirs(save_dir, exist_ok=True)
matt_ft_dict, rcc_peak_info_df = get_key_ft_dct()



# check that the Matt feats are found in the RCC peaks
print('In the original RCC dataset at the {} frequency threshold:'.format(ref_freq))
for matt_ft_name, matt_ft_list in matt_ft_dict.items():
    captured_peaks = get_captured_fts(matt_ft_list, rcc_peak_info_df.index)
    print(f'Number of {matt_ft_name} captured: {len(captured_peaks)} out of {len(matt_ft_list)}')


# %% [markdown]
# ## Prep Data and Metadata

# %%
# %%
cohort_id_file = os.path.join(data_dir, 'combined_study_cohort_ids.csv')
nan_mask_file = os.path.join(data_dir, 'combined_study_nan_mask.csv')
combined_study_file = os.path.join(data_dir, 'combined_study.csv')
align_score_file = os.path.join(data_dir, 'align_score_df.csv')
align_feats_file = os.path.join(data_dir, 'alignment_df.csv')


umap_file = os.path.join(data_dir, 'umap_df_zscore.csv')
pca_file = os.path.join(data_dir, 'pca_df_zscore.csv')

cohorts_to_rem = ['549']
print('removing cohorts: ', cohorts_to_rem)

# %%
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


metadata_df = metadata_df[~metadata_df['cohort_id'].isin(cohorts_to_rem)]


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



# %%
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
metadata_df['Cohort ID'] = metadata_df['cohort_id']

# %%
cohort_label_to_color = assign_color_map(metadata_df['Cohort Label'].unique())
study_id_to_color = {k:cohort_label_to_color[v] for k,v in study_id_to_label.items()}
cohort_id_to_color = {k:cohort_label_to_color[v] for k,v in cohort_id_to_label.items()}

study_id_to_uniq_color = assign_color_map(np.sort(metadata_df['Study ID'].unique()))
cohort_id_to_uniq_color = assign_color_map(np.sort(metadata_df['cohort_id'].unique()))

# %%
metadata_df.to_csv(os.path.join(data_dir, 'metadata_df.csv'))
color_dct = {
    'cohort_label_to_color': cohort_label_to_color,
    'study_id_to_color': study_id_to_color,
    'cohort_id_to_color': cohort_id_to_color,
    'study_id_to_uniq_color': study_id_to_uniq_color,
    'cohort_id_to_uniq_color': cohort_id_to_uniq_color
}

# save to pickle
with open(os.path.join(data_dir, 'color_dct.pkl'), 'wb') as f:
    pickle.dump(color_dct, f)
    

# %%
metadata_summary = pd.DataFrame()
metadata_summary['Cohort ID'] = metadata_df['cohort_id'].unique()
metadata_summary['Cohort ID'] = metadata_summary['Cohort ID'].astype(str)
metadata_summary['Cohort Label'] = metadata_summary['Cohort ID'].map(cohort_id_to_label)
metadata_summary['Study ID'] = metadata_summary['Cohort ID'].map(cohort_id_to_study_id)
metadata_summary['Number of Samples'] = metadata_summary['Cohort ID'].map(metadata_df['cohort_id'].value_counts())
metadata_summary['Alignment Score'] = metadata_summary['Cohort ID'].map(cohort_id_to_align_score)

metadata_summary.to_csv(os.path.join(data_dir, 'metadata_summary.csv'))
# %%


# %% [markdown]
# ## Compute Nan Mask Metrics

# %%


# %%
def compute_alignment_metrics(not_nan_mask, key_feat_dict, sample_ids=None, feat_ids=None):


    # columns are the samples, rows are the features
    # not_nan_mask is the inverse of the nan_mask, True values indicate the feature has a value that is NOT NaN

    if sample_ids is None:
        sample_ids = not_nan_mask.columns
    if feat_ids is None:
        feat_ids = not_nan_mask.index

    feat_df = pd.DataFrame(index=feat_ids)
    sample_df = pd.DataFrame(index=sample_ids)


    metrics_dct = {}
    metrics_dct['num input samples'] = len(sample_ids)
    metrics_dct['num input feats'] = len(feat_ids)

    # group features are the number of features found in at least one sample in the group
    group_feats = feat_ids[not_nan_mask.loc[feat_ids,sample_ids].any(axis=1)]
    feat_df['found'] = False
    feat_df.loc[group_feats, 'found'] = True
    metrics_dct['num found feats'] = len(group_feats)

    # sample_group_freq is the number of samples in the group that have a value for the feature
    sample_group_freq = not_nan_mask.loc[group_feats,sample_ids].sum(axis=1) / len(sample_ids)
    feat_df['freq'] = np.nan
    feat_df.loc[group_feats,'freq'] = sample_group_freq
    metrics_dct['average feat freq'] = sample_group_freq.mean()

    # feat_group_freq is the number of group-features in the group that have a value for the sample
    feat_group_freq = not_nan_mask.loc[group_feats,sample_ids].sum(axis=0) / len(group_feats)
    sample_df['freq'] = feat_group_freq
    sample_df['overall freq'] = not_nan_mask.loc[feat_ids,sample_ids].sum(axis=0) / len(feat_ids)
    metrics_dct['average sample freq'] = feat_group_freq.mean()
    metrics_dct['average overall sample freq'] = sample_df['overall freq'].mean()


    metrics_dct['value fraction'] = not_nan_mask.loc[group_feats,sample_ids].sum().sum()/ (len(group_feats) * len(sample_ids))
    metrics_dct['value frac ALL FEATs'] = not_nan_mask.loc[feat_ids,sample_ids].sum().sum() / (len(feat_ids) * len(sample_ids))

    for key_name, key_feats in key_feat_dict.items():

        # print(key_name)
        captured_feats = [ft for ft in key_feats if ft in feat_ids]
        key_sample_group_freq = not_nan_mask.loc[captured_feats,sample_ids].sum(axis=1) / len(sample_ids)
        feat_df[key_name] = np.nan
        feat_df.loc[captured_feats, key_name] = key_sample_group_freq

        key_feat_group_freq = not_nan_mask.loc[captured_feats,sample_ids].sum(axis=0) / len(key_feats)
        sample_df[key_name] = key_feat_group_freq

        specific_captured_peaks = get_captured_fts(key_feats, group_feats)
        metrics_dct[key_name + ' captured frac'] = len(specific_captured_peaks) / len(key_feats)

    metrics_dct['num samples below group 10%'] = (sample_df['freq'].fillna(0) < 0.1).sum()
    metrics_dct['num samples below overall 10%'] = (sample_df['overall freq'].fillna(0) < 0.1).sum()

    return metrics_dct, feat_df, sample_df
    # return metrics_dct


############################################################################################################

# %%
# Nan Mask turns into the "not nan" mask
nan_mask = pd.read_csv(nan_mask_file, index_col=0)
nan_mask.columns = nan_mask.columns.astype(str)
nan_mask = nan_mask[metadata_df.index]
nan_mask = ~nan_mask


metrics_dct, feat_df, sample_df = compute_alignment_metrics(nan_mask, matt_ft_dict)

# save the metrics
metrics_dct_df = pd.DataFrame(metrics_dct, index=[0]).to_csv(os.path.join(data_dir, 'overall_alignment_result_metrics.csv'))

# save the feat_df and sample_df
feat_df.to_csv(os.path.join(data_dir, 'overall_feat_df.csv'))
sample_df.to_csv(os.path.join(data_dir, 'overall_sample_df.csv'))




# %%
# group_col = 'Study ID Expanded'
group_col = 'Study ID'
feat_ids = nan_mask.index

# group_samples = metadata_df[metadata_df[group_col] == 'ST001237'].index
group_samples = metadata_df[metadata_df[group_col] == 'RCC'].index

print('number of samples: ', len(group_samples))
group_feats = feat_ids[nan_mask[group_samples].any(axis=1)]
print('number of features: ', len(group_feats))

# %%
group_col = 'Study ID Expanded'
unique_group_cols = metadata_df[group_col].unique()

all_metrics = []

for group_id in unique_group_cols:
    group_samples = metadata_df[metadata_df[group_col] == group_id].index

    metrics_dct, feat_df, sample_df = compute_alignment_metrics(nan_mask, matt_ft_dict, group_samples.to_list())
    metrics_dct['study_id'] = group_id
    all_metrics.append(metrics_dct)

all_metrics_df = pd.DataFrame(all_metrics)
all_metrics_df.to_csv(os.path.join(data_dir, 'alignment_result_metrics.csv'))

# %%
# group_col = 'Study ID Expanded'
# group_id = 'ST000422'
# group_samples = metadata_df[metadata_df[group_col] == group_id].index
# metrics_dct, feat_df, sample_df = compute_alignment_metrics(nan_mask, matt_ft_dict, group_samples.to_list())
# print(metrics_dct)

# %%
# sample_df['freq'].hist()

# %%
# feat_df.hist(column='freq', bins=50)

# %% [markdown]
# ## Analysis of Choosing different thresholds for Frequency and min number of files

# %%
group_col = 'Study ID Expanded'

group_freq = []
overall_freq = []
group_feats_dict = {}
group_samples_dict = {}
feat_ids = nan_mask.index


for sample_id in metadata_df.index:
    group_id = metadata_df.loc[sample_id, group_col]
    if group_id not in group_feats_dict.keys():
        print(group_id)
        # group_feats[group_id] = feat_ids[nan_mask[metadata_df[metadata_df[group_col] == group_id].index].any(axis=0)]
        group_samples = metadata_df[metadata_df[group_col] == group_id].index
        print('number of samples: ', len(group_samples))
        group_feats = feat_ids[nan_mask[group_samples].any(axis=1)]
        print('number of aligned features: ', len(group_feats))
        group_feats_dict[group_id] = group_feats


    group_freq.append(nan_mask.loc[group_feats_dict[group_id], sample_id].mean())

    overall_freq.append(nan_mask.loc[:, sample_id].mean())


metadata_df['group_freq'] = group_freq
metadata_df['overall_freq'] = overall_freq

# %%
# How many features are found in each group?
print('Number of features found in each group:')
for group_id, group_feats in group_feats_dict.items():
    print(group_id, ':', len(group_feats))

# %%
sns.displot(metadata_df, x='group_freq', hue=group_col, kind='kde')

# %%
# Frequency of each peak within each group
print('group col:', group_col)

feat_group_freq_info = pd.DataFrame()
feat_group_freq_info.index = nan_mask.index

for group_id, group_feats in group_feats_dict.items():
    group_samples = metadata_df[metadata_df[group_col] == group_id].index
    feat_group_freq_info[group_id] = nan_mask.loc[group_feats,group_samples].mean(axis=1)


num_samples_per_group = metadata_df.groupby(group_col).size()   


feat_group_num_samples = feat_group_freq_info * num_samples_per_group

# %%


# %%
feat_fewest_samples = feat_group_num_samples.min(axis=1)
feat_smallest_freq = feat_group_freq_info.min(axis=1)

# %%
# Try different thresholds and see how that changes the number of features
min_num_samples = 10
min_freq = 0.1
feat_ids = feat_group_freq_info.index
print('number of features before frequency filtering: ', len(feat_ids))

new_feats = feat_ids[(feat_fewest_samples > min_num_samples) & (feat_smallest_freq > min_freq)]
print('number of features after frequency filtering: ', len(new_feats))

# check overlap with Matt features
for matt_ft_name, matt_ft_list in matt_ft_dict.items():
    captured_peaks = get_captured_fts(matt_ft_list, new_feats)
    print(f'Number of {matt_ft_name} captured: {len(captured_peaks)} out of {len(matt_ft_list)}')

# %%
print('group col:', group_col)

out_dct_list = []
for min_num_samples in range(0,40):
    for min_freq in [0.1,0.15,0.2,0.25,0.30,0.35]:
        new_feats = feat_ids[(feat_fewest_samples > min_num_samples) & (feat_smallest_freq > min_freq)]
        
        out_dict = {'min_num_samples': min_num_samples, 'min_freq': min_freq, 'num_feats': len(new_feats)}

        for matt_ft_name, matt_ft_list in matt_ft_dict.items():
            captured_peaks = get_captured_fts(matt_ft_list, new_feats)
            out_dict[f'num_{matt_ft_name}'] = len(captured_peaks)
            out_dict[f'frac_{matt_ft_name}'] = len(captured_peaks)/ len(matt_ft_list)
        out_dct_list.append(out_dict)

out_df = pd.DataFrame(out_dct_list)

# %%
# out_df.plot(x='min_num_samples', y='num_feats', label='num_feats')
ax, fig = plt.subplots()
out_df[out_df['min_freq']==0.1].plot(x='min_num_samples', y='frac_rcc_targets', label='frac_rcc_targets', kind='line', ax=fig, color='r')
out_df[out_df['min_freq']==0.1].plot(x='min_num_samples', y='frac_top_25', label='frac_top_25', kind='line', ax=fig)
plt.ylabel('Fraction of features captured')
plt.savefig(os.path.join(save_dir, 'feature_capture_vs_sample_num_thresh.png'), **savefig_params)
plt.close()

# %%
# out_df.plot(x='min_num_samples', y='num_feats', label='num_feats')
ax, fig = plt.subplots()
out_df[out_df['min_num_samples']==10].plot(x='min_freq', y='frac_rcc_targets', label='frac_rcc_targets', kind='line', ax=fig, color='r')
out_df[out_df['min_num_samples']==10].plot(x='min_freq', y='frac_top_25', label='frac_top_25', kind='line', ax=fig)
plt.ylabel('Fraction of features captured')
plt.savefig(os.path.join(save_dir, 'feature_capture_vs_min_freq.png'), **savefig_params)
plt.close()


# %% [markdown]
# 

# %%
print('group col:', group_col)
group_freq_dct = {}
group_sz_dct = {}
for group_id in metadata_df[group_col].unique():
    group_samples = metadata_df[metadata_df[group_col]==group_id].index
    val = nan_mask[group_samples].mean(axis=1)
    group_freq_dct[group_id] = val
    group_sz_dct[group_id] = len(group_samples)

group_freq_df = pd.DataFrame(group_freq_dct)
group_sz_df = pd.DataFrame(group_sz_dct, index=['Number of Samples']).T

# %%
group_freq_df

# %% [markdown]
# ## Bar Plots by the Cohort and Study Ids

# %%
print('total number of samples', metadata_summary['Number of Samples'].sum())


# %%
fig, ax = plt.subplots(figsize=(7,5))
temp = metadata_summary.groupby('Cohort Label')['Number of Samples'].sum().sort_values(ascending=False)
temp.plot(kind='bar', ax=ax, color=[cohort_label_to_color[label] for label in temp.index])
ax.set_ylabel('Number of Samples')
ax.set_title('Number of Samples in Each Cohort Label')
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.savefig(os.path.join(save_dir, 'num_samples_per_cohort_label.png'), **savefig_params)
plt.close()

# %%
fig, ax = plt.subplots(figsize=(8,6))
temp = metadata_summary.groupby('Study ID')['Number of Samples'].sum().sort_values(ascending=False)
temp.plot(kind='bar', ax=ax, color=[study_id_to_color[label] for label in temp.index])
ax.set_ylabel('Number of Samples')
ax.set_title('Number of Samples in Each Study ID')
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.savefig(os.path.join(save_dir, 'num_samples_per_study_id.png'), **savefig_params)
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

# %% [markdown]
# ## Peak Robustness Analysis

# %%
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
        # val = 1 - nan_mask[cohort_samples].mean(axis=1) #if nan mask is actually locations of nans
        val = nan_mask[cohort_samples].mean(axis=1) #if nan mask is actually locations of the not nans
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
alignment_df = alignment_df[metadata_summary['Cohort ID'].unique()]

# %% Compute different peak robustness scores and related analysis

# %%
peak_robustness_dct = compute_peak_robustness(alignment_df, nan_mask, metadata_summary, metadata_df)


# %%
robustness_df = pd.DataFrame(peak_robustness_dct)
robustness_df.to_csv(os.path.join(data_dir, 'peak_robustness.csv'))



# %%
feat_info = pd.DataFrame(index=alignment_df[ref_cohort_id])
feat_info.index.name = 'Aligned Features'
for k,v in matt_ft_dict.items():
    feat_info[k] = False
    overlap = [x for x in v if x in feat_info.index]
    print(f'From {k} there are {len(overlap)} aligned out of {len(v)} features.')
    feat_info.loc[overlap, k] = True



feat_info_2 = feat_info.copy()
feat_info_2['label'] = 'others'
feat_info_2.loc[feat_info_2['rcc_targets'], 'label'] = 'rcc_targets'
feat_info_2.loc[feat_info_2['net_matched'], 'label'] = 'net_matched'
feat_info_2.loc[feat_info_2['top_25'], 'label'] = 'top_25'

for k,v in peak_robustness_dct.items():
    feat_info_2[k] = v    

# %%
for k in peak_robustness_dct.keys():

    # fig, ax = plt.subplots(figsize=(8,6))
    sns.displot(data=feat_info_2, x=k, hue='label', fill=True, multiple='stack')
    ax.set_title(k)
    plt.savefig(os.path.join(save_dir, f'x_{k}_hist.png'), **savefig_params)
    plt.close()

    # fig, ax = plt.subplots(figsize=(8,6))
    sns.displot(data=feat_info_2, x=k, hue='label', fill=True, kind='kde', common_norm=False, bw_adjust=0.5)
    ax.set_title(k)
    plt.savefig(os.path.join(save_dir, f'x_{k}_kde.png'), **savefig_params)
    plt.close()



chosen_robustness_metric_list= peak_robustness_dct.keys()

for chosen_robustness_metric in chosen_robustness_metric_list:
    robustnesss_thresholds = [0,0.05,0.1,0.15,0.2,0.25,0.3, 0.33, 0.35, 0.4, 0.45, 0.5]
    captured_info_list = []
    for threshold in robustnesss_thresholds:

        captured_peaks = feat_info_2[feat_info_2[chosen_robustness_metric] > threshold].index

        # sparsity_val = 1 - nan_mask.loc[captured_peaks].mean().mean()
        overall_freq = nan_mask.loc[captured_peaks].mean().mean()

        num_samples_below_10 = (nan_mask.loc[captured_peaks].mean(axis=1) < 0.1).sum()

        captured_info = {
            'robustness threshold': threshold,
            'number of peaks': len(captured_peaks),
            'overall freq': overall_freq,
            # '% of RCC peaks': get_captured_perc(rcc_peak_info_df.index, captured_peaks)[0],
            '% of RCC peaks': len(captured_peaks)/num_ref_cohort_peaks,
            '% of Matt top 25': get_captured_perc(matt_ft_dict['top_25'], captured_peaks)[0],
            '% of network peaks': get_captured_perc(matt_ft_dict['net_matched'], captured_peaks)[0],
            '% of RCC targets': get_captured_perc(matt_ft_dict['rcc_targets'], captured_peaks)[0],
            'num samples below 10% freq': num_samples_below_10
        }

        captured_info_list.append(captured_info)
        print(f'Number of peaks with robustness greater than {threshold}: ', (feat_info_2[chosen_robustness_metric] > threshold).sum())


    captured_df = pd.DataFrame(captured_info_list)

    captured_df.to_csv(os.path.join(save_dir, f'captured_info_{chosen_robustness_metric}.csv'))


# %% [markdown]
# ## Create the H-clustering Plots


def subset_analysis(peak_robust_name='Found, Cohort Size Weighted',peak_robust_th=0.2,rem_cohorts=['549'],recompute_robustness=False):

    subset_id = f'robust_{peak_robust_name}_{peak_robust_th}_rem_{rem_cohorts}'
    if recompute_robustness:
        subset_id = subset_id + '_recompute'
    subset_dir = os.path.join(data_dir, 'subset_'+subset_id)
    os.makedirs(subset_dir, exist_ok=True)


    # metadata = pd.read_csv(os.path.join(data_dir, 'metadata_df.csv'), index_col=0)

    metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata_df.csv'), index_col=0)
    metadata_summary = pd.read_csv(os.path.join(data_dir, 'metadata_summary.csv'), index_col=0)
    metadata_summary['Cohort ID'] = metadata_summary['Cohort ID'].astype(str)
    metadata_df['cohort_id']=metadata_df['cohort_id'].astype(str)
    metadata_df['Cohort ID']= metadata_df['Cohort ID'].astype(str)

    if len(rem_cohorts)> 0:
        metadata_df = metadata_df[~metadata_df['cohort_id'].isin(rem_cohorts)]
        metadata_summary = metadata_summary[~metadata_summary['Cohort ID'].isin(rem_cohorts)]
        cohort_id_list = metadata_df['cohort_id'].unique()

    # check if nan_mask is already in memory
    if 'nan_mask' not in locals():
        nan_mask = pd.read_csv(nan_mask_file, index_col=0)
        nan_mask.columns = nan_mask.columns.astype(str)
        nan_mask = ~nan_mask

    # check if alignment_df is already in memory
    if 'alignment_df' not in locals():
        alignment_df = pd.read_csv(align_feats_file)
        alignment_df.columns = alignment_df.columns.astype(str)
        alignment_df.dropna(axis=0, how='all', inplace=True)
        alignment_df = alignment_df[metadata_summary['Cohort ID'].unique()]

    # check if peak_robustness_dct is already in memory

    if recompute_robustness:
        peak_robustness_dct = compute_peak_robustness(alignment_df, nan_mask, metadata_summary, metadata_df)
        peak_robustness = pd.DataFrame(peak_robustness_dct)
    else:
        if 'peak_robustness_dct' not in locals():
            peak_robustness = pd.read_csv(os.path.join(data_dir, 'peak_robustness.csv'), index_col=0)
        else:
            peak_robustness = pd.DataFrame(peak_robustness_dct)
# %%

    # %%
    group_col = 'cohort_id'    
    group_freq_dct = {}
    group_sz_dct = {}
    for group_id in metadata_df[group_col].unique():
        cohort_samples = metadata_df[metadata_df[group_col]==group_id].index
        val = nan_mask[cohort_samples].mean(axis=1)
        group_freq_dct[group_id] = val
        group_sz_dct[group_id] = len(cohort_samples)

    group_freq_df = pd.DataFrame(group_freq_dct)
    group_sz_df = pd.DataFrame(group_sz_dct, index=['Number of Samples']).T

    feat_info_3 = feat_info.join(group_freq_df)



    # %%
    col_cluster = False
    row_cluster = False

    cohort_ids = metadata_df['cohort_id'].unique()
    plot_df = feat_info_3[cohort_ids].T

    col_colors = pd.DataFrame(index=plot_df.columns)
    for col in feat_info_3.columns:
        if col in cohort_ids:
            continue
        col_colors[col] = feat_info_3[col].map({True: 'black', False: 'white'})

    col_colors[f'Robustness ({peak_robust_name})'] = peak_robustness_dct[peak_robust_name].map(lambda x: plt.cm.plasma(x))
    col_colors[f'Pass Threshold ({peak_robust_th})'] = (peak_robustness_dct[peak_robust_name] > peak_robust_th).map({True: 'black', False: 'white'})

    ####
    # Change the names of the rows
    cohort_id_to_sample_num = metadata_summary[['Cohort ID','Number of Samples']].copy()
    cohort_id_to_sample_num.set_index('Cohort ID', inplace=True)
    align_summary_bool = ~alignment_df.isna()
    align_counts = align_summary_bool.sum(axis=0)

    if col_cluster==False:
        col_val_order = peak_robustness_dct[peak_robust_name]
        col_val_order.sort_values(ascending=False, inplace=True)
        plot_df = plot_df[col_val_order.index]

    if row_cluster==False:
        row_val_order = group_sz_df['Number of Samples'].copy()
        row_val_order.sort_values(ascending=False, inplace=True)
        plot_df = plot_df.loc[row_val_order.index]


    row_colors = pd.DataFrame(index=plot_df.index)
    row_colors['Cohort Label'] = [cohort_id_to_color[c] for c in plot_df.index]
    row_colors['Cohort Size'] = [group_sz_df.loc[c, 'Number of Samples'] for c in plot_df.index]
    row_colors['Cohort Size'] = row_colors['Cohort Size']/row_colors['Cohort Size'].max()
    row_colors['Cohort Size'] = row_colors['Cohort Size'].map(lambda x: plt.cm.copper_r(x))




    new_row_names = [f'{c} (Nfiles={cohort_id_to_sample_num.loc[c].values[0]}, Npeaks={align_counts[c]:.0f}, Align%={cohort_id_to_align_score[c]:.2f})' 
            for c in plot_df.index]





    plot_df.index =  new_row_names
    row_colors.index = new_row_names

    g = sns.clustermap(plot_df, cmap='viridis', figsize=(50,20), 
                row_colors=row_colors, col_colors=col_colors, 
                col_cluster=col_cluster, row_cluster=row_cluster,
                dendrogram_ratio=0.04)

    plt.setp(g.ax_heatmap.get_xticklabels(), visible=False)
    # add a label to the colorbar 
    g.cax.set_ylabel('Peak Frequency')



    plt.savefig(os.path.join(subset_dir, 'supervenn_clustermap.png'), **savefig_params)
    plt.close()

    # %%
    col_cluster = False
    row_cluster = False

    cohort_ids = metadata_df['cohort_id'].unique()
    plot_df = feat_info_3[cohort_ids].T

    col_colors = pd.DataFrame(index=plot_df.columns)
    for col in feat_info_3.columns:
        if col in cohort_ids:
            continue
        col_colors[col] = feat_info_3[col].map({True: 'black', False: 'white'})

    col_colors[f'Robustness ({peak_robust_name})'] = peak_robustness_dct[peak_robust_name].map(lambda x: plt.cm.plasma(x))
    col_colors[f'Pass Threshold ({peak_robust_th})'] = (peak_robustness_dct[peak_robust_name] > peak_robust_th).map({True: 'black', False: 'white'})

    ####
    # Change the names of the rows
    cohort_id_to_sample_num = metadata_summary[['Cohort ID','Number of Samples']].copy()
    cohort_id_to_sample_num.set_index('Cohort ID', inplace=True)
    # align_summary_bool = alignment_df.astype(bool)
    align_summary_bool = ~alignment_df.isna()
    align_counts = align_summary_bool.sum(axis=0)

    if col_cluster==False:
        col_val_order = peak_robustness_dct[peak_robust_name]
        col_val_order.sort_values(ascending=False, inplace=True)
        plot_df = plot_df[col_val_order.index]

    if row_cluster==False:
        row_val_order = group_sz_df['Number of Samples'].copy()
        row_val_order.sort_values(ascending=False, inplace=True)
        plot_df = plot_df.loc[row_val_order.index]


    row_colors = pd.DataFrame(index=plot_df.index)
    row_colors['Cohort Label'] = [cohort_id_to_color[c] for c in plot_df.index]
    row_colors['Cohort Size'] = [group_sz_df.loc[c, 'Number of Samples'] for c in plot_df.index]
    row_colors['Cohort Size'] = row_colors['Cohort Size']/row_colors['Cohort Size'].max()
    row_colors['Cohort Size'] = row_colors['Cohort Size'].map(lambda x: plt.cm.copper_r(x))




    new_row_names = [f'{c} (Nfiles={cohort_id_to_sample_num.loc[c].values[0]}, Npeaks={align_counts[c]:.0f}, Align%={cohort_id_to_align_score[c]:.2f})' 
            for c in plot_df.index]





    plot_df.index =  new_row_names
    row_colors.index = new_row_names

    g = sns.clustermap(plot_df, cmap='viridis', figsize=(8,8), 
                row_colors=row_colors, col_colors=col_colors, 
                col_cluster=col_cluster, row_cluster=row_cluster,
                dendrogram_ratio=0.1)

    plt.setp(g.ax_heatmap.get_xticklabels(), visible=False)
    # add a label to the colorbar 
    g.cax.set_ylabel('Peak Frequency')

    plt.savefig(os.path.join(subset_dir, 'supervenn_clustermap_small.png'), **savefig_params)
    plt.close()





    # %%
    rcc_metadata_file = os.path.join(dropbox_dir, 'development_CohortCombination','clean_rcc_metadata.csv')

    desc_str = subset_id
    rcc_metadata = pd.read_csv(rcc_metadata_file, index_col=0)
    # Join the two metadata together
    metadata_df = metadata_df.join(rcc_metadata, how='outer')
    metadata_df.to_csv(os.path.join(data_dir, 'subset_metadata_with_rcc.csv'))
    robust_score = peak_robustness[peak_robust_name]
    keep_peaks = peak_robustness[robust_score >=peak_robust_th].index
    keep_samples = metadata_df.index

    print(f'Keeping {len(keep_peaks)} peaks and {len(keep_samples)} samples')

    # %%
    # group_id = 'ST001237'
    # group_samples = metadata_df[metadata_df[group_col] == group_id].index

    # metrics_dct, feat_df, sample_df = compute_alignment_metrics(nan_mask, matt_ft_dict, group_samples.to_list(), keep_peaks)

    # %%
    group_col = 'Study ID Expanded'
    unique_group_cols = metadata_df[group_col].unique()

    all_metrics = []



    for group_id in unique_group_cols:
        group_samples = metadata_df[metadata_df[group_col] == group_id].index

        metrics_dct, feat_df, sample_df = compute_alignment_metrics(nan_mask, matt_ft_dict, group_samples.to_list(), keep_peaks)
        metrics_dct['study_id'] = group_id
        all_metrics.append(metrics_dct)

    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(os.path.join(subset_dir, 'alignment_result_metrics.csv'))

    # %%

    # %%

    metrics_dct, feat_df, sample_df = compute_alignment_metrics(nan_mask, matt_ft_dict, keep_samples, keep_peaks)

    # save the results
    metrics_dct = pd.Series(metrics_dct)
    metrics_dct.to_csv(os.path.join(subset_dir, 'overall_alignment_result_metrics.csv'))

    feat_df.to_csv(os.path.join(subset_dir, 'overall_alignment_result_feat_df.csv'))

    sample_df.to_csv(os.path.join(subset_dir, 'overall_alignment_result_sample_df.csv'))


    # %%
    # histogram of the frequency of the peaks

    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(data=feat_df, x='freq', bins=np.arange(0,1.1,0.05))
    ax.set_title('Frequency of Peaks (N={}) Across N={} samples'.format(len(feat_df), len(sample_df)))
    plt.savefig(os.path.join(subset_dir, 'peak_freq_hist.png'), **savefig_params)
    plt.close()

    # %%
    # histogram of frequency of the RCC target peaks
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(data=feat_df, x='rcc_targets', bins=np.arange(0,1.1,0.05))
    ax.set_title('Frequency of RCC Target Peaks (N={}) Across N={} samples'.format(feat_df['rcc_targets'].notna().sum(), len(sample_df)))
    plt.savefig(os.path.join(subset_dir, 'rcc_target_freq_hist.png'), **savefig_params)
    plt.close()

    # %%


    combined_study = pd.read_csv(os.path.join(data_dir, 'combined_study.csv'), index_col=0)
    nan_mask = pd.read_csv(os.path.join(data_dir, 'combined_study_nan_mask.csv'), index_col=0)
    combined_study.columns = combined_study.columns.astype(str)
    nan_mask.columns = nan_mask.columns.astype(str)
    combined_study = combined_study.loc[keep_peaks, keep_samples].copy()
    nan_mask = nan_mask.loc[keep_peaks, keep_samples].copy()

    metadata_df['MV'] = 100*nan_mask.sum(axis=0)/nan_mask.shape[0]



            # %%
    num_missing_vals = nan_mask.sum(axis=0).sort_values(ascending=False)
    num_missing_frac = num_missing_vals/combined_study.shape[0]

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(num_missing_frac, bins=np.linspace(0,1,21), ax=ax)
    plt.xlabel('Missing Value Fraction')
    plt.ylabel('Number of Samples')
    plt.title('Histogram of Missing Value Fraction')
    plt.savefig(os.path.join(save_dir, f'missing_value_fraction_hist_subset_{subset_id}.png'), **savefig_params)
    plt.close()

    # %%
    # Remove samples with more than 95% missing values
    mis_val_frac_th = 0.95
    print(f'number of samples with more than {100*mis_val_frac_th:.0f}% missing values: ', (num_missing_frac > mis_val_frac_th).sum())
    kept_samples = num_missing_frac[num_missing_frac < mis_val_frac_th].index
    nan_mask = nan_mask.loc[:,kept_samples]
    combined_study = combined_study.loc[:,kept_samples]
    subset_metadata = metadata_df.loc[kept_samples]
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
    # Choose to look at only a subset of the data

    # subset_metadata = metadata[metadata['Cohort Label'].isin(['adult_cancer','adult_other'])].copy()
    subset_nan_mask = nan_mask[subset_metadata.index]

    # successively remove the worst samples and features based on the nan_mask
    # for thresh in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]:
    for thresh in [0.95, 0.9]:
        subset_nan_mask = subset_nan_mask[subset_metadata.index]
        good_samples = subset_nan_mask.sum(axis=0) < subset_nan_mask.shape[0]*thresh
        subset_nan_mask = subset_nan_mask[good_samples[good_samples].index]
        good_features = subset_nan_mask.sum(axis=1) < subset_nan_mask.shape[1]*thresh
        subset_nan_mask = subset_nan_mask.loc[good_features[good_features].index]
        subset_metadata = subset_metadata.loc[subset_nan_mask.columns]
        print(subset_nan_mask.shape)

    # Assign the new subsets to overwrite the original data
    if True:

        data_corrected = data_corrected.loc[subset_nan_mask.index, subset_nan_mask.columns]
        nan_mask = nan_mask.loc[subset_nan_mask.index, subset_nan_mask.columns]
        metadata_df = metadata_df.loc[subset_nan_mask.columns]
        print(metadata_df.shape, data_corrected.shape, nan_mask.shape)


        # remove the features with zero variance
        # good_features2 = data_corrected.var(axis=1) > 0
        # good_features2= good_features2[good_features2].index
        # data_corrected = data_corrected.loc[good_features2]
        # nan_mask = nan_mask.loc[good_features2]
        # print(metadata.shape, data_corrected.shape, nan_mask.shape)

        (data_corrected.T).to_csv(os.path.join(subset_dir, 'X.csv'))
        metadata_df.to_csv(os.path.join(subset_dir, 'y.csv'))
        (nan_mask.T).to_csv(os.path.join(subset_dir, 'nans.csv'))

    # %% [markdown]
    # ## Create UMAP and PCA Plots


    # %%
    # %%
    pca_file_subset = os.path.join(data_dir, f'pca_df_zscore_subset_{subset_id}.csv')

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_corrected.T)
    pca_df = pd.DataFrame(pca_result, index=data_corrected.columns, columns=['PC1', 'PC2'])
    pca_df['MV'] = subset_metadata['MV']
    pca_df['Cohort Label'] = subset_metadata['Cohort Label']
    pca_df['Study ID'] = subset_metadata['Study ID']
    pca_df['Cohort ID'] = subset_metadata['cohort_id']

    # save the pca_df
    pca_df.to_csv(pca_file_subset)

    # %%
    # create_plot(pca_df, 'Cohort Label', cohort_label_to_color)
    # plt.xlim([-100,100])
    # plt.ylim([-50,50])

    # %%
    # create_plot(pca_df, 'Study ID', study_id_to_uniq_color)
    # plt.xlim([-100,100])
    # plt.ylim([-50,50])

    # %%

    # %%
    create_plot(pca_df, 'Cohort Label', cohort_label_to_color)
    plt.savefig(os.path.join(save_dir, f'pca_cohort_label_subset_{subset_id}.png'), **savefig_params)
    plt.close()

    # error occurs here?
    create_plot(pca_df, 'Study ID', study_id_to_uniq_color)
    plt.savefig(os.path.join(save_dir, f'pca_study_id_uniq_color_subset_{subset_id}.png'), **savefig_params)
    plt.close()

    # %% [markdown]

    # %%
    # ### UMAP Plot
    try:
        umap_file_subset = os.path.join(data_dir, f'umap_df_zscore_subset_{subset_id}.csv')


        # %%
        umap_model = umap.UMAP(n_components=2)
        umap_result = umap_model.fit_transform(data_corrected.T)

        # %%
        umap_df = pd.DataFrame(umap_result, index=data_corrected.columns, columns=['UMAP1', 'UMAP2'])
        umap_df['MV'] = subset_metadata['MV']
        umap_df['Cohort Label'] = subset_metadata['Cohort Label']
        umap_df['Study ID'] = subset_metadata['Study ID']
        umap_df['Cohort ID'] = subset_metadata['cohort_id']

        # save the umap_df
        umap_df.to_csv(umap_file_subset)

        # %%
        create_plot(umap_df, 'Cohort Label', cohort_label_to_color)
        plt.savefig(os.path.join(save_dir, f'umap_cohort_label_subset_{subset_id}.png'), **savefig_params)
        plt.close()

        # %%
        create_plot(umap_df, 'Study ID', study_id_to_uniq_color)
        plt.savefig(os.path.join(save_dir, f'umap_study_id_uniq_color_subset_{subset_id}.png'), **savefig_params)
        plt.close()
    except ValueError:
        print('UMAP failed')

    # %%
    # create_plot(umap_df, 'Study ID', study_id_to_uniq_color)
    # plt.xlim([2.5,15])
    # plt.ylim([5,10])

    # %%

    return subset_dir





# %%

for peak_robust_name in ['Freq, Cohort Log Size Weighted']:
# for peak_robust_name in peak_robustness_dct.keys():
    for peak_robust_th in [0.2]:
        # subset_dir = subset_analysis(peak_robust_name=peak_robust_name, peak_robust_th=peak_robust_th)
        subset_dir = subset_analysis(peak_robust_name=peak_robust_name, peak_robust_th=peak_robust_th,rem_cohorts=['549','551','547'],recompute_robustness=True)
