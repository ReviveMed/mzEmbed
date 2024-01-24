
# %%
import os
import pandas as pd
import sys
import json
import numpy as np

sys.path.append('/Users/jonaheaton/mzlearn/')
from mspeaks import create_mspeaks_from_mzlearn_result, MSPeaks, load_mspeaks_from_pickle
from study_alignment import align_ms_studies, combine_alignments_in_dir

sys.path.append('/Users/jonaheaton/mzlearn/peak_picking_pipeline')
from utils_targeted_data import get_potential_target_matches, process_targeted_data
# from peak_picking_pipeline.utils_targeted_data import get_potential_target_matches, process_targeted_data


# %%

dropbox_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/'
output_dir = f'{dropbox_dir}/development_CohortCombination/data_2023_november/synth_norm'
os.makedirs(output_dir,exist_ok=True)
origin_name = 'ST001236_and_ST001237'


data_engine_path = '/Users/jonaheaton/Desktop/Data-engine'
freq_th = 0.4
study_id_list = []
for study_id in os.listdir(data_engine_path):
    if '.' in study_id:
        continue
    if 'SKIP' in study_id:
        continue
    if ('ST' in study_id) or ('MTBL' in study_id):
        study_id_list.append(study_id)


alignment_df = pd.read_csv(os.path.join(output_dir,'alignment_df.csv'),index_col=0)

####################
####################
# %%
origin_studies = ['ST001236','ST001237']
selected_studies = ['ST001236','ST001237','ST001932','ST001519','ST002112','ST001422','ST001428','ST002238',
                    'ST002711','ST002331','ST001849','ST001931']

selected_studies = list(set(selected_studies))

num_studies_selected = len(selected_studies)
selected_studies_dir = os.path.join(output_dir,f'selected_studies {num_studies_selected}')
os.makedirs(selected_studies_dir,exist_ok=True)
# save the selected studies to a text file
with open(os.path.join(selected_studies_dir,'selected_studies.txt'),'w') as f:
    for study_id in selected_studies:
        f.write(study_id+'\n')



# %%
new_metadata_file = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination/data_2023_october/metadata_oct13_new2.csv'
metadata_df = pd.read_csv(new_metadata_file,index_col=0)

metadata_df = metadata_df.loc[metadata_df['Study'].isin(selected_studies),:].copy()
select_num = metadata_df['Study'].nunique()
print(f'Number of studies: {select_num}')
print( metadata_df['Study'].unique())
print('missing studies:' ,np.setdiff1d(set(selected_studies),set(metadata_df['Study'].unique())))

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(metadata_df['Study'])

metadata_df['Study_num'] = le.transform(metadata_df['Study'])


# %%
# dataset_labels = '/Users/jonaheaton/Desktop/cohort_combine_oct13/dataset_labels.csv'
dataset_labels = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/Spreadsheets/pretraining datasets Q4/pretraining_dataset_labels.csv'
labels_df = pd.read_csv(dataset_labels,index_col=0)

label_col = 'label4'
for study_id in labels_df.index:
    label_id  = labels_df.loc[study_id,label_col]
    label_id = label_id.strip(' ')
    metadata_df.loc[metadata_df['Study']==study_id,label_col] = label_id


print(metadata_df[label_col].value_counts())
# check that there are no null values
print('number of nulls:', metadata_df[label_col].isnull().sum())

metadata_df[label_col].to_csv(f'{selected_studies_dir}/sample_{label_col}.csv')

# %%

other_selected_studies = [x for x in selected_studies if x not in origin_studies]

alignment_cols = [f'Compound_ID_{study_id}' for study_id in other_selected_studies]
align_df = alignment_df[alignment_cols].copy()


# %%

import matplotlib.pyplot as plt
num_studies = align_df.shape[1]
print(f'Number of studies: {num_studies}')
num_rcc4_feats = align_df.shape[0]
print(f'Number of RCC4 features: {num_rcc4_feats}')

feat_count = (~align_df.isna()).sum(axis=1)
plt.hist(feat_count)
plt.xlabel('Number of studies')
plt.ylabel('Number of features')
plt.title('Number of features detected in each study')
plt.savefig(os.path.join(selected_studies_dir,'num_features_matched_across_studies.png'))

print(f'Number of features detected in all studies: {np.sum(feat_count==num_studies)}')

thresh90 = num_studies*0.9
print(f'Number of features detected in at least 90% of the studies: {np.sum(feat_count>=thresh90)}')

thresh75 = num_studies*0.75
print(f'Number of features detected in at least 70% of the studies: {np.sum(feat_count>=thresh75)}')
# np.sum(feat_count>5)

thresh60 = num_studies*0.6
print(f'Number of features detected in at least 60% of the studies: {np.sum(feat_count>=thresh60)}')
# np.sum(feat_count>5)

thresh50 = num_studies*0.5
print(f'Number of features detected in at least 50% of the studies: {np.sum(feat_count>=thresh50)}')
# np.sum(feat_count>5)

# %%

# choose features that appear in at least 6 data sets
# chosen_feats = align_df.index[feat_count>=thresh50].tolist()
# print(f'Number of features selected: {len(chosen_feats)}')

# create the combined data set
origin_study_pkl_file = os.path.join(output_dir,f'{origin_name}.pkl')
if os.path.exists(origin_study_pkl_file):
    origin_study = load_mspeaks_from_pickle(origin_study_pkl_file)

# %% Match the features to the targets

targeted_path = f'{dropbox_dir}/Benchmarking_Data/rcc3/verified_targets.csv'
rt_tol = 20
mz_tol_matching = 0.005
targets_df = process_targeted_data(targeted_path)


for thresh in [0.5,0.6,0.7,0.8,0.9]:
    chosen_feats = align_df.index[feat_count>=num_studies*thresh].tolist()
    print(f'Number of features detected in at least {thresh*100}% of the studies: {np.sum(feat_count>=num_studies*thresh)}')

    peaks_info  = origin_study.peak_info.copy()
    peaks_info = peaks_info.loc[chosen_feats,:].copy()
    matches, num_exp_targets = get_potential_target_matches(targets_df,
                                                            peak_info=peaks_info,
                                                                rt_tol=rt_tol,
                                                                result_path=None,
                                                                mz_tol=mz_tol_matching)


    print('number of features',peaks_info.shape[0])
    print('number of potential matches to targets',matches['potential_target_id'].nunique()-1)
    # matches['potential_target_id'].unique()


# %%
feat_thresh_name = 'thresh50'
chosen_feats = align_df.index[feat_count>=num_studies*0.5].tolist()
print(f'Number of features selected: {len(chosen_feats)}')

select_feats_dir = os.path.join(selected_studies_dir,feat_thresh_name)
os.makedirs(select_feats_dir,exist_ok=True)

# %%


def min_max_scale(df):
    overall_min = df.min().min()
    overall_max = df.max().max()
    df = (df-overall_min)/(overall_max-overall_min)
    return df


combined_study = origin_study.peak_intensity.loc[chosen_feats,:].copy()
# log2 transform
combined_study = np.log2(combined_study)
# then apply the min/max scaling?

# scale to so all values between 0 and 1
# combined_study = min_max_scale(combined_study)
# %%

for study_id in study_id_list:
    print(study_id)
    for result_id in os.listdir(os.path.join(data_engine_path,study_id)):
        if '.' in result_id:
            continue
        if 'SKIP' in result_id:
            continue
        if 'meta' in result_id:
            continue
        if study_id not in selected_studies:
            continue

        result_path = os.path.join(data_engine_path,study_id,result_id)
        result_name = study_id#+'_'+result_id
        input_study_pkl_file = os.path.join(output_dir,f'{result_name}.pkl')
        if result_name == origin_name:
            continue

        if os.path.exists(input_study_pkl_file):

            input_study = load_mspeaks_from_pickle(input_study_pkl_file)
            print(input_study.sample_info)
            subset_chosen = [i for i in chosen_feats if i in input_study.peak_intensity.index]
            input_peaks = input_study.peak_intensity.loc[subset_chosen,:].copy()
            input_peaks = np.log2(input_peaks)
            # input_peaks = min_max_scale(input_peaks)
            combined_study = combined_study.join(input_peaks,how='outer')
            #fill the un-found features using the mean of each sample
            


# %%
# combined_study = np.log2(combined_study)
# fill in the features that were not found in each study with the mean of each file
common_cols = list(set(combined_study.columns).intersection(set(metadata_df.index)))
print(len(common_cols))
metadata_df = metadata_df.loc[common_cols,:].copy()
combined_study = combined_study[common_cols].copy()
combined_study.fillna(combined_study.mean(),inplace=True)

combined_study.to_csv(os.path.join(select_feats_dir,'peak_intensity.csv'),index=True)

metadata_df[label_col].to_csv(f'{select_feats_dir}/sample_{label_col}.csv')

# %%
from inmoose.pycombat import pycombat_norm
cohort_labels = metadata_df['Study_num'].to_list()
data_corrected = pycombat_norm(combined_study,cohort_labels)

data_corrected.to_csv(f'{select_feats_dir}/peak_intensity_combat.csv')
# %%
