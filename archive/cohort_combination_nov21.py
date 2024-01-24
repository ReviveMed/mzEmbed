################
# %% Preamble
################
import os
import pandas as pd
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/jonaheaton/mzlearn/')

from mspeaks import create_mspeaks_from_mzlearn_result, MSPeaks, load_mspeaks_from_pickle
from study_alignment import align_ms_studies, combine_alignments_in_dir
from peak_picking_pipeline.utils_synth_norm import synthetic_normalization_Nov2023_wrapper_repeat
# from peak_picking_pipeline.utils_norm_benchmarking import orig_pool_map_norm

sys.path.append('/Users/jonaheaton/mzlearn/peak_picking_pipeline')
from utils_norm_benchmarking import orig_pool_map_norm, compute_TIC_norm
from utils_targeted_data import get_potential_target_matches, process_targeted_data

from inmoose.pycombat import pycombat_norm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# %%
################################################################
################################################################
# User Parameters
################################################################
################################################################


# %% Set the parameters
################
use_synthetic_norm = False

# output_dir = '/Users/jonaheaton/Desktop/cohort_combine_oct16/'
dropbox_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/'
data_path = '/Users/jonaheaton/Desktop/Data-engine'
origin_name = 'ST001236_and_ST001237'
origin_result_path = os.path.join(data_path,origin_name,'rcc1_rcc3_output_v1_2023-09-28-21-22-21-')
origin_metadata_file = os.path.join(data_path,origin_name,'sample_info.csv')


cohort_combine_dir = os.path.join(dropbox_dir,'development_CohortCombination')
norm_setting_dir = os.path.join(cohort_combine_dir,'norm_settings')
output_dir = f'{cohort_combine_dir}/hilic_pos_2023_nov_21'
os.makedirs(output_dir,exist_ok=True)
plots_dir = os.path.join(output_dir,'plots')
os.makedirs(plots_dir,exist_ok=True)
os.makedirs(norm_setting_dir,exist_ok=True)


fill_na_strat = 'min'
origin_freq_th = 0.8
norm_name = 'synthetic v1'

# assert os.path.exists(os.path.join(output_dir,'norm_settings',f'{norm_name}.json')), f'Norm function {norm_name} not found'

# study_id_list = get_list_available_study_ids(data_path)
study_id_list = ['ST001200','ST001422','ST001428','ST001519','ST001849','ST001931',\
                'ST001932','ST002112','ST002238','ST002331','ST002711']

other_study_freq_th = 0.4


align_save_dir = os.path.join(output_dir,'default_alignments')
os.makedirs(align_save_dir,exist_ok=True)


# %% Choose the files used to apply the frequency threshold on the origin study

metadata = pd.read_csv(origin_metadata_file,index_col=0)
select_files = metadata[metadata['study_week'] =='baseline'].index.tolist()
train_select_files = metadata[(metadata['study_week'] =='baseline') &
                             (metadata['phase']==3) &
                             (metadata['survival class'].isin([0,1]))
                             ].index.tolist()

test_select_files = metadata[(metadata['study_week'] =='baseline') &
                             (metadata['phase']==1) &
                             (metadata['survival class'].isin([0,1]))
                             ].index.tolist()



# %% Choose the subset of studies used for analysis
origin_studies = ['ST001236','ST001237']
selected_studies_subset = ['ST001236','ST001237','ST001932','ST001519','ST002112',
                           'ST001422','ST001428','ST002238',
                        'ST002711','ST002331','ST001849','ST001931']

selected_subset_name = 'all_studies'



# %% specify the complete metadata
new_metadata_file = f'{output_dir}/metadata_oct13_new2.csv'
assert os.path.exists(new_metadata_file)

# %% specify the pretraining labels
dataset_labels = f'{dropbox_dir}/Spreadsheets/pretraining datasets Q4/pretraining_dataset_labels.csv'
labels_df = pd.read_csv(dataset_labels,index_col=0)
pretrain_label_col = 'label6'
assert pretrain_label_col in labels_df.columns

# %% specify the target data
targeted_path = f'{dropbox_dir}/Benchmarking_Data/rcc3/verified_targets.csv'
rt_tol = 20
mz_tol_matching = 0.005
targets_df = process_targeted_data(targeted_path)


# fraction of the cohorts a feature must be found to be considered for downstream analysis
num_cohorts_thresh = 0.5

# How to correct for cohort effects?
cohort_correction_method = 'combat'



# %% What is the fine-tuning task?
finetune_label_col = 'survival class'
task_name = f'{cohort_correction_method}_{finetune_label_col}'

training_files  = train_select_files
test_files = test_select_files
validation_files = []
validation_frac = 0.2
validation_rand_seed= 42


#%%

################################################################
################################################################
# Helper Functions
################################################################
################################################################



# %%
################
# Create the normalization function, read and write to json
################

def get_synthetic_norm_func(norm_func_vals,base_func=synthetic_normalization_Nov2023_wrapper_repeat):
    other_kwargs = {}
    for key,val in norm_func_vals.items():
        if key == 'norm_method_name':
            continue
        other_kwargs[key] = val

    def my_cycle_synthetic_norm(peak_intensity,peak_info,sample_info):
        norm_df,_ = base_func(peak_intensity,sample_info,peak_info,**other_kwargs)
        return norm_df
    return my_cycle_synthetic_norm


def get_synthetic_norm_func_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return get_synthetic_norm_func(data)

def save_records_to_json(save_path,record_dct,scores_dict=None):
    # save the results to json
    if scores_dict is not None:
        for key in scores_dict.keys():
            record_dct[key] = scores_dict[key]

    # assume record_dct is your dictionary
    record_dct_converted = {}
    for key, value in record_dct.items():
        if isinstance(value, np.float32):
            record_dct_converted[key] = float(value)
        elif isinstance(value, np.int64):
            record_dct_converted[key] = int(value)
        else:
            record_dct_converted[key] = value

    with open(save_path, 'w') as fp:
        json.dump(record_dct_converted, fp)

    return


def get_norm_func(norm_name,data_dir):

    os.makedirs(os.path.join(data_dir,'norm_settings'),exist_ok=True)
    norm_func_json_file = os.path.join(data_dir,'norm_settings',f'{norm_name}.json')

    if 'pool' in norm_name:
        # save_records_to_json(norm_func_json_file,{'norm_method_name':'Map Pool'})
        return orig_pool_map_norm
    elif 'raw' in norm_name:
        # save_records_to_json(norm_func_json_file,{'norm_method_name':'Raw'})
        return None
    elif 'TIC' in norm_name:
        # save_records_to_json(norm_func_json_file,{'norm_method_name':'TIC'})
        return compute_TIC_norm


    if os.path.exists(norm_func_json_file):
        norm_func = get_synthetic_norm_func_from_json(norm_func_json_file)
        return norm_func
    else:
        raise ValueError(f'Norm function {norm_name} not found')


###################
# %% Other Helper functions
###################

def get_list_available_study_ids(data_path):

    study_id_list = []
    for study_id in os.listdir(data_path):
        if '.' in study_id:
            continue
        if 'SKIP' in study_id:
            continue
        if ('ST' in study_id) or ('MTBL' in study_id):
            study_id_list.append(study_id)
    return study_id_list


def min_max_scale(df):
    overall_min = df.min().min()
    overall_max = df.max().max()
    df = (df-overall_min)/(overall_max-overall_min)
    return df

def standardize_across_cohorts(combined_intensity,cohort_labels,method):
    assert len(combined_intensity.columns) == len(cohort_labels)
    
    if method == 'combat':
        data_corrected = pycombat_norm(combined_intensity,cohort_labels)
    elif method == 'raw':
        data_corrected = combined_intensity.copy()
    elif method =='min_max':
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(cohort_labels==cohort)[0]
            cohort_data = data_corrected.iloc[:,cohort_idx].copy()
            cohort_data = min_max_scale(cohort_data)
            data_corrected.iloc[:,cohort_idx] = cohort_data

    elif method == 'zscore_0':
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(cohort_labels==cohort)[0]
            cohort_data = data_corrected.iloc[:,cohort_idx].copy()
            cohort_data = (cohort_data-cohort_data.mean(axis=0))/cohort_data.std(axis=0)
            cohort_data.fillna(0,inplace=True)
            data_corrected.iloc[:,cohort_idx] = cohort_data
    elif method == 'zscore_1':
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(cohort_labels==cohort)[0]
            cohort_data = data_corrected.iloc[:,cohort_idx].copy()
            cohort_data = (cohort_data-cohort_data.mean(axis=1))/cohort_data.std(axis=1)
            cohort_data.fillna(0,inplace=True)
            data_corrected.iloc[:,cohort_idx] = cohort_data
    else:
        raise ValueError(f'Invalid method: {method}')

    return data_corrected



# %%
################################################################
################################################################
# Main Script
################################################################
################################################################



###################
# %% import the origin study, filter and normalize
###################


origin_study_pkl_file = os.path.join(output_dir,f'{origin_name}.pkl')
if os.path.exists(origin_study_pkl_file):
    origin_study = load_mspeaks_from_pickle(origin_study_pkl_file)
    # origin_study.pca_plot_samples(save_path=os.path.join(output_dir,f'{origin_name}_PCA.png'))
    # origin_study.umap_plot_samples(save_path=os.path.join(output_dir,f'{origin_name}_umap.png'))
else:
    origin_study = create_mspeaks_from_mzlearn_result(origin_result_path)


    origin_study.apply_freq_th_on_peaks(freq_th=origin_freq_th,
                                        inplace=True,
                                        sample_subset=train_select_files)

    origin_study.apply_freq_th_on_peaks(freq_th=origin_freq_th,
                                        inplace=True,
                                        sample_subset=test_select_files)
    
    # detect and remove outliers
    # origin_study.detect_and_remove_outliers()

    # apply normalization
    norm_func = get_norm_func(norm_name,data_dir=output_dir)
    origin_study.normalize_intensity(norm_func=norm_func)

    # fill missing values
    origin_study.fill_missing_values(method=fill_na_strat)

    origin_study.add_study_id(origin_name,rename_samples=False)

    origin_study.save_to_pickle(os.path.join(output_dir,f'{origin_name}.pkl'))
    origin_study.pca_plot_samples(save_path=os.path.join(output_dir,'plots',f'{origin_name}_PCA.png'))
    origin_study.umap_plot_samples(save_path=os.path.join(output_dir,'plots',f'{origin_name}_umap.png'))

    origin_params = {
        'freq_th':origin_freq_th,
        'fill_na_strat':fill_na_strat,
        'norm_name':norm_name,
        'num samples':origin_study.get_num_samples(),
        'num peaks':origin_study.get_num_peaks(),
    }
    save_records_to_json(os.path.join(output_dir,f'{origin_name}.json'),origin_params)


###################
# %% import each input-study, filter and normalize, then align to origin
###################



for study_id in study_id_list:
    print(study_id)
    for result_id in os.listdir(os.path.join(data_path,study_id)):
        if '.' in result_id:
            continue
        if 'SKIP' in result_id:
            continue


        result_path = os.path.join(data_path,study_id,result_id)
        result_name = study_id#+'_'+result_id
        if result_name == origin_name:
            continue

        if not os.path.exists(os.path.join(output_dir,f'{result_name}.pkl')):
            new_study = create_mspeaks_from_mzlearn_result(result_path)
            if new_study.peak_intensity is None:
                print(f'No peaks found for {result_name}')
                continue
            
            # remove outliers using low frequency thresholds
            new_study.apply_freq_th_on_peaks(freq_th=0.2) # remove super low frequency peaks
            new_study.apply_freq_th_on_samples(freq_th=0.1) # remove samples with too few peaks
            
            new_study.apply_freq_th_on_peaks(freq_th=other_study_freq_th)
            
            # detect and remove outliers, not implemented yet
            # new_study.detect_and_remove_outliers()

            # apply normalization
            norm_func = get_norm_func(norm_name,data_dir=output_dir)
            new_study.normalize_intensity(norm_func=norm_func)

            new_study.fill_missing_values(method=fill_na_strat)
            new_study.add_study_id(result_name,rename_samples=False)
            new_study.save_to_pickle(os.path.join(output_dir,f'{result_name}.pkl'))
            new_study.pca_plot_samples(save_path=os.path.join(output_dir,'plots',f'{result_name}_PCA.png'))
            new_study.umap_plot_samples(save_path=os.path.join(output_dir,'plots',f'{result_name}_umap.png'))
            

            study_params = {
                'freq_th':other_study_freq_th,
                'fill_na_strat':fill_na_strat,
                'norm_name':norm_name,
                'num samples':new_study.get_num_samples(),
                'num peaks':new_study.get_num_peaks(),
            }
            save_records_to_json(os.path.join(output_dir,f'{result_name}.json'),study_params)

            align_ms_studies(origin_study,
                            new_study,
                            origin_name=origin_name,
                            input_name=result_name,
                            save_dir = align_save_dir)
            
alignment_df = combine_alignments_in_dir(align_save_dir,origin_name=origin_name)
alignment_df.to_csv(os.path.join(align_save_dir,'alignment_df.csv'),index=True)

###################
# %% rename the features in each study to correspond to the origin, removing the features that do not align to origin
###################




alignment_df = pd.read_csv(os.path.join(align_save_dir,'alignment_df.csv'),index_col=0)

for study_id in study_id_list:
    print(study_id)
    for result_id in os.listdir(os.path.join(data_path,study_id)):
        if '.' in result_id:
            continue
        if 'SKIP' in result_id:
            continue

        result_path = os.path.join(data_path,study_id,result_id)
        result_name = study_id#+'_'+result_id
        input_study_pkl_file = os.path.join(output_dir,f'{result_name}.pkl')
        renamed_study_pkl_file = os.path.join(output_dir,f'{result_name}_renamed.pkl')
        if result_name == origin_name:
            continue

        if os.path.exists(input_study_pkl_file):

            input_study = load_mspeaks_from_pickle(input_study_pkl_file)
            input_alignment = alignment_df['Compound_ID_'+result_name].copy()
            input_alignment.dropna(inplace=True)
            
            current_ids = input_alignment.values
            new_ids = input_alignment.index
            input_study.rename_selected_peaks(current_ids,new_ids)
            input_study.save_to_pickle(renamed_study_pkl_file)


###################
# %% Choose the studies for the combined study
###################

# %% Create the directory with the selected subset of studies for analysis

selected_studies = list(set(selected_studies_subset))
num_studies_selected = len(selected_studies)
selected_studies_dir = os.path.join(output_dir,f'subset {selected_subset_name}')
os.makedirs(selected_studies_dir,exist_ok=True)
# save the selected studies to a text file
with open(os.path.join(selected_studies_dir,'selected_studies.txt'),'w') as f:
    for study_id in selected_studies:
        f.write(study_id+'\n')


# %% Load the combined metadata 
metadata_df = pd.read_csv(new_metadata_file,index_col=0)

metadata_df = metadata_df.loc[metadata_df['Study'].isin(selected_studies),:].copy()
select_num = metadata_df['Study'].nunique()
print(f'Number of studies: {select_num}')
print( metadata_df['Study'].unique())
print('missing studies:' ,np.setdiff1d(set(selected_studies),set(metadata_df['Study'].unique())))

# %% Use labelencoder to assign a number to each study   
le = LabelEncoder()
le.fit(metadata_df['Study'])

metadata_df['Study_num'] = le.transform(metadata_df['Study'])

# %% assign the user created labels to each study     
for study_id in labels_df.index:
    label_id  = labels_df.loc[study_id,pretrain_label_col]
    label_id = label_id.strip(' ')
    metadata_df.loc[metadata_df['Study']==study_id,pretrain_label_col] = label_id

print(metadata_df[pretrain_label_col].value_counts())
# check that there are no null values
print('number of nulls:', metadata_df[pretrain_label_col].isnull().sum())


# %% Select the studies that are not the origin studies, then plot the overlap with the origin study 
other_selected_studies = [x for x in selected_studies if x not in origin_studies]

alignment_cols = [f'Compound_ID_{study_id}' for study_id in other_selected_studies]
align_df = alignment_df[alignment_cols].copy()


num_studies = align_df.shape[1]
print(f'Number of studies: {num_studies}')
num_origin_feats = align_df.shape[0]
print(f'Number of Origin features: {num_origin_feats}')

feat_count = (~align_df.isna()).sum(axis=1)
plt.hist(feat_count)
plt.xlabel('Number of studies')
plt.ylabel('Number of features')
plt.title('Number of features detected in each study')
plt.savefig(os.path.join(selected_studies_dir,'num_features_matched_across_studies.png'))
print(f'Number of features detected in all studies: {np.sum(feat_count==num_studies)}')

###################
# %% Target data analysis on the subset of selected features
###################

chosen_feats = align_df.index[feat_count>=num_studies*num_cohorts_thresh].tolist()
print(f'Number of features detected in at least {num_cohorts_thresh*100}% of the studies: {np.sum(feat_count>=num_studies*num_cohorts_thresh)}')

# check if origin_study is already loaded
if 'origin_study' not in locals():
    origin_study_pkl_file = os.path.join(output_dir,f'{origin_name}.pkl')
    if os.path.exists(origin_study_pkl_file):
        origin_study = load_mspeaks_from_pickle(origin_study_pkl_file)

peaks_info  = origin_study.peak_info.copy()
peaks_info = peaks_info.loc[chosen_feats,:].copy()
matches, num_exp_targets = get_potential_target_matches(targets_df,
                                                        peak_info=peaks_info,
                                                            rt_tol=rt_tol,
                                                            result_path=None,
                                                            mz_tol=mz_tol_matching)

matches.to_csv(os.path.join(selected_studies_dir,'potential_target_matches.csv'))
print('number of features',peaks_info.shape[0])
print('number of potential matches to targets',matches['potential_target_id'].nunique()-1)
    # matches['potential_target_id'].unique()

###################
# %% Create the combined study
###################

feat_thresh_name = f'num_cohorts_thresh_{num_cohorts_thresh}'
chosen_feats = align_df.index[feat_count>=num_studies*num_cohorts_thresh].tolist()
print(f'Number of features selected: {len(chosen_feats)}')

select_feats_dir = os.path.join(selected_studies_dir,feat_thresh_name)
os.makedirs(select_feats_dir,exist_ok=True)    

if os.path.exists(os.path.join(select_feats_dir,'combined_study.csv')):
    combined_study = pd.read_csv(os.path.join(select_feats_dir,'combined_study.csv'),index_col=0)
    metadata_df = pd.read_csv(os.path.join(select_feats_dir,'combined_metadata.csv'),index_col=0)
    print('combined study loaded')

else:
    combined_study = origin_study.peak_intensity.loc[chosen_feats,:].copy()
    combined_study = np.log2(combined_study)

    for study_id in other_selected_studies:
        print(study_id)
        for result_id in os.listdir(os.path.join(data_path,study_id)):
            if '.' in result_id:
                continue
            if 'SKIP' in result_id:
                continue
            if 'meta' in result_id:
                continue
            if study_id not in selected_studies:
                continue

            result_path = os.path.join(data_path,study_id,result_id)
            result_name = study_id#+'_'+result_id
            input_study_pkl_file = os.path.join(output_dir,f'{result_name}_renamed.pkl')
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

    # Verify that the sample names are the same in the combined study as they are in the combined metadata
    common_cols = list(set(combined_study.columns).intersection(set(metadata_df.index)))            
    print(len(common_cols))
    metadata_df = metadata_df.loc[common_cols,:].copy()
    combined_study = combined_study[common_cols].copy()
    combined_study.to_csv(os.path.join(select_feats_dir,'combined_study.csv'))
    metadata_df.to_csv(os.path.join(select_feats_dir,'combined_metadata.csv'))



#fill the un-found features using the mean of each sample
combined_study.fillna(combined_study.mean(),inplace=True)

cohort_labels = metadata_df['Study_num'].to_list()

# correct for cohort effects
if os.path.exists(os.path.join(select_feats_dir,f'peak_intensity_{cohort_correction_method}.csv')):
    data_corrected = pd.read_csv(os.path.join(select_feats_dir,f'peak_intensity_{cohort_correction_method}.csv'),index_col=0)
else:
    data_corrected = standardize_across_cohorts(combined_study,cohort_labels,method=cohort_correction_method)
    data_corrected.to_csv(os.path.join(select_feats_dir,f'peak_intensity_{cohort_correction_method}.csv'))

# check that there are no null values with the labels
print(metadata_df[pretrain_label_col].isnull().sum())
metadata_df[pretrain_label_col].to_csv(f'{select_feats_dir}/sample_{pretrain_label_col}.csv')


###################
# %% Create the task specific data
###################

task_dir = os.path.join(select_feats_dir,task_name)
os.makedirs(task_dir,exist_ok=True)

input_data = data_corrected.T
input_pretrain_labels = metadata_df[pretrain_label_col].copy()
input_finetune_labels = metadata_df[finetune_label_col].copy()

pretrain_files = [x for x in input_data.index if x not in training_files+test_files]

y_pretrain = input_pretrain_labels.loc[pretrain_files].copy()
y_finetune = input_finetune_labels.loc[training_files].copy()
y_test = input_finetune_labels.loc[test_files].copy()

X_pretrain = input_data.loc[pretrain_files,:].copy()
X_finetune = input_data.loc[training_files,:].copy()
X_test = input_data.loc[test_files,:].copy()


if len(validation_files) == 0:
    try:
        X_train, X_val, y_train, y_val = train_test_split(X_finetune, y_finetune, 
                                                        test_size=validation_frac, 
                                                        random_state=validation_rand_seed, 
                                                        stratify=y_finetune)
    except ValueError:
        # y_finetune is not a format that can be stratified
        X_train, X_val, y_train, y_val = train_test_split(X_finetune, y_finetune, 
                                                        test_size=validation_frac, 
                                                        random_state=validation_rand_seed, 
                                                        stratify=None)
    validation_files = X_val.index.tolist()
else:
    X_train = X_finetune.copy()
    y_train = y_finetune.copy()
    X_val = input_data.loc[:,validation_files].copy()
    y_val = input_finetune_labels.loc[validation_files].copy()

X_pretrain.to_csv(os.path.join(task_dir,'X_pretrain.csv'))
X_train.to_csv(os.path.join(task_dir,'X_train.csv'))
X_val.to_csv(os.path.join(task_dir,'X_val.csv'))
X_test.to_csv(os.path.join(task_dir,'X_test.csv'))

y_pretrain.to_csv(os.path.join(task_dir,'y_pretrain.csv'))
y_train.to_csv(os.path.join(task_dir,'y_train.csv'))
y_val.to_csv(os.path.join(task_dir,'y_val.csv'))
y_test.to_csv(os.path.join(task_dir,'y_test.csv'))


# %%