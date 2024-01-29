################
# %% Preamble
################
import os
import pandas as pd
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil


from study_alignment.align_multi import align_multiple_ms_studies, rename_inputs_to_origin
from study_alignment.utils_targets import process_targeted_data, get_potential_target_matches
from study_alignment.mspeaks import MSPeaks, create_mspeaks_from_mzlearn_result
from study_alignment.utils_misc import load_json, save_json, get_method_param_name

from inmoose.pycombat import pycombat_norm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append('/Users/jonaheaton/mzlearn/peak_picking_pipeline')
from utils_synth_norm import synthetic_normalization_Nov2023_wrapper_repeat


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
other_data_path = '/Users/jonaheaton/Desktop/Hilic_pos'
origin_name = 'ST001236_and_ST001237'
origin_result_path = os.path.join(data_path,origin_name,'rcc1_rcc3_output_v1_2023-09-28-21-22-21-')
# origin_metadata_file = os.path.join(data_path,origin_name,'sample_info.csv')


cohort_combine_dir = os.path.join(dropbox_dir,'development_CohortCombination')
norm_setting_dir = os.path.join(cohort_combine_dir,'norm_settings')
output_dir = f'{cohort_combine_dir}/hilic_pos_2024_jan_29_read_norm'
origin_metadata_file = f'{cohort_combine_dir}/rcc_sample_info3.csv'
os.makedirs(output_dir,exist_ok=True)
plots_dir = os.path.join(output_dir,'plots')
os.makedirs(plots_dir,exist_ok=True)
os.makedirs(norm_setting_dir,exist_ok=True)

create_qc_plots = False

prep_name = 'default'
fill_na_strat = 'min'
origin_freq_th = 0.8
norm_name = 'synthetic v1'
read_norm_peaks = True
other_study_freq_th = 0.4
origin_studies = ['ST001236','ST001237']
# study_id_list = get_list_available_study_ids(data_path)
study_id_list = ['ST001200','ST001422','ST001428','ST001519','ST001849','ST001931',\
                'ST001932','ST002112','ST002238','ST002331','ST002711']

full_prep_name = f'{int(100*origin_freq_th)}_{int(100*other_study_freq_th)}_{prep_name}'
cleaned_peaks_obj_dir = os.path.join(output_dir,f'cleaned_peak_objs_{full_prep_name}')
os.makedirs(cleaned_peaks_obj_dir,exist_ok=True)

prep_peaks_params ={
    'prep_name': prep_name,
    'full_prep_name':full_prep_name,
    'fill_na_strat':fill_na_strat,
    'origin_freq_th':origin_freq_th,
    'other_study_freq_th':other_study_freq_th,
    'origin_name': origin_name,
    'origin_studies':origin_studies,
    'study_id_list':study_id_list,
    'norm_name':norm_name,
    'read_norm_peaks':read_norm_peaks,
    'cleaned_peaks_obj_dir':cleaned_peaks_obj_dir,
}

# save the prep_peaks_params to json
save_json(prep_peaks_params,os.path.join(cleaned_peaks_obj_dir,'prep_peaks_params.json'))

# assert os.path.exists(os.path.join(output_dir,'norm_settings',f'{norm_name}.json')), f'Norm function {norm_name} not found'

# alignment_param_path = '/Users/jonaheaton/Desktop/alignment_analysis/Alignment_Params/Eclipse_50_50_Original.json'
# alignment_param_path = '/Users/jonaheaton/Desktop/alignment_analysis/Alignment_Params/Eclipse_50_50_default.json'
# alignment_param_path = '/Users/jonaheaton/Desktop/alignment_analysis/Alignment_Params/metabCombiner_50_50_Jan23.json'
alignment_param_path = '/Users/jonaheaton/Desktop/alignment_analysis/Alignment_Params/Merge_50_50_Jan25.json'
params = load_json(alignment_param_path)
# Ideally I would also load the frequency threshold from the alignment params
alignment_method = params['alignment_method']
method_param_name = params['method_param_name']
if 'alignment_params' in params.keys():
    alignment_params = params['alignment_params']
else:
    alignment_params = None

params['freq_th0'] = origin_freq_th
params['freq_th1'] = other_study_freq_th
params['prep peaks'] = prep_peaks_params


if method_param_name == 'default':
    alignment_dir_name = f'{alignment_method}_align_{full_prep_name}'
else:
    alignment_dir_name = f'{alignment_method}_{method_param_name}_align_{full_prep_name}'
# save the alignment params to the output directory

align_save_dir = os.path.join(output_dir,alignment_dir_name)
os.makedirs(align_save_dir,exist_ok=True)
save_json(params,os.path.join(align_save_dir,'params.json'))

# %% Choose the files used to apply the frequency threshold on the origin study
origin_metadata = pd.read_csv(origin_metadata_file,index_col=0)
# rename the index of metadata to include the study name
origin_metadata.index = [f'{origin_name}_{y}' for y in origin_metadata.index]


select_files = origin_metadata[origin_metadata['study_week'] =='baseline'].index.tolist()
train_select_files = origin_metadata[(origin_metadata['study_week'] =='baseline') &
                             (origin_metadata['phase']==3) &
                            (origin_metadata['Treatment']=='NIVOLUMAB') &
                             (origin_metadata['Benefit'].isin(['CB','NCB']))
                             ].index.tolist()

test_select_files = origin_metadata[(origin_metadata['study_week'] =='baseline') &
                             (origin_metadata['phase']==1) &
                             (origin_metadata['Treatment']=='NIVOLUMAB') &
                             (origin_metadata['Dose (mg/kg)']>1) &
                             (origin_metadata['Benefit'].isin(['CB','NCB']))
                             ].index.tolist()

select_files = train_select_files+test_select_files

save_json(select_files,os.path.join(align_save_dir,'select_origin_files.json'))

# %% Choose the subset of studies used for analysis
initial_selected_studies_subset = ['ST001236','ST001237','ST001932','ST001519','ST002112',
                           'ST001422','ST001428','ST002238','ST002711','ST002331','ST001849','ST001931']

# study_list = selected_studies_subset

align_score_th = 0.25
selected_subset_name = f'all_studies with align score {align_score_th}'


selected_subset_name = selected_subset_name + ' from ' + alignment_dir_name



# %% specify the complete metadata
new_metadata_file = f'{cohort_combine_dir}/metadata_oct13_new2.csv'
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
# finetune_label_col = 'survival class'
finetune_label_col = 'Benefit'
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
        # return orig_pool_map_norm
        raise NotImplementedError
    elif 'raw' in norm_name:
        # save_records_to_json(norm_func_json_file,{'norm_method_name':'Raw'})
        return None
    elif 'TIC' in norm_name:
        # save_records_to_json(norm_func_json_file,{'norm_method_name':'TIC'})
        # return compute_TIC_norm
        raise NotImplementedError


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


origin_study_pkl_file = os.path.join(output_dir,f'{origin_name}.pkl')
clean_origin_study_pkl_file = os.path.join(cleaned_peaks_obj_dir,f'{origin_name}_cleaned.pkl')
alignment_df_path = os.path.join(align_save_dir,'alignment_df.csv')

# for some reason the norm function returns all NaNs?
# I think this occurs if a reasonable frequency threshold is not applied (~10%?)

if (not read_norm_peaks):
    norm_func = get_norm_func(norm_name,norm_setting_dir)
else:
    norm_func = None
# if select_files[0] not in 
# select_files = [f'{origin_name}_{x}' for x in select_files]
if not os.path.exists(alignment_df_path):

    if os.path.exists(origin_study_pkl_file):
        origin_peak_obj_path = origin_study_pkl_file
    else:
        # origin_peak_obj_path = origin_study_pkl_file
        if read_norm_peaks:
            origin_study = create_mspeaks_from_mzlearn_result(origin_result_path,peak_intensity='intensity_max_synthetic_norm')
            if origin_study.peak_intensity is None:
                raise ValueError('origin study peak intensity is None')
        else:
            origin_study = create_mspeaks_from_mzlearn_result(origin_result_path)
        origin_study.add_study_id(origin_name)
        origin_study.apply_freq_th_on_peaks(0.1)
        if (not read_norm_peaks):
            origin_study.normalize_intensity(norm_func)
        origin_study.save(origin_study_pkl_file)
        origin_peak_obj_path = origin_study_pkl_file

    input_peak_obj_path_list = []
    input_name_list = []

    for study_id in study_id_list:
        for result_id in os.listdir(os.path.join(data_path,study_id)):
            if '.' in result_id:
                continue
            if 'SKIP' in result_id:
                continue
            if 'meta' in result_id:
                continue
            if 'target' in result_id:
                continue
            
            result_path = os.path.join(data_path,study_id,result_id)
            result_name = study_id#+'_'+result_id
            if result_name == origin_name:
                continue

            print(result_name)
            # if os.path.exists(os.path.join(other_data_path,study_id,'final_peaks','intensity_max_synthetic_norm.csv')):
            #     # copy the file over
            #     new_path = os.path.join(data_path,study_id,result_id,'final_peaks','intensity_max_synthetic_norm.csv')
            #     os.makedirs(os.path.dirname(new_path),exist_ok=True)
            #     shutil.copy(os.path.join(other_data_path,study_id,'final_peaks','intensity_max_synthetic_norm.csv'),new_path)

                
            input_study_pkl_file = os.path.join(output_dir,f'{result_name}.pkl')
            if not os.path.exists(input_study_pkl_file):
                # input_study_pkl_file = result_path
                # print('OOPs')
                if read_norm_peaks:
                    new_study = create_mspeaks_from_mzlearn_result(result_path,peak_intensity='intensity_max_synthetic_norm')
                    if new_study.peak_intensity is None:
                        raise ValueError(f'new study ({result_name}) peak intensity is None')
                else:
                    new_study = create_mspeaks_from_mzlearn_result(result_path)
                new_study.add_study_id(study_id)
                new_study.apply_freq_th_on_peaks(0.2)
                if (not read_norm_peaks):
                    new_study.normalize_intensity(norm_func)
                new_study.save(input_study_pkl_file)

            input_peak_obj_path_list.append(input_study_pkl_file)
            input_name_list.append(study_id)



    # norm_func = get_norm_func(norm_name,output_dir)

    align_multiple_ms_studies(origin_peak_obj_path= origin_peak_obj_path,
                            input_peak_obj_path_list=input_peak_obj_path_list,
                            save_dir=align_save_dir,
                                origin_name=origin_name,
                                input_name_list=input_name_list,
                                alignment_method = alignment_method,
                                alignment_params = alignment_params,
                                origin_freq_th = origin_freq_th,
                                input_freq_th = other_study_freq_th,
                                origin_select_files = select_files,
                                fill_na_strategy = fill_na_strat,
                                outlier_samples_removal_strategy = 'low_frequency',
                                norm_func = None,
                                cleaned_peaks_obj_dir=cleaned_peaks_obj_dir)


# %% Quality control on the cleaned studies
if create_qc_plots:
    print('creating QC plots')

    for study_id in study_id_list+[origin_name]:
        cleaned_study_peak_obj = os.path.join(cleaned_peaks_obj_dir,f'{study_id}_cleaned.pkl')
        cleaned_study_peak_obj_plot = os.path.join(cleaned_peaks_obj_dir,f'{study_id}_umap_samples.png')
        if os.path.exists(cleaned_study_peak_obj):
            if os.path.exists(cleaned_study_peak_obj_plot):
                continue
            
            print('plotting QC for ',study_id)
            study_obj = MSPeaks()
            study_obj.load(cleaned_study_peak_obj)
            study_obj.heirarchical_clustering_plot(os.path.join(cleaned_peaks_obj_dir,f'{study_id}_heirarchical_clustering.png'))
            # study_obj.pca_plot_peaks(os.path.join(cleaned_peaks_obj_dir,f'{study_id}_pca_peaks.png'))
            study_obj.pca_plot_samples(os.path.join(cleaned_peaks_obj_dir,f'{study_id}_pca_samples.png'))
            # study_obj.umap_plot_peaks(os.path.join(cleaned_peaks_obj_dir,f'{study_id}_umap_peaks.png'))
            study_obj.umap_plot_samples(os.path.join(cleaned_peaks_obj_dir,f'{study_id}_umap_samples.png'))




# %%
alignment_df = pd.read_csv(alignment_df_path,index_col=0)
alignment_scores = pd.read_csv(os.path.join(align_save_dir,'align_score_df.csv'),index_col=0)
input_name_list = alignment_df.columns.tolist()
clean_input_path_list_txt_file = os.path.join(align_save_dir,'clean_input_path_list.txt')
# read the paths from the file
# with open(clean_input_path_list_txt_file) as f:
#     clean_input_peak_obj_path_list = f.read().splitlines()

paths_to_renamed_studies = rename_inputs_to_origin(
                                        align_save_dir,
                                        multi_alignment_df=alignment_df,
                                        load_dir=cleaned_peaks_obj_dir)



###################
# %% Choose the studies for the combined study
###################

# %% Choose the studies subset based on the alginment scores
print('alignment score subset')
selected_studies_subset = []
initial_selected_studies_subset = initial_selected_studies_subset+origin_studies
for study_id in initial_selected_studies_subset:
    if study_id == origin_name:
        selected_studies_subset.append(study_id)
        continue
    if study_id in origin_studies:
        selected_studies_subset.append(study_id)
        continue
    if 'align_score' in alignment_scores.columns:
        # old way of doing things
        if alignment_scores.loc[study_id,'align_score'] >= align_score_th:
            selected_studies_subset.append(study_id)
    else:
        #TODO update the method to include the origin study
        if alignment_scores.loc[study_id,origin_name] >= align_score_th:
            selected_studies_subset.append(study_id)



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
if 'Result_ID' not in metadata_df.columns:
    metadata_df['Result_ID'] = metadata_df['Study']
    origin_files = metadata_df[metadata_df['Study'].isin(origin_studies)].index.tolist()
    metadata_df.loc[origin_files,'Result_ID'] = origin_name

# rename the index of metadata to include the study name
metadata_df.index = [f'{x}_{y}' for x,y in zip(metadata_df['Result_ID'],metadata_df.index)]

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

# alignment_cols = [f'Compound_ID_{study_id}' for study_id in other_selected_studies]
alignment_cols = [f'{study_id}' for study_id in other_selected_studies]

align_df = alignment_df[alignment_cols].copy()
align_df.to_csv(os.path.join(selected_studies_dir,'align_df_subset.csv'))

num_studies = align_df.shape[1]
num_studies_incl_origin = num_studies+len(origin_studies)
print(f'Number of studies: {num_studies}')
num_origin_feats = align_df.shape[0]
print(f'Number of Origin features: {num_origin_feats}')

basic_align_stats = {}
basic_align_stats['num_studies'] = num_studies
basic_align_stats['num_origin_feats'] = num_origin_feats
basic_align_stats['num_studies_incl_origin'] = num_studies_incl_origin

study_rankings = align_df.notna().sum()
study_rankings.sort_values(inplace=True,ascending=False)
# set the figure size
plt.figure(figsize=(4,4))
plt.bar(study_rankings.index,study_rankings.values)
plt.xticks(rotation=90)
plt.xlabel('Study')
plt.ylabel('Number of features')
plt.title('Number of features detected in each study')
plt.savefig(os.path.join(selected_studies_dir,'num_features_in_each_study.png'),bbox_inches='tight')

if align_df.shape[1] > 1:
    basic_align_stats['best matched study'] = study_rankings.index[0]
    basic_align_stats['best study feature-count'] = study_rankings.iloc[0]
    basic_align_stats['worst matched study'] = study_rankings.index[-1]
    basic_align_stats['worst study feature-count'] = study_rankings.iloc[-1]

save_json(basic_align_stats,os.path.join(selected_studies_dir,'align_stats.json'))

feat_count = (align_df.notna()).sum(axis=1)
plt.figure(figsize=(4,4))
plt.hist(feat_count)
plt.xlabel('Number of studies')
plt.ylabel('Number of features')
plt.title('Number of features found in how many studies')
plt.savefig(os.path.join(selected_studies_dir,'num_features_by_number_of_studies.png'),bbox_inches='tight')
print(f'Number of features detected in all studies: {np.sum(feat_count==num_studies)}')

###################
# %% Target data analysis on the subset of selected features
###################

feat_thresh_name = f'num_cohorts_thresh_{num_cohorts_thresh}'
chosen_feats = align_df.index[feat_count>=num_studies*num_cohorts_thresh].tolist()
print(f'Number of features detected in at least {num_cohorts_thresh*100}% of the studies: {np.sum(feat_count>=num_studies*num_cohorts_thresh)}')

select_feats_dir = os.path.join(selected_studies_dir,feat_thresh_name)
os.makedirs(select_feats_dir,exist_ok=True)    

# check if origin_study is already loaded
if 'origin_study' not in locals():
    if os.path.exists(clean_origin_study_pkl_file):
        origin_study = MSPeaks()
        origin_study.load(origin_study_pkl_file)

peaks_info  = origin_study.peak_info.copy()
peaks_info = peaks_info.loc[chosen_feats,:].copy()
matches, num_exp_targets = get_potential_target_matches(targets_df,
                                                        peak_info=peaks_info,
                                                            rt_tol=rt_tol,
                                                            result_path=None,
                                                            mz_tol=mz_tol_matching)

matches.to_csv(os.path.join(select_feats_dir,'potential_target_matches.csv'))
print('number of features',peaks_info.shape[0])
print('number of potential matches to targets',matches['potential_target_id'].nunique()-1)

# save the number of potential matches to a text file
with open(os.path.join(select_feats_dir,'num_potential_target_matches.txt'),'w') as f:
    f.write(f'number of studies: {num_studies_incl_origin}\n')
    f.write(f'number of studies not including origin: {num_studies}\n')
    f.write(f'number of potential origin features: {num_origin_feats}\n')
    f.write(f'number of chosen features: {peaks_info.shape[0]}\n')
    f.write(f'number of potential matches to targets: {matches["potential_target_id"].nunique()-1}\n')

    # matches['potential_target_id'].unique()

# save the subset of the align_df
align_df.loc[chosen_feats,:].to_csv(os.path.join(select_feats_dir,'chosen_aligned_peaks.csv'))


###################
# %% Create the combined study
###################

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
            input_study_pkl_file = os.path.join(align_save_dir,f'{result_name}_renamed.pkl')
            if result_name == origin_name:
                continue

            if os.path.exists(input_study_pkl_file):

                input_study = MSPeaks()
                input_study.load(input_study_pkl_file)
                # print(input_study.sample_info)
                subset_chosen = [i for i in chosen_feats if i in input_study.peak_intensity.index]
                input_peaks = input_study.peak_intensity.loc[subset_chosen,:].copy()
                input_peaks = np.log2(input_peaks)
                # input_peaks = min_max_scale(input_peaks)
                combined_study = combined_study.join(input_peaks,how='outer')

    # Verify that the sample names are the same in the combined study as they are in the combined metadata
    common_cols = list(set(combined_study.columns).intersection(set(metadata_df.index)))            
    # print(len(common_cols))
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

y_test_path = os.path.join(task_dir,'y_test.csv')

if not os.path.exists(y_test_path):

    input_data = data_corrected.T
    if finetune_label_col not in metadata_df.columns:
        print(f'warning: {finetune_label_col} not in metadata_df.columns')
        if finetune_label_col in origin_metadata.columns:
            common_index = list(set(metadata_df.index).intersection(set(origin_metadata.index)))
            metadata_df[finetune_label_col] = None
            metadata_df.loc[common_index,finetune_label_col] = origin_metadata.loc[common_index,finetune_label_col]
            print(f'found {finetune_label_col} in origin_metadata, adding to metadata_df on common index (N={len(common_index)}')
        else:
            print(f'warning: {finetune_label_col} also not in origin_metadata.columns')
            raise ValueError(f'finetune_label_col {finetune_label_col} not in metadata_df.columns')

    # if training_files[0] not in metadata_df.index:
    #     print('warning: training_files[0] not in metadata_df.index')
    #     training_files = [f'{origin_name}_{x}' for x in training_files]
    #     test_files = [f'{origin_name}_{x}' for x in test_files]
        # validation_files = [f'{origin_name}_{x}' for x in validation_files]

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
else:
    print('task data already exists with the data')

# %%