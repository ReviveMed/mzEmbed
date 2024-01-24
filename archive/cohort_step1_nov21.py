################
# %% Preamble
################
import os
import pandas as pd
import sys
import json
import numpy as np

sys.path.append('/Users/jonaheaton/mzlearn/')

from mspeaks import create_mspeaks_from_mzlearn_result, MSPeaks, load_mspeaks_from_pickle
from study_alignment import align_ms_studies, combine_alignments_in_dir
from peak_picking_pipeline.utils_synth_norm import synthetic_normalization_Nov2023_wrapper_repeat
# from peak_picking_pipeline.utils_norm_benchmarking import orig_pool_map_norm

sys.path.append('/Users/jonaheaton/mzlearn/peak_picking_pipeline')
from utils_norm_benchmarking import orig_pool_map_norm, compute_TIC_norm

#%%

################################
################################
# Helper Functions
################################
################################



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
 
    if 'pool' in norm_name:
        return orig_pool_map_norm
    elif 'raw' in norm_name:
        return 
    elif 'TIC' in norm_name:
        return compute_TIC_norm

    os.makedirs(os.path.join(data_dir,'norm_settings'),exist_ok=True)
    norm_func_json_file = os.path.join(data_dir,'norm_settings',f'{norm_name}.json')

    if os.path.exists(norm_func_json_file):
        norm_func = get_synthetic_norm_func_from_json(norm_func_json_file)
        return norm_func
    else:
        norm_func_vals = {'norm_method_name':'Cycle Repeat v0',
                                'max_cycles':25,
                                'pca_comp_var_ratio_th':0.05,
                                'strength_th':0.05,
                                'max_n_clusters':60,
                                'strength_factor_bound':0.05, #lower bound on the normalization strength effect
                                'strength_factor_alpha':0.75, #regularize of the normalization strength effect
                                'max_iter':3,
                                'interpolate_cluster_edges':True,
                                'var_reduce_alpha':0.05,
                                'use_pins_for_pca':False,
                                'large_group_size_th':9999,
                                'cluster_method':'gmm',
                                'cluster_method_recursion':'gmm',
                                'init_norm_method': 'TIC',
                                'red_dim_method':'PCA_custom',
                                'red_dim_method_recurison': 'PCA_custom',
                                'eff_batch_score_th':0.5,
                                'norm_by_cluster_method':'edge_interp',
                                'fill_na_method': 'knn',
                                'num_stages': 2,
                                'init_norm_method_2': 'NA',
                                'red_dim_method_2': 'UMAP',
                                'red_dim_method_recurison_2': 'UMAP',
                                'strength_th_2': 0.2,
                                'n_repeats': 3}
        
        save_records_to_json(norm_func_json_file,norm_func_vals)
        norm_func = get_synthetic_norm_func(norm_func_vals)
        return norm_func

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


# %%
################################
################################
# User Parameters
################################
################################


# %% Set the parameters
################
use_synthetic_norm = False

# output_dir = '/Users/jonaheaton/Desktop/cohort_combine_oct16/'
dropbox_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/'
data_path = '/Users/jonaheaton/Desktop/Data-engine'
origin_name = 'ST001236_and_ST001237'
origin_result_path = os.path.join(data_path,origin_name,'rcc1_rcc3_output_v1_2023-09-28-21-22-21-')
origin_metadata_file = os.path.join(data_path,origin_name,'sample_info.csv')


norm_name = 'synthetic v1'


output_dir = f'{dropbox_dir}/development_CohortCombination/data_2023_nov_21'
os.makedirs(output_dir,exist_ok=True)
plots_dir = os.path.join(output_dir,'plots')
os.makedirs(plots_dir,exist_ok=True)
fill_na_strat = 'min'
origin_freq_th = 0.8


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



# %%
################################
################################
# Main Script
################################
################################



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


if True:

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

