
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
fill_na_strat = 'min'
origin_freq_th = 0.8
norm_name = 'synthetic v1'

output_dir = f'{cohort_combine_dir}/normalization/{origin_name}/{norm_name}'
os.makedirs(output_dir,exist_ok=True)
plots_dir = os.path.join(output_dir,'plots')
os.makedirs(plots_dir,exist_ok=True)
os.makedirs(norm_setting_dir,exist_ok=True)




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

origin_study = create_mspeaks_from_mzlearn_result(origin_result_path)


origin_study.apply_freq_th_on_peaks(freq_th=origin_freq_th,
                                    inplace=True)

# detect and remove outliers
# origin_study.detect_and_remove_outliers()

# apply normalization
norm_func = get_norm_func(norm_name,data_dir=output_dir)
origin_study.normalize_intensity(norm_func=norm_func)
origin_study.peak_intensity.to_csv(os.path.join(output_dir,f'{origin_name}_normalized.csv'))

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
