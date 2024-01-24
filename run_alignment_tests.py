# %%
###################
## Preamble
###################
import os
import pandas as pd
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import pickle

from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(dir_path, '..'))

from study_alignment.utils_targets import link_targets_to_peaks, get_potential_target_matches, process_targeted_data
# from study_alignment import mspeaks
from study_alignment.utils_eclipse import align_ms_studies_with_Eclipse
from study_alignment.utils_metabCombiner import align_ms_studies_with_metabCombiner, create_metaCombiner_grid_search
from study_alignment.mspeaks import create_mspeaks_from_mzlearn_result, MSPeaks

# from study_alignment import mspeaks as myms
from met_matching.metabolite_name_matching_main import refmet_query

# import local modules help
# https://fortierq.github.io/python-import/

# %%
###################
## Helper Functions
###################


# def load_mspeaks_from_pickle(filename):
#     with open(filename, 'rb') as f:
#         return pickle.load(f)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    # convert type int64 to int
    for k,v in data.items():
        if isinstance(v, np.int64):
            data[k] = int(v)
    with open(file_path, 'w') as f:
        json.dump(data, f)

def unravel_dict(d, prefix='a'):
    unravel = {}
    for k, v in d.items():
        unravel[prefix + '_' + k] = v
    for k, v in list(unravel.items()):  # Create a copy of the items
        if isinstance(v, dict):
            unravel.update(unravel_dict(v, k))
            del unravel[k] 
    return unravel


def peaks_to_targets_wrapper(peak_info,targets_df,rt_tol,mz_tol=0.005):
    matches_1, true_pos_perct = get_potential_target_matches(targets_df,
                    peak_info=peak_info,
                    rt_tol=rt_tol,
                    result_path=None,
                    mz_tol=mz_tol)


    matches_1,true_pos_perct, _ = link_targets_to_peaks(
        peaks_df = matches_1, 
        targets_df=targets_df, 
        rt_tol=rt_tol, 
        mz_tol=mz_tol)
    return matches_1

# %%
###################
### Cleaning up the targeted data
###################
def clean_targets(targets_df):

    refmet_names = []
    for target in targets_df.index:
        q = refmet_query(target)
        if 'refmet_name' in q:
            refmet_names.append(q['refmet_name'])
        else:
            print(target)
            # print(q)
            refmet_names.append('')
        
    targets_df['refmet_name'] = refmet_names
    return targets_df


def clean_targeted_data(target_data_path):

    if os.path.exists(target_data_path):
        targets = process_targeted_data(target_data_path)
    else:
        print("Target data not found")
        return None
    
    if 'refmet name' in targets.columns:
        targets.rename(columns={'refmet name': 'refmet_name'}, inplace=True)

    if (not 'refmet_name' in targets.columns):
        target_data_path_clean = target_data_path.replace('.csv', '_clean.csv')
        if os.path.exists(target_data_path_clean):
            targets = pd.read_csv(target_data_path_clean, index_col=0)
        else:
            targets_clean = clean_targets(targets)
            targets_clean.to_csv(target_data_path_clean)
            targets = targets_clean
    
    num_targets_start = targets.shape[0]
    # drop the targets that don't have a refmet name
    targets = targets[targets['refmet_name'] != '']
    targets = targets[targets['refmet_name'] != '-']
    # targets.reset_index(inplace=True)
    targets.set_index('refmet_name', inplace=True)
    # remove duplicates
    targets = targets[~targets.index.duplicated(keep='first')]
    num_targets_end = targets.shape[0]

    print(f'number of targets dropped during cleaning: {num_targets_start - num_targets_end}')
    
    return targets

# %%
###################
### Initialize the Study comparison
###################

def initialize_study_comparison(targets_path0, targets_path1, peaks_obj_path0, peaks_obj_path1, name0, name1, output_dir, **kwargs):

    mz_tol = kwargs.get('mz_tol', 0.005)
    yes_plot = kwargs.get('yes_plot', False)

    save_dir = os.path.join(output_dir, f'{name0}_vs_{name1}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_info_json_file = os.path.join(save_dir, 'input_info.json')

    if (os.path.exists(input_info_json_file)) and (not yes_plot):
        print('input info already exists')
        input_info = load_json(input_info_json_file)
        # overwrite the comparison id
        input_info['comparison_id'] = f'{name0}_vs_{name1}'
        save_json(input_info, input_info_json_file)
        return input_info

    # process the targeted data
    targets0 = clean_targeted_data(targets_path0)
    targets1 = clean_targeted_data(targets_path1)

    num_targets0 = targets0.shape[0]
    num_targets1 = targets1.shape[0]


    targets_common_savepath = os.path.join(save_dir, f'{name0}_{name1}_targets_common.csv')

    if os.path.exists(targets_common_savepath):
        targets_common = pd.read_csv(targets_common_savepath, index_col=0)
        id0_list = targets_common.index.to_list()
        id1_list = id0_list 
        targets0_matched = targets0.loc[id0_list][['rtmed','mzmed']]
        targets1_matched = targets1.loc[id1_list][['rtmed','mzmed']]

    else:
        # find the common targets by name
        matching_targets = []
        for id0 in targets0.index:
            for id1 in targets1.index:
                if id0 == id1:
                    id0_name = id0
                    matching_targets.append((id0, id1, id0_name))
                    break

        id0_list = [x[0] for x in matching_targets]
        id1_list = [x[1] for x in matching_targets]

        targets0_matched = targets0.loc[id0_list][['rtmed','mzmed']]
        targets1_matched = targets1.loc[id1_list][['rtmed','mzmed']]
        # targets1_matched.index = id0_list
        
        targets_common = targets0_matched.join(targets1_matched, lsuffix=f'_{name0}', rsuffix=f'_{name1}')


        # remove outliers as judged by mz difference
        targets_common_mz_diff = targets_common[f'mzmed_{name0}'] - targets_common[f'mzmed_{name1}']
        targets_common_mz_diff = targets_common_mz_diff.abs()
        num_outliers = targets_common_mz_diff[targets_common_mz_diff > 0.01].shape[0]
        print(f'number of common target outliers: {num_outliers}')
        targets_common_mz_diff = targets_common_mz_diff[targets_common_mz_diff < 0.01]
        targets_common = targets_common.loc[targets_common_mz_diff.index]
        num_remaining = targets_common.shape[0]
        print(f'number of common targets remaining: {num_remaining}')


        targets_common = targets_common[[f'rtmed_{name0}', f'rtmed_{name1}', f'mzmed_{name0}', f'mzmed_{name1}']]
        targets_common.to_csv(targets_common_savepath)

    targets_common_0 = targets_common[[f'rtmed_{name0}', f'mzmed_{name0}']].copy()
    targets_common_0.columns = ['rtmed', 'mzmed']

    targets_common_1 = targets_common[[f'rtmed_{name1}', f'mzmed_{name1}']].copy()
    targets_common_1.columns = ['rtmed', 'mzmed']


    if yes_plot:

        fig, ax = plt.subplots(1,2, figsize=(10,5))
        mz_diff_targets = targets_common[f'mzmed_{name0}'] - targets_common[f'mzmed_{name1}']
        mz_diff_targets.plot.hist(ax=ax[0], bins=10)
        ax[0].set_title(f'MZ ({name0}) - MZ ({name1})')
        ax[0].set_xlabel('mz difference')
        ax[0].set_ylabel('count')

        rt_diff_targets = targets_common[f'rtmed_{name0}'] - targets_common[f'rtmed_{name1}']
        rt_diff_targets.plot.hist(ax=ax[1], bins=10)
        ax[1].set_title(f'RT ({name0}) - RT ({name1})')
        ax[1].set_xlabel('rt difference')
        ax[1].set_ylabel('count')
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f'{name0}_{name1}_target_differences.png'))
        plt.close()


            # load the peaks object
    peaks_obj = MSPeaks()
    peaks_obj.load_from_pickle(peaks_obj_path0)
    # peaks_obj.load_from_json(peaks_obj_path0)
    # peaks_obj = load_mspeaks_from_pickle(peaks_obj_path0)
    peaks_obj.apply_freq_th_on_peaks(freq_th=0.1)
    peak_info = peaks_obj.peak_info
    num_peaks_0 = peak_info.shape[0]
    freq_peaks_0 = peak_info['freq'].mean()
    rt_peak_max_0 = peak_info['rtmed'].max()
    rt_peak_min_0 = peak_info['rtmed'].min()
    mz_peak_max_0 = peak_info['mzmed'].max()
    mz_peak_min_0 = peak_info['mzmed'].min()
    peak_info['log_int'] = np.log10(peaks_obj.peak_intensity.median())
    num_files_0 = peaks_obj.get_num_samples()

    # match the peaks to the targets
    rt_tol_matching =0.02*rt_peak_max_0

    target_peak_matches1_path = os.path.join(save_dir, f'{name0}_target_peak_matches.csv')
    if os.path.exists(target_peak_matches1_path):
        matches_0 = pd.read_csv(target_peak_matches1_path, index_col=0)
        targets_common_0_peak_matched = matches_0.loc[matches_0['target_matched'],'target_name']
        target_common_0_names = targets_common_0_peak_matched.to_list()
        target_common_0_fts = targets_common_0_peak_matched.index
        targets_common[f'FT_matched_{name0}'] = 'NA'
        targets_common.loc[target_common_0_names, f'FT_matched_{name0}'] = target_common_0_fts

    else:
        matches_0, true_pos_perct = get_potential_target_matches(targets_common_0,
                            peak_info=peak_info,
                            rt_tol=rt_tol_matching,
                            result_path=None,
                            mz_tol=mz_tol)


        matches_0,true_pos_perct, _ = link_targets_to_peaks(
            peaks_df = matches_0, 
            targets_df=targets_common_0, 
            rt_tol=rt_tol_matching, 
            mz_tol=mz_tol)

        
        # matches_0.to_csv(os.path.join(save_dir, f'{name0}_target_peak_matches.csv'))
        matches_0.to_csv(target_peak_matches1_path)
        targets_common_0_peak_matched = matches_0.loc[matches_0['target_matched'],'target_name']
        target_common_0_names = targets_common_0_peak_matched.to_list()
        target_common_0_fts = targets_common_0_peak_matched.index
        targets_common[f'FT_matched_{name0}'] = 'NA'
        targets_common.loc[target_common_0_names, f'FT_matched_{name0}'] = target_common_0_fts


    # peaks_obj = load_mspeaks_from_pickle(peaks_obj_path1)
    peaks_obj = MSPeaks()
    peaks_obj.load_from_pickle(peaks_obj_path1)
    # peaks_obj.load_from_json(peaks_obj_path1)
    peaks_obj.apply_freq_th_on_peaks(freq_th=0.1)
    peak_info = peaks_obj.peak_info
    num_peaks_1 = peak_info.shape[0]
    freq_peaks_1 = peak_info['freq'].mean()
    rt_peak_max_1 = peak_info['rtmed'].max()
    rt_peak_min_1 = peak_info['rtmed'].min()
    mz_peak_max_1 = peak_info['mzmed'].max()
    mz_peak_min_1 = peak_info['mzmed'].min()
    peak_info['log_int'] = np.log10(peaks_obj.peak_intensity.median())
    num_files_1 = peaks_obj.get_num_samples()

    rt_tol_matching =0.02*rt_peak_max_1

    target_peak_matches1_path = os.path.join(save_dir, f'{name1}_target_peak_matches.csv')
    if os.path.exists(target_peak_matches1_path):
        matches_1 = pd.read_csv(target_peak_matches1_path, index_col=0)
        targets_common_1_peak_matched = matches_1.loc[matches_1['target_matched'],'target_name']
        target_common_1_names = targets_common_1_peak_matched.to_list()
        target_common_1_fts = targets_common_1_peak_matched.index
        targets_common[f'FT_matched_{name1}'] = 'NA'
        targets_common.loc[target_common_1_names, f'FT_matched_{name1}'] = target_common_1_fts

    else:

        matches_1, true_pos_perct = get_potential_target_matches(targets_common_1,
                            peak_info=peak_info,
                            rt_tol=rt_tol_matching,
                            result_path=None,
                            mz_tol=mz_tol)


        matches_1,true_pos_perct, _ = link_targets_to_peaks(
            peaks_df = matches_1, 
            targets_df=targets_common_1, 
            rt_tol=rt_tol_matching, 
            mz_tol=mz_tol)

        # matches_1.to_csv(os.path.join(save_dir, f'{name1}_target_peak_matches.csv'))
        matches_1.to_csv(target_peak_matches1_path)
        targets_common_1_peak_matched = matches_1.loc[matches_1['target_matched'],'target_name']
        target_common_1_names = targets_common_1_peak_matched.to_list()
        target_common_1_fts = targets_common_1_peak_matched.index
        targets_common[f'FT_matched_{name1}'] = 'NA'
        targets_common.loc[target_common_1_names, f'FT_matched_{name1}'] = target_common_1_fts


    targets_common_dict = targets_common.to_dict()
    num_targets_common_both_matched =  len(set(target_common_0_names).intersection(set(target_common_1_names)))
    print(f'number of common targets matched in both studies: {num_targets_common_both_matched}')


    ########## Make a plot of the peaks and targets (found and not found)
    if yes_plot:
        max_mz = max(mz_peak_max_0, mz_peak_max_1) + 25
        min_mz = min(mz_peak_min_0, mz_peak_min_1) - 25
        which_mz_top = np.argmax([mz_peak_max_0, mz_peak_max_1])

        # should we plot the peaks with color by their average intensity? and size by frequency?

        fig, ax = plt.subplots(1,2, figsize=(12,6))
        # use the reverse copper colormap to make the low frequency peaks fade into the background
        ax[0].scatter(matches_0['rtmed'], matches_0['mzmed'], c=matches_0['freq'], label='peaks', 
                      marker = 'x', cmap='copper_r',s=(19*matches_0['freq']+1))
        # plot the captured targets
        ax[0].scatter(targets_common_0.loc[target_common_0_names,'rtmed'], 
                      targets_common_0.loc[target_common_0_names,'mzmed'], color='green', 
                      label='common capt targets', marker = 'o')
        # plot the missed targets
        ax[0].scatter(targets_common_0.loc[~targets_common_0.index.isin(target_common_0_names),'rtmed'],
                        targets_common_0.loc[~targets_common_0.index.isin(target_common_0_names),'mzmed'],
                        color='red', label='common mis targets', marker = 'o')
        # plot the other targets
        ax[0].scatter(targets0.loc[~targets0.index.isin(id0_list),'rtmed'],
                        targets0.loc[~targets0.index.isin(id0_list),'mzmed'],
                        color='cyan', label='other targets', marker = 'o',s=5,alpha=0.5)

        ax[0].set_title(f'{name0} Peaks and Targets')
        ax[0].set_xlabel('RT (sec)')
        ax[0].set_ylabel('MZ')
        ax[0].set_ylim([min_mz, max_mz])
        # add a colorbar
        cbar = plt.colorbar(ax[0].collections[0], ax=ax[0], label='Peak Frequency')
        if which_mz_top == 1:
            ax[0].legend()

        # use the reverse copper colormap to make the low frequency peaks fade into the background
        ax[1].scatter(matches_1['rtmed'], matches_1['mzmed'], c=matches_1['freq'], label='peaks', 
                      marker = 'x', cmap='copper_r',s=(19*matches_1['freq']+1))
        # plot the captured targets
        ax[1].scatter(targets_common_1.loc[target_common_1_names,'rtmed'],
                        targets_common_1.loc[target_common_1_names,'mzmed'], color='green',
                        label='common capt targets', marker = 'o')
        # plot the missed targets
        ax[1].scatter(targets_common_1.loc[~targets_common_1.index.isin(target_common_1_names),'rtmed'],
                        targets_common_1.loc[~targets_common_1.index.isin(target_common_1_names),'mzmed'],
                        color='red', label='common mis targets', marker = 'o')
        
        # plot the other targets
        ax[1].scatter(targets1.loc[~targets1.index.isin(id1_list),'rtmed'],
                        targets1.loc[~targets1.index.isin(id1_list),'mzmed'],
                        color='cyan', label='other targets', marker = 'o',s=5,alpha=0.5)

        ax[1].set_title(f'{name1} Peaks and Targets')
        ax[1].set_xlabel('RT (sec)')
        ax[1].set_ylabel('MZ')
        ax[1].set_ylim([min_mz, max_mz])
        # add a colorbar
        cbar = plt.colorbar(ax[1].collections[0], ax=ax[1], label='Peak Frequency')
        if which_mz_top == 0:
            ax[1].legend()


        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{name0}_{name1}_peaks_targets.png'))
        plt.close()

    if ('_pos' in name0):
        polarity = 'pos'
    elif '_neg' in name0:
        polarity = 'neg'
    if ('hilic_' in name0):
        ms_type = 'hilic'
    elif ('rp_' in name0):
        ms_type = 'rp'

    input_info = {
        'comparison_id': f'{name0}_vs_{name1}',
        'name0': name0,
        'name1': name1,
        'polarity': polarity,
        'ms_type': ms_type,
        'num_files0': num_files_0,
        'num_files1': num_files_1,
        'num_targets0': num_targets0,
        'num_targets1': num_targets1,
        'num_targets_common': targets_common.shape[0],
        'num_targets_common_matched0': len(target_common_0_names),
        'num_targets_common_matched1': len(target_common_1_names),
        'num_targets_common_both_matched': num_targets_common_both_matched,
        'targets_common': targets_common_dict,
        'peaks0_path': peaks_obj_path0,
        'peaks1_path': peaks_obj_path1,
        'save_dir': save_dir,
        'init_num_peaks_0': num_peaks_0,
        'init_num_peaks_1': num_peaks_1,
        'init_freq_peaks_0': freq_peaks_0,
        'init_freq_peaks_1': freq_peaks_1,
        'rt_peak_max_0': rt_peak_max_0,
        'rt_peak_min_0': rt_peak_min_0,
        'rt_peak_max_1': rt_peak_max_1,
        'rt_peak_min_1': rt_peak_min_1,
        'mz_peak_max_0': mz_peak_max_0,
        'mz_peak_min_0': mz_peak_min_0,
        'mz_peak_max_1': mz_peak_max_1,
        'mz_peak_min_1': mz_peak_min_1,
    }

    save_json(input_info, input_info_json_file)

    return input_info

# %%
###################
### Wrapper for Study Alignment
###################

def align_study_comparison(input_info,run_params,yes_plot=True,n_neighbors=20,alignment_method='Eclipse'):

    # study_id = input_info['study_id']
    name0 = input_info['name0']
    name1 = input_info['name1']
    output_study_dir = input_info['save_dir']
    run_param_name = run_params['param_name']
    run_id = f'{name0}_{name1}_{run_param_name}'

    if n_neighbors==20:
        output_saved_path = os.path.join(output_study_dir, f'output_{run_id}.json')
    else:
        output_saved_path = os.path.join(output_study_dir, f'output_knn{n_neighbors}_{run_id}.json')
    if (os.path.exists(output_saved_path)) and (not yes_plot):
        print(f'output already exists for {run_id}')
        
        # overwrite the comparison id
        with open(output_saved_path, 'r') as f:
            output_params = json.load(f)
        output_params['comparison_id'] = f'{name0}_vs_{name1}'
        save_json(output_params, output_saved_path)
        
        return


    targets_common = input_info['targets_common']
    if isinstance(targets_common, dict):
        targets_common = pd.DataFrame(targets_common)
    peaks0_path = input_info['peaks0_path']
    peaks1_path = input_info['peaks1_path']
    if 'peaks.pkl' in peaks0_path:
        peaks0_path = peaks0_path.replace('peaks.pkl', 'MSPeaks.pkl')
    if 'peaks.pkl' in peaks1_path:
        peaks1_path = peaks1_path.replace('peaks.pkl', 'MSPeaks.pkl')


    freq_th0 = run_params['freq_th0']
    freq_th1 = run_params['freq_th1']
    if 'alignment_params' in run_params:
        alignment_params = run_params['alignment_params']
    else:    
        alignment_params = None
    

    output_params = {k:v for k,v in run_params.items() if k != 'alignment_params'}
    # output_params0 = {k:v for k,v in input_info.items() if k != 'targets_common'}
    # output_params.update(output_params0)
    # add the alignment params to the output params
    if alignment_params is not None:
        unravel_alignment = unravel_dict(alignment_params,prefix='align')
        output_params.update(unravel_alignment)
    output_params['name0'] = name0
    output_params['name1'] = name1
    output_params['study_id'] = name0 + '_' + name1
    output_params['comparison_id'] = name0 + '_vs_' + name1
    output_params['num common targets'] = targets_common.shape[0]
    

    output_params['alignment_method'] = alignment_method
    output_params['KNN_neighbors'] = n_neighbors
    if 'mode' in input_info:
        output_params['mode'] = input_info['mode']


    # load the peaks object
    # peaks0 = load_mspeaks_from_pickle(peaks0_path)
    # peaks1 = load_mspeaks_from_pickle(peaks1_path)
    peaks0 = MSPeaks()
    peaks0.load_from_pickle(peaks0_path)
    # peaks0.load_from_json(peaks0_path)
    peaks1 = MSPeaks()
    peaks1.load_from_pickle(peaks1_path)
    # peaks1.load_from_json(peaks1_path)

    #  apply the frequency threshold
    peaks0.apply_freq_th_on_peaks(freq_th0)
    peaks1.apply_freq_th_on_peaks(freq_th=freq_th1)

    output_params['num_peaks_0'] = peaks0.peak_info.shape[0]
    output_params['num_peaks_1'] = peaks1.peak_info.shape[0]
    num_peaks =  min(peaks0.peak_info.shape[0], peaks1.peak_info.shape[0])
    output_params['num_peaks'] = num_peaks


    #### these are the numbers we care about
    output_params['NUM TARGETS'] = targets_common.shape[0]
    output_params['NUM POTENTIAL PEAKS'] = num_peaks
    #### #### ####

    # run the MS alignment
    try:
        if alignment_method == 'Eclipse':
            print('running Eclipse alignment')
            alignment_result = align_ms_studies_with_Eclipse(origin_study=peaks0, 
                                                            input_study=peaks1,
                                                            origin_name=name0,
                                                            input_name=name1,
                                                            alignment_params=alignment_params,
                                                            clean_output=True)
            
            alignment_result.to_csv(os.path.join(output_study_dir, f'{run_id}_alignment_result.csv'))
        elif alignment_method == 'metabCombiner':
            print('running metabCombiner alignment')
            alignment_result = align_ms_studies_with_metabCombiner(origin_study=peaks0, 
                                                            input_study=peaks1,
                                                            origin_name=name0,
                                                            input_name=name1,
                                                            alignment_params=alignment_params)
            
            alignment_result.to_csv(os.path.join(output_study_dir, f'{run_id}_alignment_result.csv'))
        else:
            raise NotImplementedError(f'{alignment_method} alignment method not implemented')
    except ValueError:
        print('alignment failed')
        save_json(output_params, output_saved_path)
        return



    ############################
    # Evluate by checking if the features matched to the common targets also match
    fts0 = alignment_result[name0].to_list()
    fts1 = alignment_result[name1].to_list()
    # fts0 = alignment_result.iloc[:,0].to_list()
    # fts1 = alignment_result.iloc[:,1].to_list()
    aligned_fts_tuples = list(zip(fts0, fts1))

    output_params['num common peaks'] = len(aligned_fts_tuples)
    output_params['num common peaks perct'] = len(aligned_fts_tuples)/num_peaks

    ##### these are the numbers we care about
    output_params['PERCENTAGE PEAKS ALIGNED'] = len(aligned_fts_tuples)/num_peaks
    #### #### ####

    targets_common.replace('NA', np.nan, inplace=True)
    targets_common_peak_matched = targets_common.copy()

    output_params['num targets captured by pre-alignment-peaks study0'] =  np.sum(~(targets_common[f'FT_matched_{name0}'].isna()))
    output_params['num targets captured by pre-alignment-peaks study1'] =  np.sum(~(targets_common[f'FT_matched_{name1}'].isna()))

    targets_common_peak_matched.dropna(subset=[f'FT_matched_{name0}',f'FT_matched_{name1}'], inplace=True)

    if targets_common_peak_matched.shape[0] > 0:
        target_fts0 = targets_common_peak_matched[f'FT_matched_{name0}'].to_list()
        target_fts1 = targets_common_peak_matched[f'FT_matched_{name1}'].to_list()
        target_fts_tuples = list(zip(target_fts0, target_fts1))

        num_target_fts_tuples = len(target_fts_tuples)
        num_target_fts_tuples_aligned = len(set(aligned_fts_tuples).intersection(set(target_fts_tuples)))
    else:
        num_target_fts_tuples = 0
        num_target_fts_tuples_aligned = 0

    ##### these are the numbers we care about
    output_params['NUM TARGETS IDENTIFIED IN BOTH STUDIES'] = num_target_fts_tuples
    #### #### ####

    output_params['num targets captured by pre-alignment-peaks both-studies'] = num_target_fts_tuples
    output_params['num targets captured by pre-alignment-peaks correctly-aligned'] = num_target_fts_tuples_aligned
    output_params['percentage targets captured by pre-alignment-peaks correctly-aligned'] = num_target_fts_tuples_aligned/num_target_fts_tuples

    output_params['num_target_fts_tuples'] = num_target_fts_tuples
    output_params['num_target_fts_tuples_aligned'] = num_target_fts_tuples_aligned
    output_params['num_target_fts_tuples_aligned_perct'] = num_target_fts_tuples_aligned/num_target_fts_tuples


    #####################
    ## match to targets after alignment
    rt_peak_max_0 = input_info['rt_peak_max_0']
    rt_peak_max_1 = input_info['rt_peak_max_1']
    aligned_peak_info0 = peaks0.peak_info.loc[fts0]
    aligned_peak_info1 = peaks1.peak_info.loc[fts1]
    
    targets_common_0 = targets_common[[f'rtmed_{name0}', f'mzmed_{name0}']].copy()
    targets_common_0.columns = ['rtmed', 'mzmed']
    targets_common_1 = targets_common[[f'rtmed_{name1}', f'mzmed_{name1}']].copy()
    targets_common_1.columns = ['rtmed', 'mzmed']

    aligned_peak_info0 = peaks_to_targets_wrapper(aligned_peak_info0, targets_common_0, rt_tol=0.02*rt_peak_max_0)
    aligned_peak_info1 = peaks_to_targets_wrapper(aligned_peak_info1, targets_common_1, rt_tol=0.02*rt_peak_max_1)

    aligned_peak_info0_matched = aligned_peak_info0[aligned_peak_info0['target_matched']]
    aligned_peak_info1_matched = aligned_peak_info1[aligned_peak_info1['target_matched']]

    num_aligned_peak_info0_matched = aligned_peak_info0_matched.shape[0]
    num_aligned_peak_info1_matched = aligned_peak_info1_matched.shape[0]
    output_params['num targets captured by post-alignment-peaks study0'] = num_aligned_peak_info0_matched
    output_params['num targets captured by post-alignment-peaks study1'] = num_aligned_peak_info1_matched

    if (num_aligned_peak_info0_matched > 0) and (num_aligned_peak_info1_matched > 0):

        target_post_align_targets0 = aligned_peak_info0_matched['target_name'].to_list()
        target_post_align_targets1 = aligned_peak_info1_matched['target_name'].to_list()
        target_post_align_targets_both = list(set(target_post_align_targets0).intersection(set(target_post_align_targets1)))
        num_target_post_align_targets_both = len(target_post_align_targets_both)
        output_params['num targets captured by post-alignment-peaks both-studies'] = num_target_post_align_targets_both
        target_post_align_targets_either = list(set(target_post_align_targets0).union(set(target_post_align_targets1)))
        num_target_post_align_targets_either = len(target_post_align_targets_either)
        output_params['num targets captured by post-alignment-peaks either-study'] = num_target_post_align_targets_either
        aligned_peak_info0_matched = aligned_peak_info0_matched[
            aligned_peak_info0_matched['target_name'].isin(target_post_align_targets_both)]
        aligned_peak_info1_matched = aligned_peak_info1_matched[
            aligned_peak_info1_matched['target_name'].isin(target_post_align_targets_both)]
        
        aligned_peak_info0_matched.sort_values(by='target_name', inplace=True)
        target_post_align_fts0 = aligned_peak_info0_matched.index.to_list()
        aligned_peak_info1_matched.sort_values(by='target_name', inplace=True)
        target_post_align_fts1 = aligned_peak_info1_matched.index.to_list()
        target_post_align_ft_tuples = list(zip(target_post_align_fts0, target_post_align_fts1))
        
        # num_target_post_align_ft_tuples = len(target_post_align_ft_tuples)
        fts_of_target_post_align_ft_tuples_aligned = list(set(aligned_fts_tuples).intersection(set(target_post_align_ft_tuples)))
        ft0_of_target_post_align_ft_tuples_aligned = [x[0] for x in fts_of_target_post_align_ft_tuples_aligned]
        target_of_target_post_align_ft_tuples_aligned = aligned_peak_info0_matched.loc[ft0_of_target_post_align_ft_tuples_aligned,'target_name'].to_list()
        num_target_post_align_ft_tuples_aligned = len(set(aligned_fts_tuples).intersection(set(target_post_align_ft_tuples)))
        output_params['num targets captured by post-alignment-peaks correctly-aligned'] = num_target_post_align_ft_tuples_aligned
        output_params['percentage targets captured by post-alignment-peaks correctly-aligned'] = (
            num_target_post_align_ft_tuples_aligned/num_target_post_align_targets_both)
        
        ##### these are the numbers we care about
        output_params['PERCENTAGE IDENTIFIED TARGETS CAPTURED'] = (
            num_target_post_align_targets_both/num_target_fts_tuples)

        output_params['PERCENTAGE IDENTIFIED TARGETS ALIGNED'] = (
            num_target_post_align_ft_tuples_aligned/num_target_fts_tuples)
        ##### ##### #####
        
        output_params['targets captured by post-alignment-peaks both-studies'] = target_post_align_targets_both
        output_params['targets captured by post-alignment-peaks correctly-aligned'] = target_of_target_post_align_ft_tuples_aligned
    else:
        output_params['num targets captured by post-alignment-peaks either-study'] = max(
            num_aligned_peak_info0_matched, num_aligned_peak_info1_matched)
        # these are the numbers we care about
        output_params['num targets captured by post-alignment-peaks both-studies'] = 0
        output_params['num targets captured by post-alignment-peaks correctly-aligned'] = 0
        output_params['percentage targets captured by post-alignment-peaks correctly-aligned'] = 0

        output_params['targets captured by post-alignment-peaks both-studies'] = []
        output_params['targets captured by post-alignment-peaks correctly-aligned'] = []
        output_params['PERCENTAGE IDENTIFIED TARGETS ALIGNED'] = 0
        output_params['PERCENTAGE IDENTIFIED TARGETS ALIGNED'] = 0

    # The are the numbers we care about
    # 'num targets captured by post-alignment-peaks both-studies'
    # 'num targets captured by post-alignment-peaks correctly-aligned'



    #####################
    # Evaluate by comparing the predicted (using the aligned peaks) mz and rt target differences to the true values
    coor0 = peaks0.peak_info.loc[fts0,['rtmed','mzmed']]
    coor0.columns = ['RT','MZ']
    coor0.reset_index(inplace=True)
    coor1 = peaks1.peak_info.loc[fts1,['rtmed','mzmed']]
    coor1.columns = ['RT','MZ']
    coor1.reset_index(inplace=True)

    # prepare data for train and testing
    fts_mz0 = coor0['MZ']
    fts_mz_diff = coor0['MZ'] - coor1['MZ']

    if coor0['RT'].max() > 100:
        fts_rt0 = coor0['RT'] # already in seconds
        fts_rt_diff = coor0['RT'] - coor1['RT']
    else:
        fts_rt0 = 60*coor0['RT'] #convert back to seconds
        fts_rt_diff = (coor0['RT'] - coor1['RT'])* 60

    targets_rt0 = targets_common[f'rtmed_{name0}']
    targets_mz0 = targets_common[f'mzmed_{name0}']
    targets_rt_diff = targets_common[f'rtmed_{name0}'] - targets_common[f'rtmed_{name1}']
    targets_mz_diff = targets_common[f'mzmed_{name0}'] - targets_common[f'mzmed_{name1}']

    # output_params['num common peaks'] = len(fts_mz_diff)

    comparison_failed = False
    if len(targets_mz_diff) < 3:
        print('not enough common targets to run KNN')
        comparison_failed = True

    if len(fts_mz_diff) < 30:
        print('not enough common peaks to run KNN')
        comparison_failed = True
        
    if comparison_failed:
        save_json(output_params, output_saved_path)
        return

    # Create the KNN regression model
    knn_model_mz_drift = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model_rt_drift = KNeighborsRegressor(n_neighbors=n_neighbors)

    # Prepare the input features and output
    X_peaks = pd.concat([fts_mz0, fts_rt0], axis=1)
    # Train the model
    knn_model_mz_drift.fit(X_peaks, fts_mz_diff)
    knn_model_rt_drift.fit(X_peaks, fts_rt_diff)

    # evaluate on the targets
    X_targets = pd.concat([targets_mz0, targets_rt0], axis=1)
    X_targets.columns = ['MZ','RT']



    targets_mz_diff_pred = knn_model_mz_drift.predict(X_targets)
    targets_rt_diff_pred = knn_model_rt_drift.predict(X_targets)

    mz_abs_error = np.mean(np.abs(targets_mz_diff - targets_mz_diff_pred))
    rt_abs_error = np.mean(np.abs(targets_rt_diff - targets_rt_diff_pred))

    print(f'mz abs error: {mz_abs_error:.5f}')
    print(f'rt abs error: {rt_abs_error:.5f}')


    output_params['rt abs error'] = rt_abs_error
    output_params['mz abs error'] = mz_abs_error
    ##### these are the numbers we care about
    output_params['ALIGNMENT RT REL ERROR'] = rt_abs_error/np.mean(np.abs(targets_rt_diff))
    ##### ##### #####

    #####################
    # Save and plot the results

    save_json(output_params, output_saved_path)


    if yes_plot:
        fig, axs = plt.subplots(1,2, figsize=(10,5))

        axs[0].scatter(targets_mz_diff, targets_mz_diff_pred)
        x0 = min(min(targets_mz_diff), min(targets_mz_diff_pred))
        x1 = max(max(targets_mz_diff), max(targets_mz_diff_pred))
        axs[0].plot([x0,x1],[x0,x1], color='red', linestyle='--')
        axs[0].set_title('Targets M/Z difference')
        axs[0].set_xlabel('True')
        axs[0].set_ylabel('Predicted')

        axs[1].scatter(targets_rt_diff, targets_rt_diff_pred)
        x0 = min(min(targets_rt_diff), min(targets_rt_diff_pred))
        x1 = max(max(targets_rt_diff), max(targets_rt_diff_pred))
        axs[1].plot([x0,x1],[x0,x1], color='red', linestyle='--')
        axs[1].set_title('Targets RT difference')
        axs[1].set_xlabel('True')
        axs[1].set_ylabel('Predicted')

        plt.suptitle(run_id)
        plt.tight_layout()
        plt.savefig(os.path.join(output_study_dir, f'{run_id}_true_vs_pred.png'))
        plt.close()

        cmap = 'coolwarm'

        # Create a normalize object the scales the colormap
        mz_diff_lim = max(np.max(np.abs(fts_mz_diff)), np.max(np.abs(targets_mz_diff)))

        norm_mz = mcolors.Normalize(vmin=-mz_diff_lim, vmax=mz_diff_lim)

        rt_diff_lim = max(np.max(np.abs(fts_rt_diff)), np.max(np.abs(targets_rt_diff)))
        norm_rt = mcolors.Normalize(vmin=-rt_diff_lim, vmax=rt_diff_lim)
                                    
                                    

        fig, axs = plt.subplots(1,2, figsize=(12,6))
        # adjust subplots to make room in between for the colorbar
        fig.subplots_adjust(wspace=0.3)

        scatter1_mz = axs[0].scatter(X_peaks['RT'], X_peaks['MZ'], c=fts_mz_diff, cmap=cmap, 
                                    norm=norm_mz, s=20, marker='x',label='peaks')

        scatter2_mz = axs[0].scatter(X_targets['RT'], X_targets['MZ'], c=targets_mz_diff, cmap=cmap,
                                        norm=norm_mz, s=50, marker='o',edgecolors='black',label='targets')

        axs[0].set_title('M/Z Drift')
        axs[0].set_xlabel('RT')
        axs[0].set_ylabel('M/Z')

        plt.colorbar(scatter1_mz, ax=axs[0])

        scatter1_rt = axs[1].scatter(X_peaks['RT'], X_peaks['MZ'], c=fts_rt_diff, cmap=cmap,
                                        norm=norm_rt, s=20, marker='x',label='peaks')

        scatter2_rt = axs[1].scatter(X_targets['RT'], X_targets['MZ'], c=targets_rt_diff, cmap=cmap,
                                        norm=norm_rt, s=50, marker='o',edgecolors='black',label='targets')

        axs[1].set_title('RT Drift')
        axs[1].set_xlabel('RT')
        axs[1].set_ylabel('M/Z')
        axs[1].legend(fontsize=10, loc='upper left')

        plt.colorbar(scatter1_rt, ax=axs[1])

        plt.suptitle(run_id + ' True Target Drift')
        plt.savefig(os.path.join(output_study_dir, f'{run_id}_true_drift.png'))
        plt.close()


    return

# %%
###################
### Pathing Functions
###################

def get_result_ids(data_dir):
    result_ids = os.listdir(data_dir)
    #  check that all result ids are directories
    result_ids = [x for x in result_ids if os.path.isdir(os.path.join(data_dir, x))]
    # check that all result ids do not have a period as the first character
    result_ids = [x for x in result_ids if x[0] != '.']

    return result_ids

def get_result_id_subset(data_dir,mode='hilic_pos'):
    result_ids = get_result_ids(data_dir)
    result_ids = [x for x in result_ids if mode in x]
    return result_ids

def get_target_path(result_id, data_dir):
    if os.path.exists(os.path.join(data_dir, result_id, 'targets_clean.csv')):
        target_path = os.path.join(data_dir, result_id, 'targets_clean.csv')

    elif os.path.exists(os.path.join(data_dir, result_id, 'targets.csv')):
        target_path = os.path.join(data_dir, result_id, 'targets.csv')

    else:
        file_list = os.listdir(os.path.join(data_dir, result_id))
        target_path = [x for x in file_list if 'targets' in x][0]
        target_path = os.path.join(data_dir, result_id, target_path)
        # copy to targets.csv
        shutil.copy(target_path, os.path.join(data_dir, result_id, 'targets.csv'))
        target_path = os.path.join(data_dir, result_id, 'targets.csv')
    
    return target_path

def load_result_id(result_id, data_dir):
    result_path = os.path.join(data_dir, result_id)
    target_path = get_target_path(result_id, data_dir)
    peaks_path = os.path.join(result_path, f'{result_id}_MSPeaks.pkl')
    # peaks_path = os.path.join(result_path, f'{result_id}_peaks.json')
    if os.path.exists(peaks_path):
        peaks_obj = MSPeaks()
        peaks_obj.load_from_pickle(peaks_path)
        # peaks_obj.load_from_json(peaks_path)
        # peaks_obj = load_mspeaks_from_pickle(peaks_path)
    else:
        peaks_obj = create_mspeaks_from_mzlearn_result(result_path)
        peaks_obj.save_to_pickle(peaks_path)
        # temp = peaks_obj.to_dict()
        # with open(peaks_path, 'wb') as file:
        #     pickle.dump(temp, file, pickle.HIGHEST_PROTOCOL)
        # peaks_obj.save_to_json(peaks_path)

    return peaks_path, target_path, result_id

# %%
##################
### Gather results
##################

def get_run_param_file_list(param_save_dir,subset_file=None):
    if subset_file is not None:
        with open(subset_file, 'r') as f:
            # subset_list = json.load(f)
            subset_list = f.read().splitlines()
        subset_list = [x+'.json' for x in subset_list]
 

    run_param_files = os.listdir(param_save_dir)
    run_param_files = [x for x in run_param_files if x.endswith('.json')]
    run_param_files = [os.path.join(param_save_dir, x) for x in run_param_files]
    if subset_file is not None:
        run_param_files = [x for x in run_param_files if x.split('/')[-1] in subset_list]

    return run_param_files


def gather_results(comparison_id_list,output_dir,run_param_files=None,n_neighbors=None,ret_output_files=False):

    ret_output_files_list = []
    if n_neighbors is not None:
        print(f'filtering outputs for n_neighbors={n_neighbors}')

    for comparison_id in comparison_id_list:
        output_study_dir = os.path.join(output_dir, comparison_id)
        files_in_dir = os.listdir(output_study_dir)

        output_files = [x for x in files_in_dir if x.startswith('output_')]
        output_files = [x for x in output_files if x.endswith('.json')]
        if run_param_files is not None:
            output_files_agg = []
            for run_param_file in run_param_files:
                if '.json' not in run_param_file:
                    run_param_file = run_param_file + '.json'
                if '/' in run_param_file:
                    run_param_file = run_param_file.split('/')[-1] 
                
                output_files_agg.extend(
                    [x for x in output_files if x.endswith(run_param_file)])
            output_files = output_files_agg


        output_files = [os.path.join(output_study_dir, x) for x in output_files]
        # redundant check
        # output_files = [x for x in output_files if os.path.exists(x)]
        if n_neighbors is not None:
            output_files = [x for x in output_files if f'knn{n_neighbors}_' in x]

        if ret_output_files:
            ret_output_files_list.extend(output_files)
            continue

        output_results = []
        for output_file in output_files:
            output_params = load_json(output_file)
            output_params_clean = {}
            
            # remove dictionary and list values
            for k,v in output_params.items():
                if isinstance(v, dict):
                    continue
                elif isinstance(v, list):
                    continue
                output_params_clean[k] = v

            output_results.append(output_params_clean)

        output_results_df = pd.DataFrame(output_results)

        if os.path.exists(os.path.join(output_study_dir, 'input_info.json')):
            input_info = load_json(os.path.join(output_study_dir, 'input_info.json'))

            # remove dictionary values
            input_info = {k:v for k,v in input_info.items() if not isinstance(v, dict)}
            # remove paths
            input_info = {k:v for k,v in input_info.items() if not (isinstance(v, str) and '/' in v)}
            input_info_df = pd.DataFrame([input_info])

            if 'comparison_id' not in input_info_df.columns:
                input_info_df['comparison_id'] = comparison_id
            if 'comparison_id' not in output_results_df.columns:
                output_results_df['comparison_id'] = comparison_id
            output_results_df = pd.merge(input_info_df, output_results_df, how='outer', on='comparison_id', suffixes=('_input', ''))
            rem_cols = [x for x in output_results_df.columns if x.endswith('_input')]
            output_results_df.drop(rem_cols, axis=1, inplace=True)

        output_results_df.to_csv(os.path.join(output_study_dir, f'{comparison_id}_output_results.csv'))

    if ret_output_files:
        return ret_output_files_list

    all_output_results = []
    for comparison_id in comparison_id_list:
        output_study_dir = os.path.join(output_dir, comparison_id)
        output_results_path = os.path.join(output_study_dir, f'{comparison_id}_output_results.csv')
        output_results_df = pd.read_csv(output_results_path)
        all_output_results.append(output_results_df)


    all_output_results_df = pd.concat(all_output_results)
    all_output_results_df.to_csv(os.path.join(output_dir, f'all_output_results.csv'))

    return None


# %%
###################
### Main
###################
data_dir = '/Users/jonaheaton/Desktop/alignment_analysis/Input_Data'
output_dir = '/Users/jonaheaton/Desktop/alignment_analysis/Output_Data'
master_comparison_id_list = []

def main(mode='hilic_pos',comparison_id_list=None,run_param_files=None,comparison_id_subset=None,n_neighbors=8):

    if comparison_id_list is None:
        comparison_id_list = []

    if run_param_files is None:
        run_param_files = []

    result_id_list = get_result_id_subset(data_dir,mode=mode)
    for i0, result_id0 in enumerate(result_id_list):
        peaks_path0, target_path0, result_id0 = load_result_id(result_id0, data_dir)
        print('id0:')
        print(peaks_path0)
        print(target_path0)
        print(result_id0)
        print('')

        for i1, result_id1 in enumerate(result_id_list):
            if i1 <= i0:
                continue
            if result_id0 == result_id1:
                continue
            

            peaks_path1, target_path1, result_id1 = load_result_id(result_id1, data_dir)
            print('id1:')
            print(peaks_path1)
            print(target_path1)
            print(result_id1)
            print('')

            comparison_id = f'{result_id0}_vs_{result_id1}'
            
            if comparison_id_subset is not None:
                if comparison_id not in comparison_id_subset:
                    continue
            comparison_id_list.append(comparison_id)

            input_info = initialize_study_comparison(target_path0, target_path1, peaks_path0, peaks_path1, 
                                                     result_id0, result_id1, output_dir,yes_plot=False)
            input_info['mode'] = mode
            yes_plot = False
            # n_neighbors = 8

            # run_params = {
            # 'freq_th0': 0.5,
            # 'freq_th1': 0.5,
            # 'param_name': 'Original_Eclipse_50_50',
            # }
            # run_params['alignment_params'] = {
            #     'cutoffs': {"RT": 6, "Intensity": 6, "MZ": 6},
            #     'weights': {"RT": 1, "Intensity": 1, "MZ": 1},
            #     'coarse_params': { "RT": {"upper": 0.5, "lower": -0.5}, 
            #         "MZ": {"upper": 15, "lower": -15}, 
            #         "Intensity": {"upper": 2, "lower": -2}}}
            
            # align_study_comparison(input_info,run_params=run_params,yes_plot=yes_plot,
            #                        n_neighbors=n_neighbors,alignment_method='Eclipse')


            for param_file in run_param_files:
                if '.json' not in param_file:
                    param_file = param_file + '.json'                 
                run_params = load_json(param_file)
                if 'param_name' not in run_params:
                    run_params['param_name'] = param_file.split('/')[-1].split('.')[0]
                
                if 'alignment_method' not in run_params:
                    raise ValueError('alignment_method must be specified in run_params')
                else:
                    alignment_method = run_params['alignment_method']
                    # check that alignment_method is valid
                    if alignment_method not in ['Eclipse','metabCombiner']:
                        raise ValueError(f'{alignment_method} is not a valid alignment method')


                align_study_comparison(input_info,run_params=run_params,yes_plot=yes_plot,
                                       n_neighbors=n_neighbors,alignment_method=alignment_method)



    return comparison_id_list

# %%
###################
## Parameter Cleaing Functions

def change_param_freq_threshold(param_file_path,freq_th0,freq_th1):
    param_file_dir = os.path.dirname(param_file_path)
    run_params = load_json(param_file_path)
    run_params['freq_th0'] = freq_th0
    run_params['freq_th1'] = freq_th1
    freq_th0_perc = int(100*freq_th0)
    freq_th1_perc = int(100*freq_th1)
    if 'method_param_name' not in run_params:
        run_params['method_param_name'] =  get_method_param_name(run_params['param_name'])
    run_params['param_name'] = run_params['alignment_method'] + f'_{freq_th0_perc}_{freq_th1_perc}' + '_' + run_params['method_param_name']
    save_path = os.path.join(param_file_dir, run_params['param_name'] + '.json')
    save_json(run_params, save_path)
    return save_path

def get_method_param_name(param_name):
    # count the number of underscores
    num_underscores = param_name.count('_')
    if num_underscores == 2:
        method_param_name = 'default'
    else:
        method_param_name = param_name.split('_')[-1]
    return method_param_name

# %%
########################

if __name__ == '__main__':

    explore_freq_threshold= True
    method_metabC_grid_search = False
    method_Eclipse_grid_search = False
    skip_alignment = False
    n_neighbors = 5

    comparison_id_list = []
    alignment_param_path = '/Users/jonaheaton/Desktop/alignment_analysis/Alignment_Params'
    
    if method_metabC_grid_search:
        # Run a grid search for exploration
        run_param_file_names = create_metaCombiner_grid_search(save_dir=alignment_param_path,
                                    freq_th0 = [0.25,0.5],
                                    freq_th1 = [0.25,0.5],
                                    tolmz = [0.003,0.005],
                                    windx = [0.01],
                                    windy = [0.01,0.03])
    elif method_Eclipse_grid_search:
        raise NotImplementedError('Eclipse grid search not implemented')
    else:
        # Comparison to the original alignment parameters
        # run_param_file_names = ['Eclipse_50_50.json','metabCombiner_50_50.json','metabCombiner_50_50_Jan23.json']
        # run_param_file_names = ['metaCombiner_50_50']
        run_param_file_names = ['Eclipse_50_50.json','metabCombiner_50_50_Jan23.json']

    # get the file paths to the run param files
    run_param_files = [os.path.join(alignment_param_path, x) for x in run_param_file_names]

    # explore the frequency threshold parameter
    if explore_freq_threshold:
        new_run_param_files = []
        for param_file in run_param_files:
            freq_th_list = [0.25,0.5,0.75]
            for freq_th0 in freq_th_list:
                for freq_th1 in freq_th_list:
                    new_run_param_files.append(change_param_freq_threshold(param_file, freq_th0, freq_th1))


    # Run the alignment tests over all the modes
    if skip_alignment:
        mode_list = []
    else:
        mode_list = ['hilic_pos','hilic_neg','rp_pos','rp_neg']
        for mode in mode_list:
            comparison_id_list = main(mode,
                                    comparison_id_list=comparison_id_list,
                                    run_param_files=run_param_files,
                                    n_neighbors=n_neighbors)


        # save the comparison id list
        # with open(os.path.join(output_dir, 'comparison_id_list.txt'), 'w') as f:
        #     for comparison_id in comparison_id_list:
        #         f.write(comparison_id + '\n')


    # open the comparison id list
    with open(os.path.join(output_dir, 'comparison_id_list.txt'), 'r') as f:
        comparison_id_list = f.read().splitlines()

    # gather the results for the params of interest
    output_files = gather_results(comparison_id_list,output_dir,
                                  run_param_files=run_param_files,
                                  n_neighbors=n_neighbors,
                                ret_output_files=False)





# %%
# delete all of the output files
# for output_file in output_files:
#     os.remove(output_file)
# %%
