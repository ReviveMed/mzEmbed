import os
import json
import pandas as pd
from .utils_eclipse import align_ms_studies_with_Eclipse
from .utils_metabCombiner import align_ms_studies_with_metabCombiner
from .utils_merge import align_ms_studies_with_merge
from .mspeaks import MSPeaks
from .utils_targets import process_targeted_data
from .utils_misc import load_json, save_json

# %% align_ms_studies
def align_ms_studies(origin_study, input_study, origin_name='origin', input_name='input',
                     alignment_method='metabcombiner', alignment_params=None, save_dir=None,
                     verbose=False, ret_score=False):
    '''
    Aligns two mass spectrometry studies using the specified alignment method.

    Parameters:
    origin_study (MSPeaks Obj): MSPeaks Object for the origin study.
    input_study (MSPeaks Obj)): MSPeaks Object for the input study.
    origin_name (str, optional): Name of the origin study. Defaults to 'origin'.
    input_name (str, optional): Name of the input study. Defaults to 'input'.
    alignment_method (str, optional): Alignment method to use. Defaults to 'metabcombiner'.
    alignment_params (dict, optional): Parameters for the alignment method. Defaults to None.
    save_dir (str, optional): Directory to save alignment results. Defaults to None.

    Returns:
    pandas.DataFrame: The alignment result as a DataFrame.

    Raises:
    NotImplementedError: If the specified alignment method is not implemented.

    '''

    alignment_result_df = None

    if save_dir is not None:
        with open(os.path.join(save_dir, f'{input_name}_aligned_to_{origin_name}_{alignment_method}_params.json'), 'w') as fp:
            json.dump(alignment_params, fp)

    try:
        if 'merge' in alignment_method.lower():
            if verbose: print('running merge alignment')
            alignment_result_df = align_ms_studies_with_merge(origin_study=origin_study, 
                                                              input_study=input_study,
                                                              origin_name=origin_name,
                                                              input_name=input_name,
                                                              params_list=alignment_params)
                                                            #   save_dir=save_dir)
        
        elif 'eclipse' in alignment_method.lower():
            if verbose: print('running Eclipse alignment')
            alignment_result_df = align_ms_studies_with_Eclipse(origin_study=origin_study, 
                                                                input_study=input_study,
                                                                origin_name=origin_name,
                                                                input_name=input_name,
                                                                alignment_params=alignment_params,
                                                                clean_output=True)
        elif ('metabcombiner' in alignment_method.lower()):
        # elif (alignment_method.lower() == 'metabcombiner') or (alignment_method.lower() == 'metacombiner'):
            if verbose: print('running metabCombiner alignment')
            alignment_result_df = align_ms_studies_with_metabCombiner(origin_study=origin_study, 
                                                                      input_study=input_study,
                                                                      origin_name=origin_name,
                                                                      input_name=input_name,
                                                                      alignment_params=alignment_params)
            
            
        else:
            raise NotImplementedError(f'{alignment_method} alignment method not implemented')
    
        # if we want to score the alignment
        num_origin_peaks = origin_study.get_num_peaks()
        num_input_peaks = input_study.get_num_peaks()
        num_peaks = min(num_origin_peaks,num_input_peaks)
        num_aligned_peaks = len(alignment_result_df)
        alignment_score = num_aligned_peaks/num_peaks
    
    except ValueError:
        print('alignment failed')
        alignment_score = 0

    if (save_dir is not None) and (alignment_result_df is not None):
        alignment_result_df.to_csv(os.path.join(save_dir, f'{input_name}_aligned_to_{origin_name}_with_{alignment_method}.csv'))

    if ret_score:
        return alignment_result_df, alignment_score

    return alignment_result_df


# %% initialize_study_pair
def initialize_study_pair(peaks_obj_path0,peaks_obj_path1,save_dir,**kwargs):

    mz_tol = kwargs.get('mz_tol', 0.005)
    yes_plot = kwargs.get('yes_plot', False)
    name0 = kwargs.get('name0', None)
    name1 = kwargs.get('name1', None)
    targets_path0 = kwargs.get('targets_path0', None)
    targets_path1 = kwargs.get('targets_path1', None)

    input_info_json_file = os.path.join(save_dir, 'input_info.json')
    if (os.path.exists(input_info_json_file)) and (not yes_plot):
        print('input info already exists')
        input_info = load_json(input_info_json_file)
        # overwrite the comparison id
        input_info['comparison_id'] = f'{name0}_vs_{name1}'
        save_json(input_info, input_info_json_file)
        return input_info


    # Choose the names of the studies
    if name0 is None:
        name0 = 'origin'
    if name1 is None:
        name1 = 'input'
    if name0 == name1:
        raise ValueError('name0 and name1 cannot be the same')


    pass

# %% initialize_study_pair_with_targets
def initialize_study_pair_with_targets():
    pass



def run_study_pair_alignment():
    pass


def evaluate_study_pair_alignment_using_targets():
    pass