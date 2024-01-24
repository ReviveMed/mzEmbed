import os
import json
import pandas as pd
from .utils_eclipse import align_ms_studies_with_Eclipse
from .utils_metabCombiner import align_ms_studies_with_metabCombiner
from .mspeaks import MSPeaks
from .utils_targets import process_targeted_data


def align_ms_studies(origin_study, input_study, origin_name='origin', input_name='input',
                     alignment_method='metabcombiner', alignment_params=None, save_dir=None):
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
        if alignment_method.lower() == 'eclipse':
            print('running Eclipse alignment')
            alignment_result_df = align_ms_studies_with_Eclipse(origin_study=origin_study, 
                                                                input_study=input_study,
                                                                origin_name=origin_name,
                                                                input_name=input_name,
                                                                alignment_params=alignment_params,
                                                                clean_output=True)
        elif (alignment_method.lower() == 'metabcombiner') or (alignment_method.lower() == 'metacombiner'):
            print('running metabCombiner alignment')
            alignment_result_df = align_ms_studies_with_metabCombiner(origin_study=origin_study, 
                                                                      input_study=input_study,
                                                                      origin_name=origin_name,
                                                                      input_name=input_name,
                                                                      alignment_params=alignment_params)
        else:
            raise NotImplementedError(f'{alignment_method} alignment method not implemented')
    except ValueError:
        print('alignment failed')

    if (save_dir is not None) and (alignment_result_df is not None):
        alignment_result_df.to_csv(filepath=os.path.join(save_dir, f'{input_name}_aligned_to_{origin_name}_with_{alignment_method}.csv'))

    return alignment_result_df



def initialize_study_pair():
    pass


def initialize_study_pair_with_targets():
    pass



def run_study_pair_alignment():
    pass


def evaluate_study_pair_alignment_using_targets():
    pass