# Align Multiple Studies together

import os
import pandas as pd
import json
import numpy as np
import shutil
import re

from .align_pair import align_ms_studies
from .mspeaks import MSPeaks, create_mspeaks_from_mzlearn_result


def align_multiple_ms_studies(origin_peak_obj_path, input_peak_obj_path_list, save_dir, **kwargs):
    '''
    Aligns multiple mass spectrometry studies to an origin study.

    Parameters:
        - origin_peak_obj_path (str): The file path of the origin study MSPeaks object. 
            Optionally may give a path to an mzlearn result directory instead.
        - input_peak_obj_path_list (list): A list of file paths of the input study MSPeaks objects.
            Optionally may give a list of paths to mzlearn result directories instead.
        - save_dir (str): The directory to save the alignment results.
        - **kwargs: Additional keyword arguments for customization.

    Keyword Arguments:
        origin_name (str): The name of the origin study. Default is 'origin'.
        input_name_list (list): A list of names for the input studies. Default is None.
        alignment_method (str): The alignment method to use. Default is None.
        alignment_params (dict): Additional parameters for the alignment method. Default is None.
        origin_freq_th (float): The frequency threshold for the origin study. Default is None.
        input_freq_th (float): The frequency threshold for the input studies. Default is None.
        origin_select_files (list): A list of files to apply the frequency threshold on in the origin study. Default is None.
        fill_na_strategy (str): The strategy to fill missing values. Default is None.
        outlier_samples_removal_strategy (str): The strategy to remove outlier samples. Default is None.
        verbose (bool): Whether to print verbose output. Default is True.
        save_cleaned_peak_obj (bool): Whether to save the cleaned MSPeaks objects. Default is False.

    Returns:
        pandas.DataFrame: The multi-aligned DataFrame containing the alignment results.
    '''
    # Function code here
def align_multiple_ms_studies(origin_peak_obj_path, input_peak_obj_path_list, save_dir, **kwargs):
    '''
    
    '''

    origin_name = kwargs.get('origin_name', 'origin')
    input_name_list = kwargs.get('input_name_list', None)
    alignment_method = kwargs.get('alignment_method', None)
    alignment_params = kwargs.get('alignment_params', None)
    origin_freq_th = kwargs.get('origin_freq_th', None)
    input_freq_th = kwargs.get('input_freq_th', None)
    origin_select_files = kwargs.get('origin_select_files', None)
    fill_na_strategy = kwargs.get('fill_na_strategy', None)
    outlier_samples_removal_strategy = kwargs.get('outlier_samples_removal_strategy', None)
    verbose = kwargs.get('verbose', True)
    save_cleaned_peak_obj = kwargs.get('save_cleaned_peak_obj', False)

    ## Origin study Preparation
    if 'pkl' not in origin_peak_obj_path:
        # assume that this is a mzlearn result directory instead
        origin_study = create_mspeaks_from_mzlearn_result(origin_peak_obj_path)
        origin_study.add_study_id(origin_name,rename_samples=False)
    else:
        origin_study = MSPeaks()
        origin_study = origin_study.load(origin_peak_obj_path)
        assert origin_study.study_id is not None, 'origin study does not have a study_id'
        if origin_study.study_id != origin_name:
            if verbose:
                print(f'WARNING: origin study_id "{origin_study.study_id}" does not match origin_name "{origin_name}"')


    if outlier_samples_removal_strategy is not None:
        origin_study.remove_outlier_samples(method=outlier_samples_removal_strategy)

    if origin_freq_th is not None:
        #TODO check that the origin_select_files are in the origin_study
        origin_study.apply_freq_th_on_peaks(freq_th=origin_freq_th,
                                        inplace=True,
                                        sample_subset=origin_select_files)
        
    if fill_na_strategy is not None:
        origin_study.fill_missing_values(method=fill_na_strategy)
    else:
        if verbose: print('WARNING: no fill_na_strategy provided, missing values will not be filled')

    if save_cleaned_peak_obj:
        origin_study.save(os.path.join(save_dir,f'{origin_name}_cleaned.pkl'))


    ## Input study Preparation
    def prep_input_study(input_peak_obj_path,input_name):
        if 'pkl' not in input_peak_obj_path:
            # assume that this is a mzlearn result directory instead
            input_study = create_mspeaks_from_mzlearn_result(input_peak_obj_path)
            input_study.add_study_id(input_name,rename_samples=False)
        else:
            input_study = MSPeaks()
            input_study = input_study.load(input_peak_obj_path)
            assert input_study.study_id is not None, 'input study does not have a study_id'
            # if input_study.study_id != input_name:
        
        if outlier_samples_removal_strategy is not None:
            input_study.remove_outlier_samples(method=outlier_samples_removal_strategy)

        if input_freq_th is not None:
            input_study.apply_freq_th_on_peaks(freq_th=input_freq_th,
                                            inplace=True)
        if fill_na_strategy is not None:
            input_study.fill_missing_values(method=fill_na_strategy)

        if save_cleaned_peak_obj:
            input_study.save(os.path.join(save_dir,f'{input_name}_cleaned.pkl'))

        return input_study


    ## Align the input studies to the origin study
    if input_name_list is None:
        input_name_list = [f'input_{i}' for i in range(len(input_peak_obj_path_list))]

    if alignment_method is None:
        alignment_method = 'eclipse'

    for input_peak_obj_path,input_name in zip(input_peak_obj_path_list,input_name_list):

        print(f'aligning {input_name} to {origin_name}')
        input_study = prep_input_study(input_peak_obj_path,input_name)
        _ = align_ms_studies(origin_study=origin_study,
                        input_study=input_study,
                        origin_name=origin_name,
                        input_name=input_name,
                        alignment_method=alignment_method,
                        alignment_params=alignment_params,
                        save_dir=save_dir,
                        verbose=verbose)
        
    # combine the pair-aligned DataFrame with the existing multi-aligned DataFrame
    multi_alignment_df = combine_alignments_in_dir(save_dir,origin_name=origin_name,
                                                  input_name_list=input_name_list,
                                                  alignment_method=alignment_method)
    

    multi_alignment_df.to_csv(os.path.join(save_dir,f'alignment_df.csv'))

    return multi_alignment_df


######################
# %% Helper Functions
######################

# %% combine_two_alignment_results
def combine_two_alignment_results(existing_align_df, align_df=None,
                                  origin_name='origin', input_name='input',
                                  origin_col=None, input_col=None):
    """
    Combines two alignment results into a single DataFrame.

    Parameters:
    - existing_align_df (pandas.DataFrame): The existing multi-alignment DataFrame.
    - align_df (pandas.DataFrame, optional): The pair-alignment DataFrame to be combined with the existing DataFrame.
    - origin_name (str, optional): The name of the origin column in the alignment DataFrames. Default is 'origin'.
    - input_name (str, optional): The name of the input column in the alignment DataFrames. Default is 'input'.
    - origin_col (str, optional): The name of the origin column in the align_df DataFrame. If not provided, it will be set to origin_name.
    - input_col (str, optional): The name of the input column in the align_df DataFrame. If not provided, it will be set to input_name.

    Returns:
    - pandas.DataFrame: The combined alignment DataFrame.

    Raises:
    - ValueError: If the index name of existing_align_df is different from the index name of align_df.
    """
    
    if align_df is None:
        return existing_align_df
    
    if origin_col is None:
        origin_col = origin_name
    if input_col is None:
        input_col = input_name

    kept_cols = [origin_col, input_col]
    align_df = align_df[kept_cols]
    if origin_col != origin_name:
        align_df.rename(columns={origin_col: origin_name}, inplace=True)
    if input_col != input_name:
        align_df.rename(columns={input_col: input_name}, inplace=True)

    align_df.set_index(origin_name, inplace=True)
    if existing_align_df is None:
        return align_df
    
    if existing_align_df.index.name != align_df.index.name:
        raise ValueError(f'existing_align_df has index name {existing_align_df.index.name} but align_df has index name {align_df.index.name}')
    
    align_df = align_df.merge(existing_align_df,
                              left_index=True,
                              right_index=True,
                              how='outer')
    return align_df





# %% combine_alignments_in_dir
def combine_alignments_in_dir(save_dir,origin_name='origin',input_name_list=None,alignment_method=None,
                              origin_col=None,input_col_format=None):
    """
    Combines multiple alignment results from CSV files in a directory into a single DataFrame.

    Parameters:
    - save_dir (str): The directory path where the alignment CSV files are located.
    - origin_name (str, optional): The name of the origin column for the  multi-alignment CSV file. Default is 'origin'.
    - input_name_list (list, optional): A list of input names to include in the combined alignment. 
                                        If None, all input names found will be included. Default is None.
    - alignment_method (str, optional): The alignment method used in for the pair-alignment CSV files. 
                                        If None, the method will be inferred from the file names. Default is None.
    - origin_col (str, optional): The name of the origin column in the pair-alignment CSV files. 
                                       If None, it will be inferred from the DataFrame columns. Default is None.
    - input_col_format (str, optional): The format of the input column names in the pair-alignment CSV files. 
                                        If None, it will be inferred from the DataFrame columns. Default is None.                                       

    Returns:
    - existing_align_df (pandas.DataFrame): The combined alignment DataFrame.

    Raises:
    - ValueError: If the origin name column or input name column cannot be found in any of the alignment CSV files.
    - ValueError: If no alignment CSV files are found in the directory.
    
    """
    
    # assume the alignment file names are of the form: 
    # {input_name}_aligned_to_{origin_name}_with_{alignment_method}.csv

    if origin_col is None:
        origin_col = origin_name
    
    if input_col_format is None:
        input_col_format = '{}'

    existing_align_df = None
    if alignment_method is None:
        # file_format = f'INPUT_aligned_to_{origin_name}.csv'
        file_format_check = f'_aligned_to_{origin_name}.csv'
    else:
        # file_format = f'INPUT_aligned_to_{origin_name}_with_{alignment_method}.csv'
        file_format_check = f'_aligned_to_{origin_name}_with_{alignment_method}.csv'

    count_multi_alignments = 0
    for file in os.listdir(save_dir):
        if file_format_check in file:
            # the easy but breakable way to find the input name
            # will break if a '_' is in the input name
            # input_name = file.split('_')[0]
            
            # find input_name using regex
            # input_name = re.findall(r'(.*)_aligned_to',file)[0]

            # find input name using regeg search
            match = re.search(r'(.*)_aligned_to',file)
            if match:
                input_name = match.group(1)
            else:
                raise ValueError(f'could not find input name in {file}')
            
            if input_name_list is not None:
                if input_name not in input_name_list:
                    continue

            new_align_df = pd.read_csv(os.path.join(save_dir,file))
            
            # Choose the origin name column to be found in the pair-alignment file
            if (origin_col is not None) and (origin_col in new_align_df.columns):
                origin_name_col = origin_col
            elif origin_name in new_align_df.columns:
                origin_name_col = origin_name
            elif 'origin' in new_align_df.columns:
                origin_name_col = 'origin'
            else:
                raise ValueError(f'could not find origin name column in {file}')
            
            # Choose the input name column to be found in the pair-alignment file
            input_name_col = input_col_format.format(input_name)
            if (input_name_col is not None) and (input_name_col in new_align_df.columns):
                input_name_col = input_name_col
            elif input_name in new_align_df.columns:
                input_name_col = input_name
            elif 'input' in new_align_df.columns:
                input_name_col = 'input'
            else:
                raise ValueError(f'could not find input name column in {file}')

            # combine the new pair-alignment with the existing multi-alignment
            existing_align_df = combine_two_alignment_results(existing_align_df,new_align_df,
                                                                origin_name=origin_name,input_name=input_name,
                                                                origin_col=origin_name_col,input_col=input_name_col)
            count_multi_alignments += 1
    
    if count_multi_alignments < 1:
        raise ValueError(f'could not find any alignment files with format {file_format}')

    existing_align_df.to_csv(os.path.join(save_dir,f'combined_{origin_name}_alignments.csv'))
    return existing_align_df