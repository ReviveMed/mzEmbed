# Align Multiple Studies together

import os
import pandas as pd
import json
import numpy as np
import shutil


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





def combine_alignments_in_dir(save_dir, origin_name='origin', input_name_list=None, alignment_method=None,
                              origin_name_col=None):
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
    # Function implementation...
def combine_alignments_in_dir(save_dir,origin_name='origin',input_name_list=None,alignment_method=None,
                              origin_col=None,input_col_format=None):
    
    # assume the alignment file names are of the form: 
    # {input_name}_aligned_to_{origin_name}_with_{alignment_method}.csv

    if origin_col is None:
        origin_col = origin_name
    
    if input_col_format is None:
        input_col_format = '{}'

    existing_align_df = None
    if alignment_method is None:
        file_format = f'INPUT_aligned_to_{origin_name}.csv'
        file_format_check = f'_aligned_to_{origin_name}.csv'
    else:
        file_format = f'INPUT_aligned_to_{origin_name}_with_{alignment_method}.csv'
        file_format_check = f'_aligned_to_{origin_name}_with_{alignment_method}.csv'

    count_multi_alignments = 0
    for file in os.listdir(save_dir):
        if file_format_check in file:
            input_name = file.split('_')[0]
            
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