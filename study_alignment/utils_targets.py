import os
import pandas as pd
import sys
import json
import numpy as np
# import matplotlib.pyplot as plt
import trackpy as tp


######### New and Improved linking between targets and features #########

def link_targets_to_peaks(peaks_df,targets_df,rt_tol=20,mz_tol=0.005,\
                          target_mz_col='mzmed',target_rt_col='rtmed',target_name_col=None,\
                          peak_freq_col=None, verbose=True,join='True'):
    
    '''
    peaks_df: dataframe of peaks
    targets_df: dataframe of targets
    rt_tol: rt tolerance for linking
    mz_tol: mz tolerance for linking
    target_mz_col: column name for target m/z
    target_rt_col: column name for target rt
    target_name_col: column name for target name
    peak_freq_col: column name for peak frequency (if provided used to weight the linking to higher frequency peaks)
    verbose: print out some info
    join: join the matches to the original peaks_df
    '''

    if target_name_col is None:
        if 'target_name' in targets_df.columns:
            target_name_col = 'target_name'

    peaks = peaks_df[['mzmed','rtmed']].copy()
    peaks['feats'] = peaks_df.index
    peaks['frame'] = 0
    peaks['x'] = peaks_df['rtmed']/rt_tol
    peaks['y'] = peaks_df['mzmed']/mz_tol
    search_range = 1  # search range for linking

    if peak_freq_col is not None:
        peaks['z'] = peaks_df.loc[:,peak_freq_col]
        search_range = 1.1*search_range  # since adding extra dimension, increase the  search range for linking



    targets = targets_df[[target_mz_col,target_rt_col]].copy()
    targets.rename(columns={target_mz_col:'target_mz',target_rt_col:'target_rt'}, inplace=True)
    if target_name_col is None: targets['target_name'] = targets_df.index
    else: targets['target_name'] = targets_df.loc[:,target_name_col]

    targets.dropna(inplace=True,subset=['target_mz','target_rt'])
    if targets.shape[0] == 0:
        print('no targets found')
        return pd.DataFrame(),np.nan,pd.DataFrame()
    targets['frame'] = 1
    targets['x'] = targets['target_rt']/rt_tol
    targets['y'] = targets['target_mz']/mz_tol
    if peak_freq_col is not None:
        targets['z'] = 1

    coors = pd.concat([peaks,targets],axis=0)
    memory = 0  # number of frames to remember past peaks
    link_strategy = 'best'
    if ~verbose:
        tp.quiet()
        # sinani matching breaks here for some reason?
    #drop the nan values
    num_nans = coors['x'].isna().sum()
    if num_nans > 0:
        print(f'dropping {num_nans} nan values before peak-to-target-linking')
    coors.dropna(subset=['x','y'],inplace=True)
    coors_matched = tp.link_df(coors, search_range=search_range, memory=memory)
    if (coors_matched is None) or (len(coors_matched) == 0):
        print('no matches found')
        return None
    
    if peak_freq_col is not None:
        coors_matched.drop(columns=['z'], inplace=True)

    peaks = coors_matched[coors_matched['frame'] == 0].drop(columns=['frame','target_name','x','y'])
    targets = coors_matched[coors_matched['frame'] == 1].drop(columns=['frame','feats','x','y'])

    matches = pd.merge(peaks[['feats','mzmed','rtmed','particle']],
                       targets[['target_mz','target_rt','target_name','particle']],
                         on='particle', how='left', suffixes=('', '_target'))
    matches.set_index('feats', inplace=True)
    matches['target_matched'] = matches['target_name'].notnull()
    

    num_matches = matches['target_matched'].sum()
    missing_targets = targets[~targets['target_name'].isin(matches['target_name'])][['target_rt','target_mz','target_name']]
    found_targets = targets[targets['target_name'].isin(matches['target_name'])][['target_rt','target_mz','target_name']]
    true_pos_perct = num_matches/len(targets)

    if verbose:
        print('found {} out of {} total targets'.format(num_matches,len(targets)))
        print('true positive rate: {}'.format(true_pos_perct))

    if join:
        matches = peaks_df.join(matches[['target_rt','target_mz','target_name','target_matched']], how='left')
        matches['target_matched'].fillna(False, inplace=True)
    # return matches, missing_targets, found_targets
    return matches,true_pos_perct, missing_targets

match_targets_to_peaks = link_targets_to_peaks

#################
## One to many linking, where one target can be linked to multiple peaks
#################

def get_potential_target_matches(targets_df,peak_info=None,rt_tol=None,result_path=None,
                                 mz_tol=0.005):
    if result_path is None:
        assert peak_info is not None
        assert rt_tol is not None

    if peak_info is None:
        peak_info_file = os.path.join(result_path,'final_peaks','extra_data.csv')
        if not os.path.exists(peak_info_file):
            peak_info_file = os.path.join(result_path, 'final_peaks', 'peak_info.csv')
        peak_info = pd.read_csv(peak_info_file,index_col=0)

    if rt_tol is None:
        overall_params = load_params_from_file(os.path.join(result_path,'params_overall.csv'))
        overall_rt_max = float(overall_params['overall_rt_max'])
        rt_tol = 0.02*overall_rt_max

    if rt_tol<0:
        # if rt_tol is negative, then only match based on mz
        only_mz_match = True
    else:
        only_mz_match = False
        # remove the targets that don't have an rt
        targets_df.dropna(subset=['rtmed','mzmed'],inplace=True) 


    num_exp_targets = targets_df.shape[0]
    peak_info['potential_target_id'] = ''
    peak_info['potential_target'] = False
    peak_info['potential_target_count'] = 0
    number_matched_targets = 0

    for id, target in targets_df.iterrows():
        target_mz = target['mzmed']
        if only_mz_match:
            potential_idx = (peak_info['mzmed'].between(target_mz-mz_tol,target_mz+mz_tol))
        else:
            target_rt = target['rtmed']
            potential_idx = (peak_info['mzmed'].between(target_mz-mz_tol,target_mz+mz_tol)) & (peak_info['rtmed'].between(target_rt-rt_tol,target_rt+rt_tol))

    # for id in targets.index:
        # potential_idx = (peak_info['mzmed'].between(targets.loc[id,'mzmed']-mz_tol,targets.loc[id,'mzmed']+mz_tol)) & (peak_info['rtmed'].between(targets.loc[id,'rtmed']-rt_tol,targets.loc[id,'rtmed']+rt_tol))
        peak_info.loc[potential_idx,'potential_target'] = True
        peak_info.loc[potential_idx,'potential_target_count'] += 1
        peak_info.loc[potential_idx,'potential_target_id'] += id

        if np.sum(potential_idx) > 0:
            number_matched_targets += 1

    # count the number of features that were matched to a target
    print(f'num peaks matched to a target: {peak_info.loc[peak_info["potential_target"]].shape[0]}')

    # count the number of targets that were matched
    print(f'num targets matched: {number_matched_targets} out of {targets_df.shape[0]}')

    # check the target_id column and if there is more than one target_id, then we need to set it to multiple
    # peak_info.loc[peak_info['potential_target_count'] > 1,'potential_target_id'] = 'WARNING multiple targets matched'

    # print the number of peaks that were matched to multiple targets
    print(f'num peaks matched to multiple targets: {peak_info.loc[peak_info["potential_target_count"] > 1].shape[0]}')
    true_pos_percent = number_matched_targets/num_exp_targets
    return peak_info, true_pos_percent



###################
## Process the input target data file
###################

def process_targeted_data(targeted_data_path):
    # process the targeted data into a dataframe with a consistent format
    if (targeted_data_path is None):
        return pd.DataFrame()
    if not (os.path.exists(targeted_data_path)):
        return pd.DataFrame()
    
    input_targets_df = read_tabular_file(targeted_data_path)
    # lower the column names
    input_targets_df.columns = [x.lower() for x in input_targets_df.columns]
    # replace underscores with spaces
    input_targets_df.columns = [x.replace('_',' ') for x in input_targets_df.columns]

    # check if the targeted data has the right columns
    accept_mz_col_strs = ['mzmed','mz','mz_avg','mz_med','mz_mean','mz_centroid','m/z','target_mz','mz_target','M/Z RATIO']
    accept_rt_col_strs = ['rtmed','rt','rt_avg','rt_med','rt_mean','rt_centroid','rt (sec)','target_rt','rt_target','RETENTIONTIME/INDEX','RETENTIONTIME',
                          'retention time','retention time (sec)','retention time (s)','retention time (seconds)']
    accept_name_col_strs = ['target name','name','metabolite name','target','compound name','compound','metabolite','molecule name']
    accept_other_cols_strs = ['alt_name','metabolite_id','chemical_id','adduct','annotation','species','RefMet Name/Standardized Name',	'WorkBench Metabolite ID',
                              'RefMet Name','Standardized Name','database_identifier','Molecule Formula']
    
    accept_mz_col_strs = [x.lower() for x in accept_mz_col_strs]
    accept_mz_col_strs = [x.replace('_',' ') for x in accept_mz_col_strs]
    
    accept_rt_col_strs = [x.lower() for x in accept_rt_col_strs]
    accept_rt_col_strs = [x.replace('_',' ') for x in accept_rt_col_strs]
    
    accept_name_col_strs = [x.lower() for x in accept_name_col_strs]
    accept_name_col_strs = [x.replace('_',' ') for x in accept_name_col_strs]

    accept_other_cols_strs = [x.lower() for x in accept_other_cols_strs]
    accept_other_cols_strs = [x.replace('_',' ') for x in accept_other_cols_strs]
    # if the rt column is not found, check for the other rt columns
    other_rt_col_str = ['rt (min)', 'rt [min]', 'RT (min)','retention time (min)']


    # find the first columns that has a match in the above lists
    target_mz_col = None
    target_rt_col = None
    target_name_col = None
    for col in input_targets_df.columns:
        if target_mz_col is None:
            if col.lower() in accept_mz_col_strs:
                target_mz_col = col
                input_targets_df[col] = pd.to_numeric(input_targets_df[col], errors='coerce')
        if target_rt_col is None:
            if col.lower() in accept_rt_col_strs:
                target_rt_col = col
                input_targets_df[col] = pd.to_numeric(input_targets_df[col], errors='coerce')
        if target_name_col is None:
            if col.lower() in accept_name_col_strs:
                target_name_col = col

    if target_rt_col is None:
        target_rt_alt_col = None
        # check the other rt columns
        for col in input_targets_df.columns:
            if target_rt_alt_col is None:
                if col.lower() in other_rt_col_str:
                    target_rt_alt_col = col
        if target_rt_alt_col is not None:
            print(f'using the {target_rt_alt_col} to create an rt (sec) column')
            col = target_rt_alt_col
            input_targets_df[col] = pd.to_numeric(input_targets_df[col], errors='coerce')
            input_targets_df['rt (sec)'] = input_targets_df[target_rt_alt_col]*60
            # input_targets_df.to_csv(targeted_data_path,index=False)
            target_rt_col = 'rt (sec)'

    # convert to floats 

    if target_rt_col is not None:
        # check if the rt column is in seconds
        if 'sec' in target_rt_col.lower():
            print(f'Based on the column name, the units of {target_rt_col} is in seconds')
        elif input_targets_df[target_rt_col].max() < 60:
            print(f'The highest rt value is less than 60, so assuming that the units of {target_rt_col} is in minutes')
            input_targets_df['rt (sec)'] = input_targets_df[target_rt_col]*60
            target_rt_col = 'rt (sec)'
        else:
            print(f'assume that the units of {target_rt_col} is in seconds')

    if (target_mz_col is None) or (target_rt_col is None):
        print('targeted data does not have the right columns')
        print(input_targets_df.columns)
        return pd.DataFrame()

    if target_name_col is None:
        #if there is no target name columns, create one with 'target_number'
        target_name_col = 'target_name'
        input_targets_df[target_name_col] = ['target_'+str(x) for x in range(len(input_targets_df))]

    # create a new dataframe with the right columns
    targets_df = input_targets_df[[target_mz_col,target_rt_col,target_name_col]].copy()
    targets_df.rename(columns={target_mz_col:'mzmed',target_rt_col:'rtmed',target_name_col:'target_name'}, inplace=True)
    # add the other columns if they exist
    for col in input_targets_df.columns:
        if col.lower() in accept_other_cols_strs:
            targets_df[col.lower()] = input_targets_df[col]

    targets_df.index = targets_df['target_name']
    # drop na
    targets_df.dropna(inplace=True,subset=['mzmed'])
    print('number of targets after dropping missing mz values:',len(targets_df))
    return targets_df


########### 
## Helpers
###########

def load_params_from_file(csv_path,verbose=True):
    loaded_params = pd.read_csv(csv_path)
    dct = {}
    for i in range(loaded_params.shape[0]):
        row = loaded_params.iloc[i]
        dct[row[0]] = row[1]

    if verbose: print(f'loaded previously-derived params from {csv_path}')
    return dct

def load_params_from_file_improved(csv_path,verbose=True):
    loaded_params = pd.read_csv(csv_path)
    loaded_params.dropna(inplace=True)
    dct = {}
    for i in range(loaded_params.shape[0]):
        row = loaded_params.iloc[i]
        if row[1]=='True':
             dct[row[0]] = True
        elif row[1] == 'False':
            dct[row[0]] = False
        else:
            dct[row[0]] = pd.to_numeric(row[1],errors='ignore')

    for key in dct.keys():
        if isinstance(dct[key],np.int64):
            dct[key] = int(dct[key])
        elif isinstance(dct[key],np.float64):
            dct[key] = float(dct[key])

    if verbose: print(f'loaded previously-derived params from {csv_path}')
    return dct    

def read_tabular_file(file_path, delimiter=','):
    """
    Reads a tabular file (CSV/TSV) into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the tabular file.
    delimiter (str): The delimiter used in the file (default is ',').

    Returns:
    pandas.DataFrame: The tabular data as a DataFrame.
    """
    if file_path.endswith('.tsv'):
        delimiter = '\t'
    df = pd.read_csv(file_path, delimiter=delimiter)
    return df