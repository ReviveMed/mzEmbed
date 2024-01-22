'''
Define the alignment function between two studies
'''



import os
import pandas as pd
import json
import numpy as np

from bmxp.eclipse import MSAligner
from mspeaks import MSPeaks

# %%

def get_eclipse_format(study_peaks,study_name,log_intensity=False):
    assert isinstance(study_peaks,MSPeaks)
    eclipse_df = pd.DataFrame()
    eclipse_df['Compound_ID'] =  study_peaks.peak_info.index
    eclipse_df['MZ'] =  study_peaks.peak_info['mz'].values
    eclipse_df['RT'] =  study_peaks.peak_info['rt'].values/60 # convert to minutes
    # if scale_rt:
        # coor_info['RT'] = coor_info['RT']*10/coor_info['RT'].max()
    if log_intensity:
        eclipse_df['Intensity'] = np.log10(study_peaks.peak_intensity.median(axis=1).values)
    else:
        eclipse_df['Intensity'] = study_peaks.peak_intensity.median(axis=1).values
    # add information about the targets
    for col in study_peaks.peak_info.columns:
        if 'target' in col:
            eclipse_df[study_name+'_'+col] = study_peaks.peak_info[col].values
    return eclipse_df



def align_ms_studies(origin_study,input_study,origin_name='origin',input_name='input',
                     alignment_params=None,save_dir=None,log_intensity=False, clean_output=False):
    #align the input study to the origin study
    # origin_study: MSPeaks object
    # input_study: MSPeaks object

    os.makedirs('tmp',exist_ok=True)

    origin_df = get_eclipse_format(origin_study,origin_name,log_intensity=log_intensity)
    origin_df.to_csv('tmp/origin.csv')

    input_df = get_eclipse_format(input_study,input_name,log_intensity=log_intensity)
    input_df.to_csv('tmp/input.csv')

    csv_list = ['tmp/origin.csv','tmp/input.csv']
    a = MSAligner(*csv_list, names=[origin_name,input_name]) # Initialization Phase

    # a.set_defaults( {'weights':
        # {"RT": 1, "Intensity": 1, "MZ": 5}})
    if alignment_params is None:
        alignment_params = {
            # 'scalers': 
            'cutoffs': {"RT": 6, "Intensity": 6, "MZ": 6},
            'weights': {"RT": 1, "Intensity": 0.5, "MZ": 1},
            'coarse_params': { "RT": {"upper": 7, "lower": -7}, 
                            "MZ": {"upper": 50, "lower": -50}, 
                            "Intensity": {"upper": 2, "lower": -2}}}
            # 'scaler_params':  {"smoothing_method": "lowess",
            # "smoothing_params": {"frac": 0.1}},

    if save_dir is not None:
        with open(os.path.join(save_dir,f'{input_name}_aligned_to_{origin_name}_params.json'), 'w') as fp:
            json.dump(alignment_params, fp)

    a.set_defaults(alignment_params)
    a.align() # Subalignment Phase
    # a.to_csv() # Feature Aggregation Phase
    # a.report() # Generate a report
    if save_dir is not None:
        a.to_csv(filepath=os.path.join(save_dir,f'{input_name}_aligned_to_{origin_name}.csv'))

    # remove the temp files
    os.remove('tmp/origin.csv')
    os.remove('tmp/input.csv')
    os.rmdir('tmp')

    if clean_output:
        alignment_result = a.results()
        align_out = alignment_result[['Compound_ID','Compound_ID_'+input_name]].copy()
        align_out.columns = [origin_name,input_name]
        return align_out
    else:
        return a


align_ms_studies_with_Eclipse = align_ms_studies

def combine_two_alignment_results(existing_align_df,align_df=None):
    if align_df is None:
        return existing_align_df
    kept_cols = [x for x in align_df.columns if 'Compound_ID' in x]
    align_df = align_df[kept_cols]
    align_df.set_index('Compound_ID',inplace=True)
    if existing_align_df is None:
        return align_df
    align_df = align_df.merge(existing_align_df,
                              left_index=True,
                              right_index=True,
                              how='outer')
    return align_df



def combine_alignments_in_dir(save_dir,origin_name='origin'):
    existing_align_df = None

    for file in os.listdir(save_dir):
        if f'aligned_to_{origin_name}.csv' in file:
            new_align_df = pd.read_csv(os.path.join(save_dir,file))
            existing_align_df = combine_two_alignment_results(existing_align_df,new_align_df)
    
    existing_align_df.to_csv(os.path.join(save_dir,f'combined_{origin_name}_alignments.csv'))
    return existing_align_df

# %%

if __name__ == "__main__":

# %%

    from mspeaks import create_mspeaks_from_mzlearn_result

    rcc4_dir = '/Users/jonaheaton/Desktop/Data-engine/ST001236_and_ST001237/rcc1_rcc3_output_v1_2023-09-28-21-22-21-'
    origin_study = create_mspeaks_from_mzlearn_result(rcc4_dir)

    other_dir = '/Users/jonaheaton/Desktop/Data-engine/ST001519/farmm_HP_Plasma_2023-09-12-21-06-54-'
    input_study = create_mspeaks_from_mzlearn_result(other_dir)
    # %%
    origin_study.apply_freq_th_on_peaks(0.6)
    # %%
    origin_study.num_peaks
    # %%
    input_study.apply_freq_th_on_peaks(0.2)
    # %%
