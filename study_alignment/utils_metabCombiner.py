
from mspeaks import MSPeaks
import os
import pandas as pd
import json
import numpy as np
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
path_to_r_script = dir_path + "/metabCombiner_wrapper.R"

def get_metabCombiner_format(study_peaks,save_dir=None,study_name='study'):
    assert isinstance(study_peaks,MSPeaks)
    # peaks_df = pd.DataFrame()
    # peaks_df['feats'] =  study_peaks.peak_info.index
    # peaks_df['mzmed'] =  study_peaks.peak_info['mz'].values
    # peaks_df['rtmed'] =  study_peaks.peak_info['rt'].values/60 # convert to minutes
    peaks_df = study_peaks.peak_info[['rtmed','mzmed']].copy()
    peaks_df['rtmed'] = peaks_df['rtmed']/60 # convert to minutes
    peak_intensity = study_peaks.peak_intensity
    # peak_intensity.columns = ['sample_'+str(i) for i in range(peak_intensity.shape[1])]
    peak_intensity.fillna(0,inplace=True)
    peaks_df = pd.concat([peaks_df,peak_intensity],axis=1)
    if save_dir is not None:
        peaks_df.to_csv(os.path.join(save_dir,f'{study_name}_peak_df.txt'),sep='\t',index=True)

    return peaks_df


def align_ms_studies_with_metabCombiner(origin_study,input_study,
                                        origin_name='origin',input_name='input',
                                        save_dir=None,alignment_params=None):

    tmp_dir = os.makedirs('tmp', exist_ok=True)
    tmp_data1_path = os.path.join('tmp', 'data1.txt')
    tmp_data2_path = os.path.join('tmp', 'data2.txt')
    tmp_output_path = os.path.join('tmp', 'output.csv')
    
    if save_dir is not None:
        output_path = os.path.join(save_dir, f'{input_name}_aligned_to_{origin_name}_match_ids.csv')

    if save_dir is not None:
        with open(os.path.join(save_dir,f'{input_name}_aligned_to_{origin_name}_metabCombine_params.json'), 'w') as fp:
            json.dump(alignment_params, fp)

    if alignment_params is not None:
        raise NotImplementedError('alignment_params for metabCombiner is not implemented yet')

    origin_df = get_metabCombiner_format(origin_study)
    origin_df.to_csv(tmp_data1_path,sep='\t',index=True)

    input_df = get_metabCombiner_format(input_study)
    input_df.to_csv(tmp_data2_path,sep='\t',index=True)

    # I still need to change the overall RT min and max
    os.system("Rscript " + path_to_r_script + " 'tmp/data1.txt' 'tmp/data2.txt' 'tmp/output.csv' FALSE")

    if not os.path.exists(tmp_output_path):
        shutil.rmtree('tmp')
        raise ValueError('metabCombiner did not run successfully')

    # copy the output file to the output directory
    if save_dir is not None:
        shutil.copyfile(tmp_output_path, output_path)

    align_out = pd.read_csv(tmp_output_path)
    align_out.columns = [origin_name,input_name,'score']
    # delete the tmp directory
    shutil.rmtree('tmp')
    return align_out