

import os
import pandas as pd
import json
import numpy as np
import shutil
from .mspeaks import MSPeaks
# from mspeaks import MSPeaks
import itertools

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
    tmp_params_path = os.path.join('tmp', 'params.json')
    
    if save_dir is not None:
        output_path = os.path.join(save_dir, f'{input_name}_aligned_to_{origin_name}_match_ids.csv')

    if alignment_params is None:
        alignment_params = {}


    if origin_study.overall_RT_min:
        overall_RT_params = {
            'rtmin1' : origin_study.overall_RT_min/60,
            'rtmax1' : origin_study.overall_RT_max/60,
            'rtmin2' : input_study.overall_RT_min/60,
            'rtmax2' : input_study.overall_RT_max/60
        }

        alignment_params.update(overall_RT_params)

    if alignment_params is not None:
        # raise NotImplementedError('alignment_params for metabCombiner is not implemented yet')
        with open(tmp_params_path, 'w') as fp:
            json.dump(alignment_params, fp)

    if save_dir is not None:
        with open(os.path.join(save_dir,f'{input_name}_aligned_to_{origin_name}_metabCombine_params.json'), 'w') as fp:
            json.dump(alignment_params, fp)

    origin_df = get_metabCombiner_format(origin_study)
    origin_df.to_csv(tmp_data1_path,sep='\t',index=True)

    input_df = get_metabCombiner_format(input_study)
    input_df.to_csv(tmp_data2_path,sep='\t',index=True)

    # I still need to change the overall RT min and max
    os.system("Rscript " + path_to_r_script + " 'tmp/data1.txt' 'tmp/data2.txt' 'tmp/output.csv' 'tmp/params.json'")

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


import hashlib
import json

def create_hash(**kwargs):
    hash_object = hashlib.sha1(usedforsecurity=False)
    hash_object.update(json.dumps(kwargs, sort_keys=True).encode())
    return hash_object.hexdigest()

def create_metaCombiner_params(**kwargs):
    # create a unique hash based on the kwargs

    freq_th0 = kwargs.get('freq_th0',0.5)
    freq_th1 = kwargs.get('freq_th1',0.5)
    
    alignment_params = {}
    for key, value in kwargs.items():
        if key not in ['freq_th0','freq_th1']:
            alignment_params[key] = value
    hash_string = create_hash(**alignment_params)

    freq_th0_perc = int(100*freq_th0)
    freq_th1_perc = int(100*freq_th1)
    param_name = 'metabCombiner_{freq_th0}_{freq_th1}_{hash_string}'.format(freq_th0=freq_th0_perc,freq_th1=freq_th1_perc,hash_string=hash_string)
    alignment_method = 'metabCombiner'

    alignment_params = {
        'freq_th0' : freq_th0,
        'freq_th1' : freq_th1,
        'param_name' : param_name,
        'alignment_method' : alignment_method,
        'alignment_params' : {
            'binGap' : kwargs.get('binGap',0.005),
            'misspc1' : kwargs.get('misspc1',90),
            'misspc2' : kwargs.get('misspc2',90),
            'windx' : kwargs.get('windx',0.03),
            'windy' : kwargs.get('windy',0.03),
            'tolmz' : kwargs.get('tolmz',0.003),
            'tolQ' : kwargs.get('tolQ',0.3),
            'tolrtq' : kwargs.get('tolrtq',0.3),
        }
    }
    return alignment_params


def create_metaCombiner_grid_search(save_dir,**kwargs):
    param_name_list = []
    key_list = []
    value_list = []
    for key, value in kwargs.items():
        key_list.append(key)
        if isinstance(value, list):
            value_list.append(value)
        else:
            value_list.append([value])


    num_permutations = 1
    for i in range(len(value_list)):
        num_permutations *= len(value_list[i])
    print(f'Number of permutations: {num_permutations}')

    for values in itertools.product(*value_list):
        input_kwargs = dict(zip(key_list, values))

        alignment_params = create_metaCombiner_params(**input_kwargs)
        param_name_list.append(alignment_params['param_name'])
        with open(os.path.join(save_dir,f'{alignment_params["param_name"]}.json'), 'w') as fp:
            json.dump(alignment_params, fp)
            
    return param_name_list



# if __name__ == '__main__':
#    save_dir= '/Users/jonaheaton/Desktop/alignment_analysis/Alignment_Params'
#    create_metaCombiner_grid_search(save_dir=save_dir,
#                                    freq_th0 = [0.25,0.5],
#                                     freq_th1 = [0.25,0.5],
#                                     tolmz = [0.003,0.005],
#                                     windx = [0.01,0.03,0.05,0.1],
#                                     windy = [0.01,0.03,0.05,0.1])





