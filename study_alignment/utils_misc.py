###################
## Helper Functions
###################

import json
import os

###################
## Basic File I/O Functions
###################

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




###################
## Parameter File Cleaing and Organization Functions
###################

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