import neptune


NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

from neptune.exceptions import NeptuneException
from misc import unravel_dict, download_data_dir

import os
import shutil

def get_latest_dataset(data_dir='/DATA',project='revivemed/RCC',api_token=None):
    if api_token is None:
        api_token = NEPTUNE_API_TOKEN
    project = neptune.init_project(project=project, api_token=api_token)
    data_url = project["dataset/latest_link"].fetch()
    latest_hash = project["dataset/latest"].fetch_hash()
    need_to_download = False

    if not os.path.exists(data_dir+'/hash.txt'):
        need_to_download = True
    else:
        with open(data_dir+'/hash.txt','r') as f:
            current_hash = f.read()
        if current_hash != latest_hash:
            need_to_download = True

    if need_to_download:
        # delete the old data
        if os.path.exists(data_dir):
            print('Deleting the old data')
            shutil.rmtree(data_dir, ignore_errors=True)

        os.makedirs(data_dir, exist_ok=True)
        with open(data_dir+'/hash.txt','w') as f:
            f.write(latest_hash)

        print('Downloading the latest dataset')
        download_data_dir(data_url, save_dir=data_dir)
    project.stop()
    return data_dir


def count_fields(d):
    return sum([count_fields(v) if isinstance(v, dict)
                     else 1 for v in d.values()])

def get_num_fields_in_run(run_id,project_id='revivemed/RCC'):
    # https://docs.neptune.ai/help/error_field_count_limit_exceeded/
    #  The limit for a single run or other Neptune object is 9000 unique fields.
    run = neptune.init_run(project=project_id,
        api_token= NEPTUNE_API_TOKEN,
        with_id=run_id,
        mode="read-only")    
    
    num_fields = count_fields(run.get_structure())
    return num_fields

def check_if_path_in_struc(run_struc,target_path):
    # check if the neptune file path is in the neptune run structure
    target_keys = target_path.split('/')
    for key in target_keys:
        if key in run_struc:
            run_struc = run_struc[key]
        else:
            return False
    return True

def get_sub_struc_from_path(run_struc,target_path):
    # check if the neptune file path is in the neptune run structure
    target_keys = target_path.split('/')
    run_substruc = run_struc
    for key in target_keys:
        if key in run_substruc:
            run_substruc = run_substruc[key]
        else:
            return {}
    return run_substruc

def convert_neptune_kwargs(kwargs):
    if isinstance(kwargs, dict):
        return {k: convert_neptune_kwargs(v) for k, v in kwargs.items()}
    elif isinstance(kwargs, str):
        try:
            return eval(kwargs)
        except:
            return kwargs
    else:
        return kwargs    


def get_run_id_list_from_query(query,limit=2000,project_id='revivemed/RCC'):
    project = neptune.init_project(
        project=project_id,
        mode="read-only",
        api_token=NEPTUNE_API_TOKEN
    )
    runs_table_df = project.fetch_runs_table(query=query,limit=limit).to_pandas()
    
    #drop the failed runs
    # runs_table_df = runs_table_df[~runs_table_df['sys/failed']].copy()
    # select only the inactive runs
    runs_table_df = runs_table_df[runs_table_df['sys/state']== 'Inactive'].copy()
    
    run_id_list = runs_table_df['sys/id'].tolist()
    project.stop()
    return run_id_list


def get_run_id_list(encoder_kind='AE',tag=None,project_id='revivemed/RCC'):

    project = neptune.init_project(
        project=project_id,
        mode="read-only",
        api_token=NEPTUNE_API_TOKEN
    )

    query = f'(`pretrain/original_kwargs/encoder_kind`:string = "{encoder_kind}") AND `sys/state`:experimentState = "inactive"'
    if tag is not None:
        query += f' AND`sys/tags`:stringSet CONTAINS "{tag}"'
    runs_table_df = project.fetch_runs_table(query=query,limit=2000).to_pandas()
    # runs_table_df = project.fetch_runs_table(tag=tags,state='inactive').to_pandas()

    #drop the failed runs
    runs_table_df = runs_table_df[~runs_table_df['sys/failed']].copy()

    #filter by encoder_kind
    # runs_table_df = runs_table_df[runs_table_df['pretrain/original_kwargs/encoder_kind'] == encoder_kind].copy()

    run_id_list = runs_table_df['sys/id'].tolist()
    project.stop()
    return run_id_list



def check_neptune_existance(run,attribute):
    try:
        run[attribute].fetch()
        return True
    except:
        return False
    


def start_neptune_run(with_run_id=None,tags=['v3.3'],
                      yes_logging=False,
                      api_token = None,
                      project_id = 'revivemed/RCC',
                      neptune_mode='async'):
    is_run_new = False
    if api_token is None:
        api_token = NEPTUNE_API_TOKEN
    if with_run_id is None:
        run = neptune.init_run(project=project_id,
            api_token=api_token,
            tags=tags)
        is_run_new = True
    else:
        try:
            run = neptune.init_run(project=project_id,
                                   api_token=api_token,
                                   with_id=with_run_id,
                                   mode=neptune_mode,
                                    capture_stdout=yes_logging,
                                    capture_stderr=yes_logging,
                                    capture_hardware_metrics=yes_logging)
            print('Continuing run:', with_run_id)
            # add tags to the run
            run["sys/tags"].add(tags[0])

            #check if 'setup_id' exists in the run, current code doesnt work
            # if f'{setup_id}/original_kwargs' in run:
                # print(f'{setup_id} already exists in run:', with_run_id)
                # return with_run_id

        except NeptuneException:
            print('RunNotFound')
            run = neptune.init_run(project=project_id,
                api_token=api_token,
                # custom_run_id=with_run_id,
                capture_stdout=yes_logging,
                capture_stderr=yes_logging,
                capture_hardware_metrics=yes_logging,
                tags=tags)
            print('Starting new run:', run['sys/id'].fetch())
            is_run_new = True

    return run, is_run_new



def neptunize_dict_keys(eval_res,prefix='/'):
    return unravel_dict(eval_res,prefix,sep='_')
    # new_dict = {}
    # for key, val in eval_res.items():
    #     if isinstance(val,dict):
    #         if prefix is None:
    #             prefix = key
    #         else:
    #             prefix = prefix + '/' + key
    #         new_dict.update(neptunize_dict_keys(val,prefix))
    #     else:
    #         new_dict[prefix + '/' + key] = val
    # return new_dict


######
if __name__ == '__main__':
    run_id = 'RCC-3011'
    num_fields = get_num_fields_in_run(run_id)
    print(num_fields)
