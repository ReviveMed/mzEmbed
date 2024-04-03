import neptune


NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

from neptune.exceptions import NeptuneException


def count_fields(d):
    return sum([count_fields(v) if isinstance(v, dict)
                     else 1 for v in d.values()])

def get_num_fields_in_run(run_id):
    # https://docs.neptune.ai/help/error_field_count_limit_exceeded/
    #  The limit for a single run or other Neptune object is 9000 unique fields.
    run = neptune.init_run(project='revivemed/RCC',
        api_token= NEPTUNE_API_TOKEN,
        with_id=run_id,
        mode="read-only")    
    
    num_fields = count_fields(run.get_structure())
    return num_fields

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

def get_run_id_list(encoder_kind='AE',tags=['v3.1']):

    project = neptune.init_project(
        project='revivemed/RCC',
        mode="read-only",
        api_token=NEPTUNE_API_TOKEN
    )

    runs_table_df = project.fetch_runs_table(tag=tags,state='inactive').to_pandas()

    #drop the failed runs
    runs_table_df = runs_table_df[~runs_table_df['sys/failed']].copy()

    #filter by encoder_kind
    runs_table_df = runs_table_df[runs_table_df['pretrain/kwargs/encoder_kind'] == encoder_kind].copy()

    run_id_list = runs_table_df['sys/id'].tolist()
    return run_id_list



def check_neptune_existance(run,attribute):
    try:
        run[attribute].fetch()
        return True
    except:
        return False
    


def start_neptune_run(with_run_id=None,tags=['v3.1']):
    is_run_new = False
    if with_run_id is None:
        run = neptune.init_run(project='revivemed/RCC',
            api_token=NEPTUNE_API_TOKEN,
            tags=tags)
        is_run_new = True
    else:
        try:
            run = neptune.init_run(project='revivemed/RCC',
                                   api_token=NEPTUNE_API_TOKEN,
                                   with_id=with_run_id)
            print('Continuing run:', with_run_id)
            # add tags to the run
            run["sys/tags"].add(tags[0])

            #check if 'setup_id' exists in the run, current code doesnt work
            # if f'{setup_id}/kwargs' in run:
                # print(f'{setup_id} already exists in run:', with_run_id)
                # return with_run_id

        except NeptuneException:
            print('RunNotFound')
            run = neptune.init_run(project='revivemed/RCC',
                api_token=NEPTUNE_API_TOKEN,
                # custom_run_id=with_run_id,
                tags=tags)
            print('Starting new run:', run['sys/id'].fetch())
            is_run_new = True

    return run, is_run_new



######
if __name__ == '__main__':
    run_id = 'RCC-1603'
    num_fields = get_num_fields_in_run(run_id)
    print(num_fields)
