from utils_finetune import *
from utils_neptune import get_latest_dataset
from neptune.utils import stringify_unsupported


NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

PROJECT_ID = 'revivemed/Survival-RCC'


default_eval_params_list = [
    # {},
    {
        'y_col_name':'NIVO OS',
        'y_head':'NIVO OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'NIVO OS',
        'y_head':'EVER OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'NIVO OS',
        'y_head':'Both OS', # which head to apply to the y_col
        'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col


    {
        'y_col_name':'EVER OS',
        'y_head':'NIVO OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'EVER OS',
        'y_head':'EVER OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'EVER OS',
        'y_head':'Both OS', # which head to apply to the y_col
        'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
    

    {
        'y_col_name':'Both OS',
        'y_head':'EVER OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col
    {
        'y_col_name':'Both OS',
        'y_head':'NIVO OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col   
    {
        'y_col_name':'Both OS',
        'y_head':'Both OS', # which head to apply to the y_col
        'y_cols': ['OS','OS_Event']}, # which columns to use for the y_col             
    

    {
        'y_col_name':'MSKCC BINARY',
        'y_head':'MSKCC', # which head to apply to the y_col
        'y_cols': ['MSKCC BINARY']}, # which columns to use for the y_col
    {
        'y_col_name':'IMDC BINARY',
        'y_head':'IMDC', # which head to apply to the y_col
        'y_cols': ['IMDC BINARY']}, # which columns to use for the y_col
]


def run_multiple_iterations(data_dir,params,output_dir,eval_params_list,run,prefix_name,num_iterations=10):

    record_train_metrics = defaultdict(list)
    num_train_success = 0
    record_test_metrics = defaultdict(list)
    num_test_success = 0
    for iter in range(num_iterations):
        try:
            train_metrics = run_model_wrapper(data_dir,params,
                            output_dir=output_dir,
                            train_name='train',
                            prefix=f'training_{prefix_name}_{iter}', 
                            eval_name_list=['val','train'],
                            eval_params_list=eval_params_list,
                            run_dict=run)

            for key,val in train_metrics.items():
                if isinstance(val,dict):
                    for k,v in val.items():
                        if isinstance(v,dict):
                            for kk,vv in v.items():
                                record_train_metrics[key+'_'+k+'_'+kk].append(vv)
                        else:
                            record_train_metrics[key+'_'+k].append(v)
                else:
                    record_train_metrics[key].append(val)
            num_train_success += 1
        except ValueError as e:
            print(e)
            
        try:
            
            test_metrics = run_model_wrapper(data_dir,params,
                            output_dir=output_dir,
                            train_name='trainval',
                            prefix=f'testing_{prefix_name}_{iter}', 
                            eval_name_list=['test','trainval'],
                            eval_params_list=eval_params_list,
                            run_dict=run)
            
            if test_metrics is None:
                continue

            for key,val in test_metrics.items():
                if isinstance(val,dict):
                    for k,v in val.items():
                        if isinstance(v,dict):
                            for kk,vv in v.items():
                                record_test_metrics[key+'_'+k+'_'+kk].append(vv)
                        else:
                            record_test_metrics[key+'_'+k].append(v)
                else:
                    record_test_metrics[key].append(val)
            num_test_success += 1
        except ValueError as e:
            print(e)
    
    run[f'all_training_{prefix_name}/metrics/{key}'] = []
    for key,val in record_train_metrics.items():
        run[f'all_training_{prefix_name}/metrics/{key}'].extend(val)
        run[f'avg_training_{prefix_name}/metrics/{key}'] = np.mean(val)
    run[f'avg_training_{prefix_name}/num_success'] = num_train_success

    run[f'all_testing_{prefix_name}/metrics/{key}'] = []
    for key,val in record_test_metrics.items():
        run[f'all_testing_{prefix_name}/metrics/{key}'].extend(val)
        run[f'avg_testing_{prefix_name}/metrics/{key}'] = np.mean(val)
    run[f'avg_testing_{prefix_name}/num_success'] = num_test_success

    return run


# Main for Surival Finetuning
def main(desc_str_list=None):

    home_dir = os.path.expanduser("~")
    data_dir = f'{home_dir}/DATA3'
    output_save_dir = f'{home_dir}/OUTPUT'
    print('Getting the latest dataset')
    data_dir = get_latest_dataset(data_dir=data_dir,project=PROJECT_ID)
    
    eval_params_list = default_eval_params_list
    with_id = None
    original_sweep_kwargs = parse_sweep_kwargs_from_command_line()
    random_sweep_kwargs =  {k:v for k,v in original_sweep_kwargs.items()}
    random_sweep_kwargs['use_rand_init'] = True
    # run_dict = run['training_run']

    if desc_str_list is None:
        desc_str_list = [
            # 'Both-OS',
            # 'NIVO-OS',
            # 'EVER-OS',
            # 'NIVO-OS AND EVER-OS',
            'NIVO-OS ADV EVER-OS',
            'IMDC',
            'MSKCC',
        ]

    for desc_str in desc_str_list:
        print(desc_str)


        run = neptune.init_run(project=PROJECT_ID,
                                    api_token=NEPTUNE_API_TOKEN,
                                    with_id=with_id)

        run_id = run["sys/id"].fetch()
        output_dir = f'{output_save_dir}/{run_id}'
        os.makedirs(output_dir,exist_ok=True)
        
        params = get_params(desc_str,sweep_kwargs=original_sweep_kwargs)
        run['sweep_kwargs'] = stringify_unsupported(original_sweep_kwargs)
        run['params'] = stringify_unsupported(params)
        run['desc_str'] = desc_str
        run['model_name'] = 'SurvivalNet'
        run['pretrained_model'] = 'Model2925'
        run['dataset'].track_files(data_dir)

        print('RUN with PRETRAINED INITIALIZATION')
        run = run_multiple_iterations(data_dir,params,output_dir,eval_params_list,run,prefix_name='finetune')


        ### Repeat for Random Initialization ###
        print('RUN with RANDOM INITIALIZATION')
        params = get_params(desc_str,sweep_kwargs=random_sweep_kwargs)
        run = run_multiple_iterations(data_dir,params,output_dir,eval_params_list,run,prefix_name='randinit')

        run.stop()

    print('Done')


# Main for Surival Finetuning
def main2(run_id_list=None):

    home_dir = os.path.expanduser("~")
    data_dir = f'{home_dir}/DATA3'
    output_save_dir = f'{home_dir}/OUTPUT'
    print('Getting the latest dataset')
    data_dir = get_latest_dataset(data_dir=data_dir,project=PROJECT_ID)
    eval_params_list = default_eval_params_list
    
    with_id = None
    # run_dict = run['training_run']

    if run_id_list is None:
        run_id_list = [
            'SUR-2504',
            "SUR-2503",
            "SUR-2502",
            "SUR-2501" ]

    for with_id in run_id_list:
        print(with_id)

        run = neptune.init_run(project=PROJECT_ID,
                                    api_token=NEPTUNE_API_TOKEN,
                                    with_id=with_id)

        run_id = run["sys/id"].fetch()
        params = run['params'].fetch()
        params = convert_neptune_kwargs(params)
        output_dir = f'{output_save_dir}/{run_id}'
        os.makedirs(output_dir,exist_ok=True)


        print('RUN with PRETRAINED INITIALIZATION')
        run = run_multiple_iterations(data_dir,params,output_dir,eval_params_list,run,prefix_name='finetune')


        ### Repeat for Random Initialization ###
        print('RUN with RANDOM INITIALIZATION')
        params_w_randominit = {k:v for k,v in params.items()}
        params_w_randominit['train_kwargs']['use_rand_init'] = True
        run = run_multiple_iterations(data_dir,params,output_dir,eval_params_list,run,prefix_name='randinit')

        run.stop()

    print('Done')


def main0():
    home_dir = os.path.expanduser("~")
    data_dir = f'{home_dir}/DATA3'
    output_save_dir = f'{home_dir}/OUTPUT'
    print('Getting the latest dataset')
    data_dir = get_latest_dataset(data_dir=data_dir,project=PROJECT_ID)
    original_sweep_kwargs = parse_sweep_kwargs_from_command_line()
    desc_str = original_sweep_kwargs.get('desc_str',None)
    with_id = original_sweep_kwargs.get('with_id',None)
    if with_id is not None:
        with_id = 'SUR-'+str(with_id)
    eval_params_list = default_eval_params_list

    num_iterations = original_sweep_kwargs.get('num_iterations',10)

    run = neptune.init_run(project=PROJECT_ID,
                                    api_token=NEPTUNE_API_TOKEN,
                                    with_id=with_id)

    if (desc_str is None) and (with_id is None):
        desc_str = 'Both-OS'
    if with_id:
        desc_str = run['desc_str'].fetch()
        params = run['params'].fetch()
        params = convert_neptune_kwargs(params)
        original_sweep_kwargs = run['sweep_kwargs'].fetch()
        original_sweep_kwargs = convert_neptune_kwargs(original_sweep_kwargs)
    else:
        params = get_params(desc_str,sweep_kwargs=original_sweep_kwargs)
        run['sweep_kwargs'] = stringify_unsupported(original_sweep_kwargs)
        run['params'] = stringify_unsupported(params)
        run['desc_str'] = desc_str
        run['model_name'] = 'SurvivalNet'
        run['pretrained_model'] = 'Model2925'
        run['dataset'].track_files(data_dir)

    run_id = run["sys/id"].fetch()
    output_dir = f'{output_save_dir}/{run_id}'
    os.makedirs(output_dir,exist_ok=True)
    
    run = run_multiple_iterations(data_dir,params,output_dir,eval_params_list,run,prefix_name='run',num_iterations=num_iterations)
    run.stop()
    return run_id



if __name__ == '__main__':

    main0()
    # main()
    # main2()