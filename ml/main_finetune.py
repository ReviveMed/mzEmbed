from utils_finetune import *
from utils_neptune import get_latest_dataset


NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

PROJECT_ID = 'revivemed/Survival-RCC'


# Main for Surival Finetuning
def main():

    home_dir = os.path.expanduser("~")
    data_dir = f'{home_dir}/DATA3'
    output_save_dir = f'{home_dir}/OUTPUT'
    print('Getting the latest dataset')
    data_dir = get_latest_dataset(data_dir=data_dir,project=PROJECT_ID)
    
    
    with_id = None
    sweep_kwargs = parse_sweep_kwargs_from_command_line()

    # run_dict = run['training_run']

    desc_str_list = [
        'Both-OS',
        'NIVO-OS',
        'EVER-OS',
        'NIVO-OS AND EVER-OS',
        # 'NIVO-OS ADV EVER-OS',
    ]

    for desc_str in desc_str_list:
        print(desc_str)

        params = get_params(desc_str,sweep_kwargs=sweep_kwargs)

        eval_params_list = [
            {},
            {
                'y_col_name':'NIVO OS',
                'y_head':'EVER OS', # which head to apply to the y_col
                'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col
            {
                'y_col_name':'EVER OS',
                'y_head':'NIVO OS', # which head to apply to the y_col
                'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
            {
                'y_col_name':'NIVO OS',
                'y_head':'Both OS', # which head to apply to the y_col
                'y_cols': ['NIVO OS','OS_Event']}, # which columns to use for the y_col
            {
                'y_col_name':'EVER OS',
                'y_head':'Both OS', # which head to apply to the y_col
                'y_cols': ['EVER OS','OS_Event']}, # which columns to use for the y_col
        ]


        run = neptune.init_run(project=PROJECT_ID,
                                    api_token=NEPTUNE_API_TOKEN,
                                    with_id=with_id)

        run_id = run["sys/id"].fetch()
        output_dir = f'{output_save_dir}/{run_id}'
        os.makedirs(output_dir,exist_ok=True)

        run['params'] = params
        run['desc_str'] = desc_str
        run['model_name'] = 'SurvivalNet'
        run['pretrained_model'] = 'Model2925'
        run['dataset'].track_files(data_dir)

        record_train_metrics = defaultdict(list)
        record_test_metrics = defaultdict(list)
        for iter in range(5):
            train_metrics = run_model_wrapper(data_dir,params,
                            output_dir=output_dir,
                            train_name='train',
                            prefix=f'training_finetune_{iter}', 
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

            test_metrics = run_model_wrapper(data_dir,params,
                            output_dir=output_dir,
                            train_name='trainval',
                            prefix=f'testing_finetune_{iter}', 
                            eval_name_list=['test','trainval'],
                            eval_params_list=eval_params_list,
                            run_dict=run)
            
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

        for key,val in record_train_metrics.items():
            run[f'avg_training_finetune/metrics/{key}'] = np.mean(val)

        for key,val in record_test_metrics.items():
            run[f'avg_testing_finetune/metrics/{key}'] = np.mean(val)

        ### Repeat for Random Initialization ###
        sweep_kwargs['random_init'] = True
        params = get_params(desc_str,sweep_kwargs=sweep_kwargs)
        record_train_metrics = defaultdict(list)
        num_train_success = 0
        record_test_metrics = defaultdict(list)
        num_test_success = 0
        for iter in range(5):
            try:
                train_metrics = run_model_wrapper(data_dir,params,
                                output_dir=output_dir,
                                train_name='train',
                                prefix=f'training_randinit_{iter}', 
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
                                prefix=f'testing_randinit_{iter}', 
                                eval_name_list=['test','trainval'],
                                eval_params_list=eval_params_list,
                                run_dict=run)
                
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

        for key,val in record_train_metrics.items():
            run[f'avg_training_randinit/metrics/{key}'] = np.mean(val)
        run[f'avg_training_randinit/num_success'] = num_train_success

        for key,val in record_test_metrics.items():
            run[f'avg_testing_randinit/metrics/{key}'] = np.mean(val)
        run[f'avg_testing_randinit/num_success'] = num_test_success


        run.stop()

    print('Done')


if __name__ == '__main__':

    main()