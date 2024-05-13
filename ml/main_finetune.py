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
    
    # run[f'all_training_{prefix_name}/metrics/{key}'] = []
    for key,val in record_train_metrics.items():
        # run[f'all_training_{prefix_name}/metrics/{key}'].extend(val)
        run[f'avg_training_{prefix_name}/metrics/{key}'] = np.mean(val)
        run[f'avg_training_{prefix_name}/metrics/std_{key}'] = np.std(val)
    run[f'avg_training_{prefix_name}/num_success'] = num_train_success

    # run[f'all_testing_{prefix_name}/metrics/{key}'] = []
    for key,val in record_test_metrics.items():
        # run[f'all_testing_{prefix_name}/metrics/{key}'].extend(val)
        run[f'avg_testing_{prefix_name}/metrics/{key}'] = np.mean(val)
        run[f'avg_testing_{prefix_name}/metrics/std_{key}'] = np.std(val)
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


def main0(user_kwargs):
    home_dir = os.path.expanduser("~")
    data_dir = f'{home_dir}/DATA3'
    output_save_dir = f'{home_dir}/OUTPUT'
    print('Getting the latest dataset')
    data_dir = get_latest_dataset(data_dir=data_dir,project=PROJECT_ID)
    desc_str = user_kwargs.get('desc_str',None)
    with_id = user_kwargs.get('with_id',None)
    if with_id is not None:
        with_id = 'SUR-'+str(with_id)
    eval_params_list = default_eval_params_list

    num_iterations = user_kwargs.get('num_iterations',20)

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
        params = get_params(desc_str,sweep_kwargs=user_kwargs)
        run['sweep_kwargs'] = stringify_unsupported(user_kwargs)
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



def parse_sweep_kwargs_from_command_line2():

    parser = argparse.ArgumentParser(description='Parse command line arguments for sweep kwargs')
    parser.add_argument('--use_randinit', action='store_true', help='Use random initialization')
    parser.add_argument('--holdout_frac', type=float, default=0, help='Holdout fraction')
    parser.add_argument('--head_hidden_layers', type=int, default=0, help='Number of hidden layers in the head')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--l2_reg_weight', type=float, default=0.0, help='L2 regularization weight')
    parser.add_argument('--l1_reg_weight', type=float, default=0.0, help='L1 regularization weight')
    parser.add_argument('--noise_factor', type=float, default=0.1, help='Noise factor')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--adversary_weight', type=float, default=1, help='Adversary Task weight')
    parser.add_argument('--auxillary_weight', type=float, default=1, help='Auxillary Task weight')
    parser.add_argument('--adversarial_start_epoch', type=int, default=10, help='Adversarial start epoch')
    parser.add_argument('--encoder_weight', type=float, default=0, help='Encoder weight')
    # parser.add_argument('--clip_grads', action='store_true', help='Clip gradients with norm')
    parser.add_argument('--no_clip_grads', action='store_false', help='Clip gradients with norm')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--remove_nans', action='store_false', help='Remove rows with NaNs in the y-data')
    parser.add_argument('--train_name', type=str, default='train', help='Training name')
    parser.add_argument('--desc_str', type=str, help='Description string', nargs='?')
    parser.add_argument('--with_id', type=int, help='Include the ID in the description string', nargs='?')
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations for the sweep')

    args = parser.parse_args()

    sweep_kwargs = {
        'use_rand_init': args.use_randinit,
        'holdout_frac': args.holdout_frac,
        'head_hidden_layers': args.head_hidden_layers,
        'dropout_rate': args.dropout_rate,
        'num_epochs': args.num_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'learning_rate': args.learning_rate,
        'l2_reg_weight': args.l2_reg_weight,
        'l1_reg_weight': args.l1_reg_weight,
        'noise_factor': args.noise_factor,
        'weight_decay': args.weight_decay,
        'adversary_weight': args.adversary_weight,
        'auxillary_weight': args.auxillary_weight,
        'adversarial_start_epoch': args.adversarial_start_epoch,
        'encoder_weight': args.encoder_weight,
        # 'clip_grads_with_norm': args.clip_grads,
        'clip_grads_with_norm': args.no_clip_grads,
        'batch_size': args.batch_size,
        'train_name': args.train_name,
        'num_iterations': args.num_iterations,
    }

    if args.desc_str is not None:
        sweep_kwargs['desc_str'] = args.desc_str
    if args.with_id is not None:
        sweep_kwargs['with_id'] = args.with_id

    return sweep_kwargs


if __name__ == '__main__':

    method1 = {
        'noise_factor': 0.25,
        'learning_rate': 0.0007869775056037999,
        'l2_reg_weight': 1.0092405183765013e-05,
        'l1_reg_weight': 3.137204254745065e-05,
        'num_epochs': 87,
        'head_hidden_layers': 1,
        'EVER-OS__weight': 10.0,
        'NIVO-OS__weight': 5.0
    }

    method2 = {
        'noise_factor': 0.1,
        'learning_rate': 0.0006221023998363983,
        'l2_reg_weight': 0.0,
        'l1_reg_weight': 0.0025635844524779894,
        'num_epochs': 93,
        'encoder_weight': 1.0,
        'dropout_rate': 0.3,
        'adversarial_start_epoch': 0,
        'adversary_weight': 1.0,
        'head_hidden_layers': 1,
        'EVER-OS__weight': 10.0,
        'NIVO-OS__weight': 4.0
    }

    method3 = {
        'dropout_rate': 0.0,
        'encoder_weight': 0.0,
        'learning_rate': 0.000541422,
        'noise_factor': 0.2,
        'num_epochs': 71,
        'batch_size': 64,
        'head_hidden_layers': 0
    }

    method4 = {
        'dropout_rate': 0.0,
        'encoder_weight': 0.0,
        'learning_rate': 0.000541422,
        'noise_factor': 0.2,
        'num_epochs': 71,
        'batch_size': 32,
        'head_hidden_layers': 0
    }

    method5 = {
        'dropout_rate': 0.0,
        'encoder_weight': 0.0,
        'learning_rate': 0.000541422,
        'noise_factor': 0.2,
        'num_epochs': 71,
        'batch_size': 32,
        'head_hidden_layers': 0,
        'remove_nans': True
    }

    method6 = {
        'dropout_rate': 0.0,
        'encoder_weight': 0.0,
        'learning_rate': 0.0013,
        'noise_factor': 0.2,
        'num_epochs': 98,
        'batch_size': 32,
        'head_hidden_layers': 0,
        'weight_decay': 0.000016,
    }

    method7 = {
        'dropout_rate': 0.0,
        'encoder_weight': 0.0,
        'learning_rate': 0.0013,
        'noise_factor': 0.2,
        'num_epochs': 98,
        'batch_size': 32,
        'head_hidden_layers': 0,
        'weight_decay': 0.000016,
        'remove_nans': True,
    }

    method8 = {
        'dropout_rate': 0.4,
        'encoder_weight': 0.75,
        'learning_rate': 0.0021,
        'noise_factor': 0.25,
        'num_epochs': 38,
        'batch_size': 32,
        'head_hidden_layers': 0,
        'weight_decay': 0.0,
        'remove_nans': True,
    }

    method9 = {
        'dropout_rate': 0.4,
        'encoder_weight': 0.75,
        'learning_rate': 0.0021,
        'noise_factor': 0.25,
        'num_epochs': 38,
        'batch_size': 32,
        'head_hidden_layers': 0,
        'weight_decay': 0.0,
        'remove_nans': True,
    }

    # user_kwargs = parse_sweep_kwargs_from_command_line()

    for method in  [method1, method2, method3, method4, method5, method6, method7, method8, method9]:
        for desc_str in ['Both-OS','NIVO-OS','EVER-OS','NIVO-OS AND EVER-OS','IMDC','MSKCC','NIVO-OS ADV EVER-OS']:
            for use_randinit in [True,False]:
                user_kwargs = {k:v for k,v in method.items()}
                user_kwargs['use_rand_init'] = use_randinit
                user_kwargs['desc_str'] = desc_str
                main0(user_kwargs)

    # main()
    # main2()