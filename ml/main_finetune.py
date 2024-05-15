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

    {
        'y_col_name':'MSKCC ORDINAL',
        'y_head':'MSKCC_Ordinal', # which head to apply to the y_col
        'y_cols': ['MSKCC ORDINAL']}, # which columns to use for the y_col
    {
        'y_col_name':'IMDC ORDINAL',
        'y_head':'IMDC_Ordinal', # which head to apply to the y_col
        'y_cols': ['IMDC ORDINAL']}, # which columns to use for the y_col

    {
        'y_col_name':'MSKCC ORDINAL',
        'y_head':'MSKCC_MultiClass', # which head to apply to the y_col
        'y_cols': ['MSKCC ORDINAL']}, # which columns to use for the y_col
    {
        'y_col_name':'IMDC ORDINAL',
        'y_head':'IMDC_MultiClass', # which head to apply to the y_col
        'y_cols': ['IMDC ORDINAL']}, # which columns to use for the y_col        

    {
        'y_col_name': 'Benefit',
        'y_head': 'Benefit',
        'y_cols': ['Benefit']},
]



def get_params(desc_str,sweep_kwargs=None):

    task_components_dict = parse_task_components_dict_from_str(desc_str,sweep_kwargs)

    train_kwargs_list = [
        'use_rand_init',
        'batch_size',
        'yes_clean_batches',
        'holdout_frac',
        'early_stopping_patience',
        'scheduler_kind',
        'num_epochs',
        'learning_rate',
        'noise_factor',
        'weight_decay',
        'l2_reg_weight',
        'l1_reg_weight',
        'dropout_rate',
        'adversarial_start_epoch',
        'clip_grads_with_norm',
        'encoder_weight',
        'head_weight',
        'adversary_weight',
        'remove_nans',
        'train_name'
    ]

    train_kwargs_dict = {k:v for k,v in sweep_kwargs.items() if k in train_kwargs_list}
    params = {
        'task_kwargs': task_components_dict,
        'train_kwargs': train_kwargs_dict
    }
    return params


def parse_sweep_kwargs_from_command_line():

    parser = argparse.ArgumentParser(description='Parse command line arguments for sweep kwargs')
    parser.add_argument('--use_randinit', action='store_true', help='Use random initialization')
    parser.add_argument('--holdout_frac', type=float, default=0, help='Holdout fraction')
    parser.add_argument('--head_hidden_layers', type=int, default=0, help='Number of hidden layers in the head')
    parser.add_argument('--dropout_rate', type=float, default=0, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--l2_reg_weight', type=float, default=0.0, help='L2 regularization weight')
    parser.add_argument('--l1_reg_weight', type=float, default=0.0, help='L1 regularization weight')
    parser.add_argument('--noise_factor', type=float, default=0.1, help='Noise factor')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--head_weight', type=float, default=1, help='Head Task weight')
    parser.add_argument('--adversary_weight', type=float, default=1, help='Adversary Task loss weight')
    parser.add_argument('--auxillary_weight', type=float, default=0.5, help='Auxillary Task loss weight')
    parser.add_argument('--encoder_weight', type=float, default=0, help='Encoder loss weight')
    parser.add_argument('--adversarial_start_epoch', type=int, default=0, help='Adversarial start epoch')
    # parser.add_argument('--clip_grads', action='store_true', help='Clip gradients with norm')
    parser.add_argument('--no_clip_grads', action='store_false', help='Clip gradients with norm')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--yes_clean_batches', action='store_true', help='Clean batch size so last batch is good size')
    parser.add_argument('--remove_nans', action='store_false', help='Remove rows with NaNs in the y-data')
    parser.add_argument('--train_name', type=str, default='train', help='Training name')
    parser.add_argument('--desc_str', type=str, help='Description string', nargs='?')
    parser.add_argument('--with_id', type=int, help='Include the ID in the description string', nargs='?')
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations for the sweep')
    parser.add_argument('--name', type=str, help='Name of the sweep', nargs='?')

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
        'head_weight': args.head_weight,
        'adversary_weight': args.adversary_weight,
        'auxillary_weight': args.auxillary_weight,
        'adversarial_start_epoch': args.adversarial_start_epoch,
        'encoder_weight': args.encoder_weight,
        # 'clip_grads_with_norm': args.clip_grads,
        'clip_grads_with_norm': args.no_clip_grads,
        'batch_size': args.batch_size,
        'yes_clean_batches': args.yes_clean_batches,
        'train_name': args.train_name,
        'num_iterations': args.num_iterations,
        'remove_nans': args.remove_nans,
        'name': args.name
    }

    if args.desc_str is not None:
        sweep_kwargs['desc_str'] = args.desc_str
    if args.with_id is not None:
        sweep_kwargs['with_id'] = args.with_id

    return sweep_kwargs

#######################################################
########### Main Function Wrappers  ###########
#######################################################


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

            if train_metrics is None:
                continue
            
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
    
    for iter in range(num_iterations):
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
def finetune_both_pretrain_and_rand_wrapper(desc_str_list=None):

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




def finetune_run_wrapper(**user_kwargs):
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

    num_iterations = user_kwargs.get('num_iterations',10)

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

    eval_params_list = [x for x in eval_params_list if x['y_cols'][0] in params['task_kwargs']['y_head_cols']]
    print('eval_params_list',eval_params_list)
    run_id = run["sys/id"].fetch()
    output_dir = f'{output_save_dir}/{run_id}'
    os.makedirs(output_dir,exist_ok=True)
    
    run = run_multiple_iterations(data_dir,params,output_dir,eval_params_list,run,prefix_name='run',num_iterations=num_iterations)
    run['sys/failed'] = False
    run.stop()
    return run_id



if __name__ == '__main__':

    user_kwargs = parse_sweep_kwargs_from_command_line()
    finetune_run_wrapper(**user_kwargs)
    # finetune_run_wrapper(with_id=2569)
    # finetune_run_wrapper(with_id=2590)
   