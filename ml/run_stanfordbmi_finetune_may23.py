
import os
import optuna
from main_finetune import finetune_run_wrapper
from misc import round_to_sig
import numpy as np
from misc import download_data_dir
import neptune

from utils_neptune import convert_neptune_kwargs

lung_cancer_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/stanford-hmp2'
data_dir = os.path.join(lung_cancer_dir, 'data_v1') # data_v1 is only the baseline data

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='
PROJECT_ID = 'revivemed/Survival-RCC'

# desc_str = 'lungcancer v1'
desc_str = 'stanford bmi bs v2'
DATE_STR = 'May23'
head_hidden_layers = 0
name = 'basic-R Stanfor-BMI opt'
file_suffix = '_stanford_bs'

study_name = f'{desc_str}_{DATE_STR}'

if not os.path.exists(data_dir):
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "Stanford_BS_DATA")
    if not os.path.exists(data_dir):
        dropbox_url = 'https://www.dropbox.com/scl/fo/j7gwiuuc7m192bggtm5wp/ANlQ7FKqVXCJOEFsMmqYlwU?rlkey=c96yn3k682alsx4ou8pfdlhaf&dl=1'
        os.makedirs(data_dir)
        download_data_dir(dropbox_url, save_dir=data_dir)


# finetune_run_wrapper(data_dir=data_dir,
#                      desc_str = 'bmi',
#                      num_iterations = 10,
#                      eval_on_test = False,
#                      file_suffix = '_stanford_bs',
#                      num_epochs = 20,
#                      dropout_rate=0.2,
#                      head_hidden_layers=0,
#                      noise_factor=0.1,
#                      encoder_weight=0.1,
#                      use_cross_val=True,
#                      yes_plot_latent_space=True)

# TODO cross-validation
# TODO: umaps of latent space

############

# exit()

def objective(trial):
    # Define your hyperparameters for finetune_run_wrapper
    num_iter = 10

    use_l2_reg = trial.suggest_categorical('use_l2_reg', [True, False])
    if use_l2_reg:
        l2_reg_weight = trial.suggest_float('l2_reg_weight', 1e-5, 1e-3, log=True)
    else:
        l2_reg_weight = 0.0

    use_l1_reg = trial.suggest_categorical('use_l1_reg', [True, False])
    if use_l1_reg:
        l1_reg_weight = trial.suggest_float('l1_reg_weight', 1e-5, 1e-3, log=True)
    else:
        l1_reg_weight = 0.0

    use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])
    if use_weight_decay:
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    else:
        weight_decay = 0.0

    use_aux = trial.suggest_categorical('use_aux', [True, False])
    if use_aux:
        aux_weight = trial.suggest_float('auxillary_weight', 0.0, 0.5, step=0.1)
    else:
        aux_weight = 0

    sweep_kwargs = {
        'data_dir': data_dir,
        'use_rand_init': False,
        'holdout_frac': 0,
        'head_hidden_layers': head_hidden_layers,
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5,step=0.1),
        'num_epochs': trial.suggest_int('num_epochs', 5, 100, step=5),
        'early_stopping_patience': 0,
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'l2_reg_weight': l2_reg_weight,
        'l1_reg_weight': l1_reg_weight,
        'noise_factor': trial.suggest_float('noise_factor', 0.0, 0.25,step=0.05),
        'weight_decay': weight_decay,
        'head_weight': 1.0,
        'adversary_weight': 0, #trial.suggest_float('adversary_weight', 1e-2, 1e2, log=True),
        'auxillary_weight': aux_weight,
        'adversarial_start_epoch': 0, #trial.suggest_int('adversarial_start_epoch', 0, 20, step=5),
        'encoder_weight': trial.suggest_float('encoder_weight', 0.0, 1.0, step=0.1),
        # 'clip_grads_with_norm': args.clip_grads,
        'clip_grads_with_norm': trial.suggest_categorical('clip_grads_with_norm', [True, False]),
        'batch_size': 64,
        'yes_clean_batches': trial.suggest_categorical('yes_clean_batches', [True, False]),
        'train_name': 'train',
        'num_iterations': num_iter,
        'remove_nans': False,
        'name': name,
        'desc_str': desc_str,
        'eval_on_test': False,
        'file_suffix' : file_suffix,
        'use_cross_val': True,
        'study_name': study_name,
    }

    for k,v in sweep_kwargs.items():
        if isinstance(v, float):
            sweep_kwargs[k] = round_to_sig(v, 3)


    # Call finetune_run_wrapper with the hyperparameters
    try:
        run_id, all_metrics = finetune_run_wrapper(**sweep_kwargs)
        print('run_id:', run_id)
        trial.set_user_attr('run_id', run_id)

        # print(all_metrics.keys())
        # Assume that result is a dictionary with the two objectives
        # Replace 'objective1' and 'objective2' with your actual objectives
        if 'avg_training_run/metrics/val__head_BMI__on_BMI_MAE' not in all_metrics:
            optuna.TrialPruned()

        objective1_array = all_metrics['avg_training_run/metrics/val__head_BMI__on_BMI_MAE']

        num_success_iter = len(objective1_array)
        trial.set_user_attr('num_success_iter', num_success_iter)

        if num_success_iter < 3*num_iter/4:
            optuna.TrialPruned()
        
        objective1 = np.mean(objective1_array)
    except Exception as e:
        optuna.TrialPruned()
        objective1 = -1
    

    return objective1




WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

study = optuna.create_study(directions=['minimize'],
                        study_name=study_name, 
                        storage=WEBAPP_DB_LOC, 
                        load_if_exists=True)

study.optimize(objective, n_trials=5)


# get the best trial and run it with random initialization
best_trial = study.best_trial
best_trial_params = best_trial.params
best_trial_run_id = best_trial.user_attrs['run_id']
print(best_trial_params)
print(best_trial_run_id)


for with_id in [best_trial_run_id]:
    run = neptune.init_run(project=PROJECT_ID,
                            api_token=NEPTUNE_API_TOKEN,
                            with_id=with_id)
    
    run['sweep_kwargs/name2'] = with_id
    run.wait()
    original_sweep_kwargs = run['sweep_kwargs'].fetch()
    run.stop()

    original_sweep_kwargs = convert_neptune_kwargs(original_sweep_kwargs)
    original_sweep_kwargs['use_rand_init'] = True
    print(original_sweep_kwargs)
    finetune_run_wrapper(**original_sweep_kwargs)