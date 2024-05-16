import pandas as pd
import numpy as np
import optuna
from main_finetune import finetune_run_wrapper
from misc import round_to_sig

desc_str = 'NIVO-OS ADV EVER-OS'

def objective(trial):
    # Define your hyperparameters for finetune_run_wrapper
    num_iter = 10
    head_hidden_layers = 1

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

    sweep_kwargs = {
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
        'adversary_weight': trial.suggest_float('adversary_weight', 1e-2, 1e2, log=True),
        'auxillary_weight': 0,
        'adversarial_start_epoch': trial.suggest_int('adversarial_start_epoch', 0, 20, step=5),
        'encoder_weight': trial.suggest_float('encoder_weight', 0.0, 1.0, step=0.1),
        # 'clip_grads_with_norm': args.clip_grads,
        'clip_grads_with_norm': trial.suggest_categorical('clip_grads_with_norm', [True, False]),
        'batch_size': 64,
        'yes_clean_batches': trial.suggest_categorical('yes_clean_batches', [True, False]),
        'train_name': 'train',
        'num_iterations': num_iter,
        'remove_nans': False,
        'name': 'layer-R ADVopt',
        'desc_str': desc_str,
        'eval_on_test': False
    }

    for k,v in sweep_kwargs.items():
        if isinstance(v, float):
            sweep_kwargs[k] = round_to_sig(v, 3)


    # Call finetune_run_wrapper with the hyperparameters
    run_id, all_metrics = finetune_run_wrapper(**sweep_kwargs)
    print('run_id:', run_id)
    trial.set_user_attr('run_id', run_id)

    # print(all_metrics.keys())
    # Assume that result is a dictionary with the two objectives
    # Replace 'objective1' and 'objective2' with your actual objectives
    if 'trainrun__val__head_NIVO OS__on_NIVO OS_Concordance Index' not in all_metrics:
        optuna.TrialPruned()

    objective1_array = all_metrics['trainrun__val__head_NIVO OS__on_NIVO OS_Concordance Index']
    objective2_array = all_metrics['trainrun__val__head_NIVO OS__on_EVER OS_Concordance Index']
    
    num_success_iter = len(objective1_array)
    trial.set_user_attr('num_success_iter', num_success_iter)

    if num_success_iter < 3*num_iter/4:
        optuna.TrialPruned()
    
    objective1 = np.mean(objective1_array)
    objective2 = np.mean(objective2_array)
    

    return objective1, objective2


DATE_STR = 'May16'
study_name = f'{desc_str}_{DATE_STR}'

WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

study = optuna.create_study(directions=['maximize', 'minimize'],
                        study_name=study_name, 
                        storage=WEBAPP_DB_LOC, 
                        load_if_exists=True)

study.optimize(objective, n_trials=100)

# Print the best parameters
# print(study.best_params)