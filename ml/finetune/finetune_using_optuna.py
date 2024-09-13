from pretrain.setup4 import setup_wrapper
import os
import pandas as pd
import optuna
from utils.utils_neptune import get_latest_dataset
from pretrain.prep_study2 import objective_func4

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZDlmZGM4ZC05OGM2LTQ2YzctYmRhNi0zMjIwODMzMWM1ODYifQ=='
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

project_id = 'revivemed/RCC'
homedir = os.path.expanduser("~")
input_data_dir = f'{homedir}/INPUT_DATA'
os.makedirs(input_data_dir, exist_ok=True)
input_data_dir = get_latest_dataset(data_dir=input_data_dir, api_token=NEPTUNE_API_TOKEN, project=project_id)

# Load the selection dataframe
selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv', index_col=0)

model_id_names = [37133]
for model_id_name in model_id_names:

    output_dir = f'{homedir}/PROCESSED_DATA'
    os.makedirs(output_dir, exist_ok=True)
    subdir_col = 'Study ID'
    # model_id_name = 11517
    optua_study_name = f'finetune-optuna-RCC-{model_id_name}-recon-only-Sep5'
    encoder_kind = 'VAE'

    STUDY_INFO_DICT = {
        'study_name': optua_study_name,
        'encoder_kind': encoder_kind,
    }


    # Define the Optuna objective function
    def objective(trial):
        setup_id = 'finetune-optuna'

        # Hyperparameters to optimize
        # num_epochs use fixed
        num_epochs = 20
        # num_epochs = trial.suggest_int('num_epochs', 5, 50)
        # learning_rate give range
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        # dropout_rate a range as well
        dropout_rate = trial.suggest_uniform('dropout_rate', 0, 0.5)
        # weight_decay a range as well
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        # l1_reg_weight a range as well
        l1_reg_weight = trial.suggest_uniform('l1_reg_weight', 0, 0.01)
        # l2_reg_weight a range as well
        l2_reg_weight = trial.suggest_uniform('l2_reg_weight', 0, 0.01)

        # Call the setup_wrapper with Optuna parameters
        run_id, all_metrics = setup_wrapper(
            project_id=project_id,
            api_token=NEPTUNE_API_TOKEN,
            setup_id=setup_id,
            tags=optua_study_name,
            fit_subset_col='Finetune Discovery Train',
            eval_subset_col_list=['Finetune Discovery Val'],
            selections_df=selections_df,
            output_dir=output_dir,
            head_name_list=['Both-OS'],
            # head_name_list=['IMDC'],
            overwrite_params_fit_kwargs={
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'dropout_rate': dropout_rate,
                'weight_decay': weight_decay,
                'l1_reg_weight': l1_reg_weight,
                'l2_reg_weight': l2_reg_weight,
                'use_rand_init': False,
            },
            overwrite_params_task_kwargs={
                'hidden_size': 0,
                'num_hidden_layers': 0,
            },
            overwrite_existing_params=True,
            pretrained_model_id=f'RCC-{model_id_name}',
            pretrained_loc='pretrain',
            optuna_study_info_dict=STUDY_INFO_DICT,
            optuna_trial=trial,
            num_iterations=10,
            use_rand_init=False,
        )

        trial.set_user_attr('run_id', run_id)
        trial.set_user_attr('setup_id', setup_id)

        print(f"all_metrics: {all_metrics}")
        # get the loss from the all_metrics
        # Compute val_loss using the method above
        # task for both os
        val_loss_list = all_metrics.get(
            f'{setup_id}__Finetune_Discovery_Val_Reconstruction MSE', [0])
        if val_loss_list:
            # use the average of the list
            val_loss = sum(val_loss_list) / len(val_loss_list)
        else:
            val_loss = 10
        if val_loss == 0:
            val_loss = 10
        print(f"val_loss: {val_loss}")
        return val_loss

    # Run the optimization
    num_trials = 50
    study = optuna.create_study(direction='minimize',  # corrected this line
                                study_name=optua_study_name,
                                storage=WEBAPP_DB_LOC,
                                load_if_exists=True)

    study.optimize(objective, n_trials=num_trials)

    # Output the best hyperparameters
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (validation loss): {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
