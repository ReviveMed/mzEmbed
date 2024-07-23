# Example Optuna run script to optimize finetuning using setup4.py
'''
runs an optuna optimization of finetuning model 3828 on two tasks: reconstruction of the finetuning data
and predicting the OS of the finetuning data. the finetuning process is run 5 times and the average of the
after 5 runs is used as the objective value
'''

########################################################################################


import optuna
from prep_run import make_kwargs_set
from setup4 import setup_wrapper
from prep_study2 import objective_func5, create_optuna_study
import os
import pandas as pd



project_id = 'revivemed/RCC'
homedir = os.path.expanduser("~")
input_data_dir = f'{homedir}/INPUT_DATA'


NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'


USE_WEBAPP_DB = True
SAVE_TRIALS = True
ADD_EXISTING_RUNS_TO_STUDY = True
limit_add = -1 # limit the number of runs added to the study
encoder_kind = 'VAE'

selections_df = pd.read_csv(f'{input_data_dir}/selection_df.csv',index_col=0)
output_dir = f'{homedir}/PROCESSED_DATA'
os.makedirs(output_dir, exist_ok=True)
subdir_col = 'Study ID'


########################################################################################
##### Define the Optuna Study #####


STUDY_DICT = {
    'study_name': 'MultiObj Minimize Example with setup4',
    'encoder_kind': encoder_kind,
    'sum objectives' : False,
    'objectives': {
        'Reconstruction Loss':{
            'weight': 3,
            'eval_loc': 'avg_finetune v0/metrics/Finetune_Discovery_Val_Reconstruction MSE',
            'name': 'Reconstruction Loss',
            'direction': 'minimize',
            'transform': 'log10',
            'default_value': 9999
        },
        'Both OS C-Index':{
            'weight': 1,
            'eval_loc': 'avg_finetune v0/metrics/Finetune_Discovery_Val__head_Both-OS__on_Both-OS_Concordance Index',
            'name': 'Both OS Concordance Index',
            'direction': 'maximize',
            'default_value': 0,
        },
    }
}


########################################################################################
###### What are hyper-parameters to search over? ######

def get_study_kwargs(head_kwargs_dict,adv_kwargs_dict,**kwargs):
    encoder_kind = kwargs.get('encoder_kind','VAE')
    run_kwargs = make_kwargs_set(encoder_kind=encoder_kind,
                head_kwargs_dict=head_kwargs_dict,
                adv_kwargs_dict=adv_kwargs_dict,

                latent_size=96, latent_size_min=96, latent_size_max=128, latent_size_step=4,
                hidden_size=-1, hidden_size_min=16, hidden_size_max=64, hidden_size_step=16,
                hidden_size_mult=1.5, hidden_size_mult_min=1.25, hidden_size_mult_max=2, hidden_size_mult_step=0.25,
                num_hidden_layers=2, num_hidden_layers_min=2, num_hidden_layers_max=3, num_hidden_layers_step=1,
                
                num_attention_heads=-1, num_attention_heads_min=1, num_attention_heads_max=5, num_attention_heads_step=1,
                num_decoder_layers=-1, num_decoder_layers_min=1, num_decoder_layers_max=5, num_decoder_layers_step=1,
                embed_dim=-1, embed_dim_min=4, embed_dim_max=64, embed_dim_step=4,
                decoder_embed_dim=-1, decoder_embed_dim_min=4, decoder_embed_dim_max=64, decoder_embed_dim_step=4,
                default_hidden_fraction=-1, default_hidden_fraction_min=0, default_hidden_fraction_max=0.5, default_hidden_fraction_step=0.1,

                dropout_rate=0, dropout_rate_min=0, dropout_rate_max=0.5, dropout_rate_step=0.1,
                encoder_weight=1.0, encoder_weight_min=0, encoder_weight_max=2, encoder_weight_step=0.5,
                head_weight=1.0, head_weight_min=0, head_weight_max=2, head_weight_step=0.5,
                adv_weight=1.0, adv_weight_min=0, adv_weight_max=2, adv_weight_step=0.5,

                task_head_weight=None, task_head_weight_min=0.25, task_head_weight_max=2, task_head_weight_step=0.25,
                task_adv_weight=-1, task_adv_weight_min=0, task_adv_weight_max=10, task_adv_weight_step=0.1,
                task_hidden_size=4, task_hidden_size_min=4, task_hidden_size_max=64, task_hidden_size_step=4,
                task_num_hidden_layers=1, task_num_hidden_layers_min=1, task_num_hidden_layers_max=3, task_num_hidden_layers_step=1,

                weight_decay=None, weight_decay_min=1e-5, weight_decay_max=1e-2, weight_decay_step=0.00001, weight_decay_log=True,
                l1_reg_weight=0, l1_reg_weight_min=0, l1_reg_weight_max=0.01, l1_reg_weight_step=0.0001, l1_reg_weight_log=False,
                l2_reg_weight=0, l2_reg_weight_min=0, l2_reg_weight_max=0.01, l2_reg_weight_step=0.0001, l2_reg_weight_log=False,
                
                batch_size=96, batch_size_min=16,batch_size_max=128,batch_size_step=16,
                noise_factor=None, noise_factor_min=0, noise_factor_max=0.25, noise_factor_step=0.05,
                num_epochs=None, num_epochs_min=10, num_epochs_max=100, num_epochs_step=10, num_epochs_log=False,
                learning_rate=None, learning_rate_min=0.00001, learning_rate_max=0.005, learning_rate_step=None,learning_rate_log=True,
                early_stopping_patience=0, early_stopping_patience_min=0, early_stopping_patience_max=50, early_stopping_patience_step=5,
                adversarial_start_epoch=0, adversarial_start_epoch_min=-1, adversarial_start_epoch_max=10, adversarial_start_epoch_step=2,
                )
    
    return run_kwargs


########################################################################################
##### Set up the objectives for the Optuna Study


def compute_objective(run_id):
    return objective_func5(run_id,
                            study_info_dict=STUDY_DICT,
                            project_id=project_id,
                            neptune_api_token=NEPTUNE_API_TOKEN)


def objective(trial):

    try:
        run_id, all_metrics = setup_wrapper(
            project_id = 'revivemed/RCC',
            api_token = NEPTUNE_API_TOKEN,
            setup_id = 'finetune v0',
            tags = 'setup4-test',
            fit_subset_col = 'Finetune Discovery Train',
            eval_subset_col_list = ['Finetune Discovery Val','Finetune Test'],
            selections_df = selections_df,
            output_dir = output_dir,
            pretrained_model_id = 'RCC-3828',
            pretrained_loc='pretrain',
            num_iterations = 5,
            optuna_trial=trial,
            head_name_list = ['Both-OS'],
            optuna_study_info_dict=STUDY_DICT,
            get_kwargs_func = get_study_kwargs,
            )
        
        trial.set_user_attr('run_id',run_id)
        return compute_objective(run_id)
        
    except ValueError as e:
        print(e)
        # return float('nan')
        raise optuna.TrialPruned()
    

########################################################################################
###### Create and optimize the Optuna study #####

optuna_study = create_optuna_study(study_info_dict=STUDY_DICT,
                    get_kwargs_wrapper=get_study_kwargs,
                    compute_objective=compute_objective)



optuna_study.optimize(objective, n_trials=5)