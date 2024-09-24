'''

plot latent space of pre-trained models

'''


import os
ml_code_path='/home/leilapirhaji/mz_embed_engine/ml'
os.chdir(ml_code_path)

import pandas as pd
import json



pretrain_save_dir='/home/leilapirhaji/pretrained_models'
trial_name= 'pretrain_VAE_L_425_485_e_400_p_25'


trail_list=os.listdir(f'{pretrain_save_dir}/{trial_name}')


## combining the eval results of all trials
# Initialize an empty DataFrame to store all evaluation results
all_eval_results = pd.DataFrame()

# Iterate over all trials in the Optuna study
for trial in trail_list:
    # Path to the folder where the model evaluation results are saved
    model_folder = f'{pretrain_save_dir}/{trial_name}/{trial}'
    
    # Path to the model_eval_results.csv file for the current trial
    eval_results_path = os.path.join(model_folder, 'model_eval_results.csv')
    
    # Path to the JSON file containing hyperparameters
    json_params_path = os.path.join(model_folder, 'model_hyperparameters.json')
    
    # Check if the evaluation results file exists before trying to load it
    if os.path.exists(eval_results_path):
        eval_results = pd.read_csv(eval_results_path)
        
        # Add trial-specific information
        eval_results['trial_number'] = trial
        eval_results['trial_name'] = trial_name
        
        # Reorder the columns to have 'trial_number' and 'trial_name' first
        columns_order = ['trial_number', 'trial_name'] + [col for col in eval_results.columns if col not in ['trial_number', 'trial_name']]
        eval_results = eval_results[columns_order]
        
        # Check if the JSON file with hyperparameters exists before loading it
        if os.path.exists(json_params_path):
            with open(json_params_path, 'r') as json_file:
                hyperparams = json.load(json_file)
            
            # Add the hyperparameters as new columns in eval_results
            for key, value in hyperparams.items():
                eval_results[key] = value
        
        # Concatenate the current trial's results to the overall results DataFrame
        all_eval_results = pd.concat([all_eval_results, eval_results], ignore_index=True)
    else:
        print(f"Warning: {eval_results_path} does not exist for trial {trial}")

# Save the combined evaluation results to a CSV file
combined_results_path = f'{pretrain_save_dir}/{trial_name}/{trial_name}_all_model_par_eval_results.csv'
all_eval_results.to_csv(combined_results_path, index=False)

print(f"Combined evaluation results with hyperparameters saved to {combined_results_path}")


        
#save the results
print ('All done and results are saved') 


