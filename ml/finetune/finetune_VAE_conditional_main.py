'''

fine tune  VAE model with and without transfer learning


'''

import argparse
import pandas as pd
import os
import shutil

import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
import pickle
import json

import optuna.visualization as vis
import plotly.io as pio


#Import my fucntions
from finetune.get_finetune_encoder import get_finetune_input_data


from models.models_VAE import VAE
from finetune.train_finetune_VAE_conditional import fine_tune_vae
from pretrain.get_pretrain_encoder import get_pretrain_encoder_from_local



def objective(trial, pretrain_VAE, X_data_train, X_data_val, y_data_train_cond, y_data_val_cond, transfer_learning, result_name, combined_params):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial (optuna.trial.Trial): The trial object for hyperparameter suggestions.
        pretrain_VAE (nn.Module): Pretrained VAE model.
        X_data_train, X_data_val, X_data_test (DataFrame): Training, validation, and test data.
        
    Returns:
        float: Validation loss after fine-tuning the VAE model.
    """
    # Suggest hyperparameters using Optuna
    # Dropout rate: float range or fixed
    if isinstance(combined_params['dropout_rate'], tuple):
        min_val, max_val, step = combined_params['dropout_rate']
        dropout_rate = trial.suggest_float('dropout_rate', min_val, max_val, step=step)
    else:
        dropout_rate = combined_params['dropout_rate']
        
    # Learning rate: loguniform or fixed
    if isinstance(combined_params['learning_rate'], tuple):
        learning_rate = trial.suggest_loguniform('learning_rate', combined_params['learning_rate'][0], combined_params['learning_rate'][1])
    else:
        learning_rate = combined_params['learning_rate']
        
    # L2 regularization: loguniform or fixed
    if isinstance(combined_params['l1_reg'], tuple):
        l1_reg = trial.suggest_loguniform('l1_reg', combined_params['l1_reg'][0], combined_params['l1_reg'][1])
    else:
        l1_reg = combined_params['l1_reg']

    # Weight decay: loguniform or fixed
    if isinstance(combined_params['weight_decay'], tuple):
        weight_decay = trial.suggest_loguniform('weight_decay', combined_params['weight_decay'][0],combined_params['weight_decay'][1] )
    else:
        weight_decay = combined_params['weight_decay']

    # Batch size: categorical (fixed list of values) or fixed
    if isinstance(combined_params['batch_size'], tuple):
        min_val, max_val, step = combined_params['batch_size']
        batch_size = trial.suggest_int('batch_size', min_val, max_val, step=step)
    else:
        batch_size = combined_params['batch_size']

    # Patience: integer range or fixed
    if isinstance(combined_params['patience'], tuple):
        min_val, max_val, step = combined_params['patience']
        patience = trial.suggest_int('patience', min_val, max_val, step=step)
    else:
        patience = combined_params['patience']

    # Number of epochs: integer range or fixed
    if isinstance(combined_params['num_epochs'], tuple):
        min_val, max_val, step = combined_params['num_epochs']
        num_epochs = trial.suggest_int('num_epochs', min_val, max_val, step=step)
    else:
        num_epochs = combined_params['num_epochs']

    
    
    if len(trial.study.trials) == 1:
        best_val_loss = float('inf')
    else:
        best_val_loss = trial.study.best_value

    
    # Call fine_tune_vae with the suggested hyperparameters
    fine_tuned_model, val_loss, log_dir = fine_tune_vae(pretrain_VAE, result_name, X_data_train, X_data_val, y_data_train_cond, y_data_val_cond, batch_size, num_epochs, learning_rate, dropout_rate, l1_reg, weight_decay, patience, transfer_learning)
    
    # Save the model if it's the first trial or has the best performance so far
    # Track the best model and logs
    
    best_model_path = f'{result_name}_TL_{transfer_learning}_best_model'
    best_log_dir = f'{result_name}_TL_{transfer_learning}_best_logs'
    
    
    try:
        if fine_tuned_model is not None:
            
            if len(trial.study.trials) == 1 or val_loss  < best_val_loss:
                
                # Save the best model to a file
                torch.save(fine_tuned_model, f'{best_model_path}_state.pt')
                
                # Save the model hyperparameters
                hyperparameters = fine_tuned_model.get_hyperparameters()

                # Filter out non-serializable values
                serializable_hyperparameters = {k: v for k, v in hyperparameters.items() if isinstance(v, (int, float, str, bool, list, dict))}
                
                with open(f'{best_model_path}_model_hyperparameters.json', 'w') as f:
                    json.dump(serializable_hyperparameters, f, indent=4)

                
                # Copy logs to a "best logs" directory
                if os.path.exists(best_log_dir):
                    shutil.rmtree(best_log_dir)  # Remove previous best logs
                shutil.copytree(log_dir, best_log_dir)  # Copy current trial's logs as the best logs

    except ValueError:
        # Handle the case where no trials are completed yet
        print("No trials are completed yet. Proceeding with current trial.")
                
    # Return validation loss for Optuna to minimize
    return val_loss



def optimize_finetune_vae(pretrain_VAE, X_data_train, X_data_val, y_data_train_cond, y_data_val_cond, transfer_learning, result_name, combined_params, n_trials=50):
    """
    Optimize the fine-tuning of a VAE using Optuna.
    
    Args:
        pretrain_VAE (nn.Module): Pretrained VAE model.
        X_data_train, X_data_val, X_data_test (DataFrame): Training, validation, and test data.
        n_trials (int): Number of trials to run for hyperparameter optimization.
        
    Returns:
        optuna.study.Study: Optuna study object with optimization results.
    """
    # Create an Optuna study to minimize the validation loss
    study = optuna.create_study(direction='minimize')

    # Define the objective function
    study.optimize(lambda trial: objective(trial, pretrain_VAE, X_data_train, X_data_val, y_data_train_cond, y_data_val_cond, transfer_learning, result_name, combined_params), n_trials=n_trials )

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)

    # Return the study object for further analysis
    return study




def save_combined_optimization_history_html(study_TL, study_rand, filename):
    """
    Save the optimization history plots, best model details, and separate trial results tables of two Optuna studies into a single HTML file.

    Args:
        study_TL (optuna.study.Study): The first Optuna study (transfer learning).
        study_rand (optuna.study.Study): The second Optuna study (random initialization).
        filename (str): The name of the HTML file to save (without the extension).
        
    Returns:
        None
    """
    # Generate optimization history plots for both studies
    fig_TL = vis.plot_optimization_history(study_TL)
    fig_rand = vis.plot_optimization_history(study_rand)
    
    # Get the best trial details for both studies
    best_trial_TL = study_TL.best_trial
    best_trial_rand = study_rand.best_trial
    
    # Get all parameter keys to create a consistent table structure
    all_param_keys_TL = set()
    all_param_keys_rand = set()

    for trial in study_TL.trials:
        all_param_keys_TL.update(trial.params.keys())
    for trial in study_rand.trials:
        all_param_keys_rand.update(trial.params.keys())
        
    all_param_keys_TL = sorted(list(all_param_keys_TL))  # Sort the keys for consistency in TL
    all_param_keys_rand = sorted(list(all_param_keys_rand))  # Sort the keys for consistency in rand
    
    # Open the HTML file and write both plots, best trial details, and trial results into it
    with open(f"{filename}.html", "w") as f:
        f.write("<html><head><title>Optuna Study Results</title>\n")
        f.write("""
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.js"></script>
        <script>
            $(document).ready(function() {
                $('#optuna_table_TL').DataTable({
                    "scrollX": true,  // Enable horizontal scrolling for TL
                    "paging": true,   // Enable pagination for TL
                    "searching": true,  // Enable search filtering for TL
                    "ordering": true,  // Enable column sorting for TL
                    "pageLength": 10   // Set number of rows per page for TL
                });
                $('#optuna_table_rand').DataTable({
                    "scrollX": true,  // Enable horizontal scrolling for rand
                    "paging": true,   // Enable pagination for rand
                    "searching": true,  // Enable search filtering for rand
                    "ordering": true,  // Enable column sorting for rand
                    "pageLength": 10   // Set number of rows per page for rand
                });
            });
        </script>
        </head><body>\n""")
        
        # Transfer Learning (TL) study details
        f.write("<h1>Transfer Learning (TL) Study</h1>\n")
        f.write("<h2>Best Trial</h2>\n")
        f.write(f"<p>Trial number: {best_trial_TL.number}</p>\n")
        f.write(f"<p>Objective value: {best_trial_TL.value}</p>\n")
        f.write("<p>Best Parameters:<br>")
        for key, value in best_trial_TL.params.items():
            f.write(f"{key}: {value}<br>")
        f.write("</p>\n")
        
        # Transfer Learning (TL) Optimization History Plot
        f.write("<h2>Optimization History for Transfer Learning (TL)</h2>\n")
        f.write(pio.to_html(fig_TL, full_html=False))
        
        # Transfer Learning (TL) Trial Results Table
        f.write("<h2>All Trials for Transfer Learning (TL)</h2>\n")
        f.write('<table id="optuna_table_TL" class="display" style="width:100%">\n')
        f.write("<thead><tr><th>Trial Number</th><th>Objective Value</th>")
        for key in all_param_keys_TL:
            f.write(f"<th>{key}</th>")
        f.write("</tr></thead>\n")

        f.write("<tbody>\n")
        for trial in study_TL.trials:
            f.write(f"<tr><td>{trial.number}</td><td>{trial.value}</td>")
            for key in all_param_keys_TL:
                f.write(f"<td>{trial.params.get(key, 'N/A')}</td>")
            f.write("</tr>\n")
        f.write("</tbody>\n")
        f.write("</table>\n")
        
        # Random Initialization study details
        f.write("<h1>Random Initialization Study</h1>\n")
        f.write("<h2>Best Trial</h2>\n")
        f.write(f"<p>Trial number: {best_trial_rand.number}</p>\n")
        f.write(f"<p>Objective value: {best_trial_rand.value}</p>\n")
        f.write("<p>Best Parameters:<br>")
        for key, value in best_trial_rand.params.items():
            f.write(f"{key}: {value}<br>")
        f.write("</p>\n")
        
        # Random Initialization Optimization History Plot
        f.write("<h2>Optimization History for Random Initialization</h2>\n")
        f.write(pio.to_html(fig_rand, full_html=False))
        
        # Random Initialization Trial Results Table
        f.write("<h2>All Trials for Random Initialization</h2>\n")
        f.write('<table id="optuna_table_rand" class="display" style="width:100%">\n')
        f.write("<thead><tr><th>Trial Number</th><th>Objective Value</th>")
        for key in all_param_keys_rand:
            f.write(f"<th>{key}</th>")
        f.write("</tr></thead>\n")

        f.write("<tbody>\n")
        for trial in study_rand.trials:
            f.write(f"<tr><td>{trial.number}</td><td>{trial.value}</td>")
            for key in all_param_keys_rand:
                f.write(f"<td>{trial.params.get(key, 'N/A')}</td>")
            f.write("</tr>\n")
        f.write("</tbody>\n")
        f.write("</table>\n")
        
        f.write("</body></html>\n")
    
    print(f"Combined HTML report with separate tables for TL and Random saved as {filename}.html")




def parse_range_or_single(values, is_int=False):
    """
    Custom argparse type to parse either a single value or a range from a list.
    If `is_int` is True, parses integers; otherwise, parses floats.
    
    Example inputs: 
        ["256"] -> Outputs: 256 (int)
        ["256", "512", "32"] -> Outputs: (256, 512, 32) (int)
    """
    # If it's a list but contains only one element, treat it as a single value
    if len(values) == 1:
        if is_int:
            return int(values[0])  # Single integer value
        else:
            return float(values[0])  # Single float value
    
    # Otherwise, treat it as a range
    if is_int:
        return tuple(map(int, values))  # Convert to tuple of integers
    else:
        return tuple(map(float, values))  # Convert to tuple of floats




def main ():

    
    #Set up the argument parser
    parser = argparse.ArgumentParser(description='It get a list of pre-trained models, and use them to develop finetune VAE models with trasnfer learning of the pretrained model. It also develop a fine-tune model with NO tansfer learning (i,e, random init.). It uses optuna to find the best hyper-paramteres to minize the Recon loss of finetune models. it also visulaize the latent space of the pretrained model used for transfer learning and save all the results.')

    # Define the arguments with defaults and help messages
    parser.add_argument('--input_data_location', type=str,
                        default='/home/leilapirhaji/PROCESSED_DATA_2',
                        help='Path to the input data location.')

    parser.add_argument('--finetune_save_dir', type=str,
                        default='/home/leilapirhaji/finetune_VAE_models',
                        help='Directory to save finetuned models.')
    
    parser.add_argument('--pretrain_save_dir', type=str,
                        default='/home/leilapirhaji/pretrained_models',
                        help='Directory of saved pre-trained models.')

    parser.add_argument('--pretrain_model_list_file', type=str,
                        default='/home/leilapirhaji/top_pretrained_models_local.txt',
                        help='This is a tsv file, that for each top pre-trained model it includes a colum of trial number and a column of trial name.')
    
    parser.add_argument('--condition_list', type=str,
                        default='OS, OS_event',
                        help='A list of conditions to be used for the conditional VAE model. It should be a string of condition names separated by commas. Example: "OS,MSI,MSS"')
        
    # Add arguments for the hyperparameters
    parser.add_argument('--dropout_rate', nargs='*', default=[0.1], 
                        help='Dropout rate: Either a single value or "min max step" for range (float)')
    parser.add_argument('--learning_rate', nargs='*', default=["1e-5", "1e-2"], 
                        help='Learning rate range: Either a single value or "min max" for range (float)')
    parser.add_argument('--l1_reg', nargs='*', default=["1e-6", "1e-2"],
                        help='Weight decay range: Either a single value or "min max" for range (float)')
    parser.add_argument('--weight_decay', nargs='*', default=["1e-6", "1e-2"],
                        help='Weight decay range: Either a single value or "min max" for range (float)')
    parser.add_argument('--batch_size', nargs='*', default=[32],
                        help='Batch size (either fixed or a range, integer)')
    parser.add_argument('--patience', nargs='*', default=[0], 
                        help='Patience: Either a single value or "min max step" for range (integer), 0 means no early stopping')
    parser.add_argument('--num_epochs', nargs='*', default=[30], 
                        help='Number of epochs (either fixed or a range, integer)')
    parser.add_argument('--n_trials', type=int, default=50, 
                        help='Number of trials for hyperparameter optimization')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Use parsed arguments and convert them using parse_range_or_single
    combined_params = {
        'dropout_rate': parse_range_or_single(args.dropout_rate, is_int=False),
        'learning_rate': parse_range_or_single(args.learning_rate, is_int=False),
        'l1_reg': parse_range_or_single(args.l1_reg, is_int=False),
        'weight_decay': parse_range_or_single(args.weight_decay, is_int=False),
        'batch_size': parse_range_or_single(args.batch_size, is_int=True),
        'patience': parse_range_or_single(args.patience, is_int=True),
        'num_epochs': parse_range_or_single(args.num_epochs, is_int=True)
    }

    print ('combined_params:', combined_params)
    
    # Access the arguments
    input_data_location = args.input_data_location
    finetune_save_dir = args.finetune_save_dir
    pretrain_save_dir = args.pretrain_save_dir
    pretrain_model_df_file = args.pretrain_model_list_file
    n_trials = args.n_trials
    
    condition_list_str=args.condition_list
    condition_list=condition_list_str.split(',')

    print ('task is : {task}')
    
    #get the input data
    print ('get the input data')

    (X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_finetune_input_data(input_data_location)
    
    #creating conditional data from y_data
    y_data_train_cond = y_data_train[condition_list].fillna(-1)
    y_data_val_cond = y_data_val[condition_list].fillna(-1)

    #get pretrain encoder
    print ('get pretrain encoder')
 
    # Read the text file into a DataFrame
    pretrain_model_df = pd.read_csv(pretrain_model_df_file, sep='\t', header=0)
    pretrain_model_df = pretrain_model_df.replace(r'\n|\$|\'', '', regex=True)


    for index, row in pretrain_model_df.iterrows():
        pretrain_name = row['trial_name']  # Access the value for 'trial_name' for the current row
        pretrain_id = row['trial_number']  # Access the value for 'trial_number' for the current row
        
        print(pretrain_name, pretrain_id)
        
        VAE_results_dir = f'{finetune_save_dir}/{pretrain_name}/trial_{pretrain_id}'
        
        os.makedirs(VAE_results_dir, exist_ok=True)

        # Use a modified version of task only for file naming (spaces replaced with underscores)
        condition_for_filename = condition_list_str.replace(' ', '_')

        result_file_name = f'{finetune_save_dir}/{pretrain_name}/trial_{pretrain_id}/{condition_for_filename}_finetune_VAE_combined_optimization_history_TL_rand.html'

        # Check if the output file already exists
        if os.path.exists(result_file_name):
            print(f'Output file {result_file_name} already exists. Skipping this model.')
            continue  # Skip to the next pretrain_modelID


        (pretrain_VAE, Z_pretrain_train, Z_pretrain_val, Z_pretrain_test)=get_pretrain_encoder_from_local(pretrain_name, pretrain_id, pretrain_save_dir)


        # #UMAP visualization of the latent space of pretrain model
        # print ('UMAP visualization of the latent space of pretrain model')
        # output_path=f'{finetune_save_dir}/{pretrain_modelID}'
        # visualize_latent_space_multiple_tasks( pretrain_modelID, output_path, Z_pretrain_all, Z_pretrain_train, Z_pretrain_val, Z_pretrain_test, y_data_all, y_data_train, y_data_val, y_data_test)

        # print ('the UMUP visualization of the latent space of pretrain model is saved in:', output_path)


        #fine tune the encoder with transfer learning
        print ('fine tune the encoder with transfer learning')

        result_name=f'{finetune_save_dir}/{pretrain_name}/trial_{pretrain_id}/{condition_for_filename}_finetune_VAE'
        
        transfer_learning=True
        study_TL= optimize_finetune_vae(pretrain_VAE, 
                                        X_data_train, 
                                        X_data_val, 
                                        y_data_train_cond,
                                        y_data_val_cond,
                                        transfer_learning, result_name, combined_params, n_trials=n_trials)

        with open(f"{result_name}_TL_{transfer_learning}_optune_results.pkl", "wb") as f:
            pickle.dump(study_TL, f)

        #fine tune the encoder without transfer learning
        print ('fine tune the encoder without transfer learning')
        transfer_learning=False
        study_rand= optimize_finetune_vae(pretrain_VAE, 
                                            X_data_train, 
                                            X_data_val, 
                                            y_data_train_cond,
                                            y_data_val_cond,
                                            transfer_learning, result_name, combined_params, n_trials=n_trials)

        with open(f"{result_name}_TL_{transfer_learning}_optune_results.pkl", "wb") as f:
            pickle.dump(study_rand, f)
            
        #save the results
        print ('save the results')

        save_combined_optimization_history_html(study_TL, study_rand, f'{result_name}_combined_optimization_history_TL_rand')

print ('All DONE!')





if __name__ == '__main__':
    main()  # run the main function