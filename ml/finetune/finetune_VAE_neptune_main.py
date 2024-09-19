'''

fine tune  VAE model with and without transfer learning


'''

import argparse
import pandas as pd
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
import pickle

import optuna.visualization as vis
import plotly.io as pio


#Import my fucntions
from finetune.get_finetune_encoder import get_finetune_input_data

from pretrain_neptune.get_pretrain_encoder import get_pretrain_encoder_from_modelID, visualize_latent_space_multiple_tasks


from models.models_VAE import VAE
from finetune.train_finetune_VAE import fine_tune_vae



def objective(trial, pretrain_VAE, X_data_train, X_data_val, X_data_test, transfer_learning, result_name):
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
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    l1_reg_weight = trial.suggest_loguniform('l1_reg_weight', 1e-6, 1e-2)
    l2_reg_weight = trial.suggest_loguniform('l2_reg_weight', 1e-6, 1e-2)

    # Call fine_tune_vae with the suggested hyperparameters
    fine_tuned_model, val_loss = fine_tune_vae(pretrain_VAE, 
                                               X_data_train, 
                                               X_data_val, 
                                               X_data_test, 
                                               batch_size=32, 
                                               num_epochs=20, 
                                               learning_rate=learning_rate, 
                                               dropout_rate=dropout_rate, 
                                               l1_reg_weight=l1_reg_weight, 
                                               l2_reg_weight=l2_reg_weight, 
                                               transfer_learning=transfer_learning)
    
    # Save the model if it's the first trial or has the best performance so far
    if len(trial.study.trials) == 1 or val_loss < trial.study.best_value:
        # Save the best model to a file
        torch.save(fine_tuned_model, f'{result_name}_TL_{transfer_learning}_model.pth')
    
    # Return validation loss for Optuna to minimize
    return val_loss



def optimize_finetune_vae(pretrain_VAE, X_data_train, X_data_val, X_data_test, transfer_learning, result_name, n_trials=50):
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
    study.optimize(lambda trial: objective(trial, pretrain_VAE, X_data_train, X_data_val, X_data_test, transfer_learning, result_name), n_trials=n_trials)

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

    parser.add_argument('--pretrain_model_list_file', type=str,
                        default='/home/leilapirhaji/pretrained_models_to_finetune.txt',
                        help='File containing list of pre-trained model IDs to finetune.')

    parser.add_argument('--n_trial', type=int, default=50,
                        help='Number of trials for optimization.')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    input_data_location = args.input_data_location
    finetune_save_dir = args.finetune_save_dir
    pretrain_model_list_file = args.pretrain_model_list_file
    n_trial = args.n_trial


    # Read the text file into a DataFrame
    pretrain_model_list = pd.read_csv(pretrain_model_list_file, header=None)[0]
    pretrain_model_list = pretrain_model_list.dropna().tolist()


    
    #get the input data
    print ('get the input data')

    (X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_finetune_input_data(input_data_location)

    #get pretrain encoder
    print ('get pretrain encoder')
 

    for pretrain_modelID in pretrain_model_list:
        
        print ('pretrain_modelID:', pretrain_modelID)

        result_file_name = f'{finetune_save_dir}/{pretrain_modelID}/Finetune_VAE_pretain_{pretrain_modelID}_combined_optimization_history_TL_rand.html'

        # Check if the output file already exists
        if os.path.exists(result_file_name):
            print(f'Output file {result_file_name} already exists. Skipping this model.')
            continue  # Skip to the next pretrain_modelID


        (pretrain_VAE, Z_pretrain_all, Z_pretrain_train, Z_pretrain_val, Z_pretrain_test, y_data_all, y_pretrain_data_train, y_pretain_data_val, y__pretrain_data_test)=get_pretrain_encoder_from_modelID(pretrain_modelID, input_data_location, finetune_save_dir)


        # #UMAP visualization of the latent space of pretrain model
        # print ('UMAP visualization of the latent space of pretrain model')
        # output_path=f'{finetune_save_dir}/{pretrain_modelID}'
        # visualize_latent_space_multiple_tasks( pretrain_modelID, output_path, Z_pretrain_all, Z_pretrain_train, Z_pretrain_val, Z_pretrain_test, y_data_all, y_data_train, y_data_val, y_data_test)

        # print ('the UMUP visualization of the latent space of pretrain model is saved in:', output_path)


        #fine tune the encoder with transfer learning
        print ('fine tune the encoder with transfer learning')

        result_name=f'{finetune_save_dir}/{pretrain_modelID}/Finetune_VAE_pretain_{pretrain_modelID}'
        
        transfer_learning=True
        study_TL= optimize_finetune_vae(pretrain_VAE, X_data_train, X_data_val, X_data_test, transfer_learning, result_name, n_trials=n_trial)

        with open(f"{result_name}_TL_{transfer_learning}_optune_results.pkl", "wb") as f:
            pickle.dump(study_TL, f)

        #fine tune the encoder without transfer learning
        print ('fine tune the encoder without transfer learning')
        transfer_learning=False
        study_rand= optimize_finetune_vae(pretrain_VAE, X_data_train, X_data_val, X_data_test, transfer_learning, result_name, n_trials=n_trial)

        with open(f"{result_name}_TL_{transfer_learning}_optune_results.pkl", "wb") as f:
            pickle.dump(study_rand, f)
            
        #save the results
        print ('save the results')

        save_combined_optimization_history_html(study_TL, study_rand, f'{result_name}_combined_optimization_history_TL_rand')

print ('All DONE!')





if __name__ == '__main__':
    main()  # run the main function