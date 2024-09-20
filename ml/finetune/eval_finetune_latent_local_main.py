
''' 

this function get an VAE models and a task and return the prediction of the task using the average latent space of the VAE model


'''
import argparse
import pandas as pd
import importlib
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import optuna
import imaplib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

import torch
import torch.nn.functional as F


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import seaborn as sns


#importing fundtion to get encoder info and perfrom tasks 

from finetune.get_finetune_encoder import  get_finetune_input_data
from models.models_VAE import VAE
from finetune.latent_task_predict import log_reg_multi_class, cox_proportional_hazards, cox_proportional_hazards_l1_sksurv




def generate_latent_space(X_data, encoder, batch_size=128):
    if isinstance(X_data, pd.DataFrame):
        x_index = X_data.index
        X_data = torch.tensor(X_data.to_numpy(), dtype=torch.float32)
    Z = torch.tensor([])
    encoder.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    with torch.inference_mode():
        for i in range(0, len(X_data), batch_size):
            # print(i, len(X_data))
            X_batch = X_data[i:i+batch_size].to(device)
            Z_batch = encoder.transform(X_batch)
            Z_batch = Z_batch.cpu()
            Z = torch.cat((Z, Z_batch), dim=0)
        Z = Z.detach().numpy()
        Z = pd.DataFrame(Z, index=x_index)
    encoder.to('cpu')
    return Z




def generate_average_latent_space(X_data, model, batch_size, num_times=10):
    latent_space_sum = None
    for i in range(num_times):
        latent_space = generate_latent_space(X_data, model, batch_size)
        if latent_space_sum is None:
            latent_space_sum = latent_space
        else:
            latent_space_sum += latent_space  # Sum the latent spaces
    latent_space_avg = latent_space_sum / num_times  # Compute the average
    return latent_space_avg





def predict_task_from_latent_avg(vae_model, task, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10):
    
    # Generate latent spaces 10 times for each dataset
    # Generate averaged latent spaces
    Z_train = generate_average_latent_space(X_data_train, vae_model, batch_size, num_times)
    Z_val = generate_average_latent_space(X_data_val, vae_model, batch_size, num_times)
    Z_test = generate_average_latent_space(X_data_test, vae_model, batch_size, num_times)


    (best_val_accuracy, best_val_auc, test_accuracy, test_auc)= log_reg_multi_class(task, Z_train, y_data_train, Z_val, y_data_val, Z_test, y_data_test)

    return best_val_accuracy, best_val_auc, test_accuracy, test_auc





def predict_survival_from_latent_avg(vae_model, task, task_event, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10):
    
    # Generate latent spaces 10 times for each dataset
    # Generate averaged latent spaces
    Z_train = generate_average_latent_space(X_data_train, vae_model, batch_size, num_times)
    Z_val = generate_average_latent_space(X_data_val, vae_model, batch_size, num_times)
    Z_test = generate_average_latent_space(X_data_test, vae_model, batch_size, num_times)


    Y_train_OS = y_data_train[task]
    Y_train_event = y_data_train[task_event]

    Y_val_OS = y_data_val[task]
    Y_val_event = y_data_val[task_event]

    Y_test_OS = y_data_test[task]
    Y_test_event = y_data_test[task_event]

    #l2 cox
    #best_val_c_index, best_test_c_index, best_params= cox_proportional_hazards(Z_train, Y_train_OS, Y_train_event, Z_val, Y_val_OS, Y_val_event, Z_test, Y_test_OS, Y_test_event)

    #l1 cox
    best_val_c_index, best_test_c_index, best_params= cox_proportional_hazards_l1_sksurv(Z_train, Y_train_OS, Y_train_event, Z_val, Y_val_OS, Y_val_event, Z_test, Y_test_OS, Y_test_event)

    return best_val_c_index, best_test_c_index, best_params





# Custom function to convert DataFrame to Tensor
def dataframe_to_tensor(df, device):
    return torch.tensor(df.values, dtype=torch.float32).to(device)

# Custom function to compute reconstruction and KL loss, normalized by the number of samples
def compute_losses(model, X_data_train, X_data_val, X_data_test, device):
    model = model.to(device)  # Move the model to the appropriate device (GPU or CPU)
    model.eval()  # Set the model to evaluation mode
    losses = {}
    
    # Convert the input data to tensors
    X_data_train = dataframe_to_tensor(X_data_train, device)
    X_data_val = dataframe_to_tensor(X_data_val, device)
    X_data_test = dataframe_to_tensor(X_data_test, device)

    # Function to compute reconstruction and KL divergence loss, normalized by the number of samples
    def compute_recon_kl_loss(x, model):
        with torch.no_grad():  # Disable gradient computation
            recon_x, mu, log_var = model(x)  # Forward pass
            num_samples = x.size(0)  # Get the number of samples

            # Compute reconstruction loss (MSE here, you can change to BCE if needed)
            recon_loss = F.mse_loss(recon_x, x, reduction='mean') #/ num_samples  # Normalize by number of samples

            # Compute KL divergence loss and normalize by the number of samples
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1) # / num_samples
            
            # # Take the mean over the batch and normalize by the latent dimensions
            kl_loss = kl_loss.mean() / model.latent_size

            return recon_loss, kl_loss

    # Train dataset
    train_recon_loss, train_kl_loss = compute_recon_kl_loss(X_data_train, model)
    losses['train_recon_loss'] = train_recon_loss.item()
    losses['train_kl_loss'] = train_kl_loss.item()

    # Validation dataset
    val_recon_loss, val_kl_loss = compute_recon_kl_loss(X_data_val, model)
    losses['val_recon_loss'] = val_recon_loss.item()
    losses['val_kl_loss'] = val_kl_loss.item()

    # Test dataset
    test_recon_loss, test_kl_loss = compute_recon_kl_loss(X_data_test, model)
    losses['test_recon_loss'] = test_recon_loss.item()
    losses['test_kl_loss'] = test_kl_loss.item()

    return losses




def visualize_latent_space_multiple_tasks( vae_model, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file, batch_size=64, num_times=10, method='UMAP', n_components=2, dpi=300, palette='tab10', **kwargs):
    """
    Visualize the latent space for multiple tasks (categorical and numerical) using PCA or UMAP, and plot subplots for train, val, test, and all combined for each task.

    Parameters:
    - pretrain_encoder: Pre-trained encoder for getting latent space
    - Z_all, Z_train, Z_val, Z_test: Latent space data for all, train, val, and test.
    - y_data_all, y_data_train, y_data_val, y_data_test: Metadata for all, train, val, and test.
    - method: 'PCA' or 'UMAP' for dimensionality reduction.
    - n_components: Number of dimensions to reduce to (default is 2).
    - dpi: Resolution of the figure (default is 300 for high-quality images).
    - palette: Color palette to use for distinct colors (default is 'Set1').
    - kwargs: Additional parameters to pass to UMAP or PCA.
    """
    
    #Availbe taks for pre-trained meta-data
    #tasks to predict using encoder
    task_list_cat=['Treatment', 'IMDC BINARY', 'IMDC ORDINAL', 'MSKCC BINARY', 'MSKCC ORDINAL', 'ORR', 'Benefit', 'Prior_2' ]

    #survival tasks
    task_list_num=[ 'OS', 'NIVO OS', 'EVER OS', 'PFS']  # List of numerical tasks

    # Generate latent spaces 10 times for each dataset
    # Generate averaged latent spaces
    # making the emebedding for all the data
    X_data_all = pd.concat([X_data_train, X_data_val, X_data_test])
    y_data_all = pd.concat([y_data_train, y_data_val, y_data_test])

    Z_all = generate_average_latent_space(X_data_all, vae_model, batch_size, num_times)
    Z_train = generate_average_latent_space(X_data_train, vae_model, batch_size, num_times)
    Z_val = generate_average_latent_space(X_data_val, vae_model, batch_size, num_times)
    Z_test = generate_average_latent_space(X_data_test, vae_model, batch_size, num_times)


    # Perform dimensionality reduction (UMAP or PCA) once using all samples
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, **kwargs)
    else:
        raise ValueError("Method must be either 'PCA' or 'UMAP'")


    # Apply dimensionality reduction to all the samples
    Z_reduced_all = reducer.fit_transform(Z_all)
    Z_reduced_train = reducer.fit_transform(Z_train)
    Z_reduced_val = reducer.fit_transform(Z_val)
    Z_reduced_test = reducer.fit_transform(Z_test)


    # Function to filter and plot categorical data
    def filter_and_plot_cat(Z_reduced, y_data, data_label, task, ax):
        valid_rows = y_data[[task]].dropna().index
        # Convert valid_rows into integer indices that NumPy arrays understand
        valid_indices = [y_data.index.get_loc(idx) for idx in valid_rows]
        Z_filtered = Z_reduced[valid_indices]
        y_filtered = y_data.loc[valid_rows, task]

        # Count the number of samples for each category
        category_counts = y_filtered.value_counts().to_dict()

        # Create a custom legend label with sample counts
        custom_legend_labels = [f'{cat}: {count} samples' for cat, count in category_counts.items()]

        # Plot the reduced data with smaller dots and transparency for categorical data
        sns.scatterplot(x=Z_filtered[:, 0], y=Z_filtered[:, 1], hue=y_filtered, palette=palette, s=50, alpha=0.5, ax=ax)

        # Title with the total number of samples
        ax.set_title(f"{data_label} colored by {task} (n={len(valid_rows)})", fontsize=20)
        ax.set_xlabel(f'{method} Component 1', fontsize=18)
        ax.set_ylabel(f'{method} Component 2', fontsize=18)

        # Update the legend with sample counts
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, custom_legend_labels, title=f"{task} (n={len(valid_rows)})", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)


    # Function to filter and plot numerical data
    def filter_and_plot_num(Z_reduced, y_data, data_label, task, ax):
        valid_rows = y_data[[task]].dropna().index
        # Convert valid_rows into integer indices that NumPy arrays understand
        valid_indices = [y_data.index.get_loc(idx) for idx in valid_rows]
        Z_filtered = Z_reduced[valid_indices]
        y_filtered = y_data.loc[valid_rows, task]

        # Special handling for BMI: cap the values at 40 for better visualization
        if task == 'PFS':
            y_filtered_capped = y_filtered.copy()
            y_filtered_capped = np.where(y_filtered_capped > 20, 20, y_filtered_capped)  # Cap BMI values at 20
            scatter = ax.scatter(Z_filtered[:, 0], Z_filtered[:, 1], c=y_filtered_capped, cmap='coolwarm', s=50, alpha=0.5)
            plt.colorbar(scatter, ax=ax, label=f"{task} (capped at 20)")
        else:
            scatter = ax.scatter(Z_filtered[:, 0], Z_filtered[:, 1], c=y_filtered, cmap='coolwarm', s=50, alpha=0.5)
            plt.colorbar(scatter, ax=ax, label=task)

        # Title with the total number of samples
        ax.set_title(f"{data_label} colored by {task} (n={len(valid_rows)})", fontsize=20)
        ax.set_xlabel(f'{method} Component 1', fontsize=18)
        ax.set_ylabel(f'{method} Component 2', fontsize=18)


    # Adjust the figure to be square-shaped
    num_tasks = len(task_list_cat) + len(task_list_num)
    fig, axs = plt.subplots(num_tasks, 4, figsize=(42, 7 * num_tasks), dpi=dpi)  # Adjusted figsize for square subplots


    # Plot for categorical tasks
    for i, task in enumerate(task_list_cat):
        filter_and_plot_cat(Z_reduced_all, y_data_all, 'All Combined', task, axs[i, 0])
        filter_and_plot_cat(Z_reduced_train, y_data_train, 'Train', task, axs[i, 1])
        filter_and_plot_cat(Z_reduced_val, y_data_val, 'Validation', task, axs[i, 2])
        filter_and_plot_cat(Z_reduced_test, y_data_test, 'Test', task, axs[i, 3])

    # Plot for numerical tasks
    for i, task in enumerate(task_list_num, start=len(task_list_cat)):
        filter_and_plot_num(Z_reduced_all, y_data_all, 'All Combined', task, axs[i, 0])
        filter_and_plot_num(Z_reduced_train, y_data_train, 'Train', task, axs[i, 1])
        filter_and_plot_num(Z_reduced_val, y_data_val, 'Validation', task, axs[i, 2])
        filter_and_plot_num(Z_reduced_test, y_data_test, 'Test', task, axs[i, 3])

    plt.tight_layout()

    plt.savefig(result_png_file)
    print (f'Latent space visualization saved at {result_png_file}')
  








def main():
    
     #Set up the argument parser
    parser = argparse.ArgumentParser(description='Evaluating the latent space of the VAE models with and without transfer learning for Recon and KL loss, and predicting categorical and survival tasks using the average latent space.')

    # Define the arguments with defaults and help messages
    parser.add_argument('--input_data_location', type=str,
                        default='/home/leilapirhaji/PROCESSED_DATA_2',
                        help='Path to the input data location.')

    parser.add_argument('--finetune_save_dir', type=str,
                        default='/home/leilapirhaji/finetune_VAE_models',
                        help='Directory to finetuned VAE models, the results will be saved there too.')
    
    parser.add_argument('--pretrain_model_list_file', type=str,
                        default='/home/leilapirhaji/top_pretrained_models_local.txt',
                        help='This is a tsv file, that for each top pre-trained model it includes a colum of trial number and a column of trial name.')
    
    parser.add_argument('--result_name', type=str, default='',
                        help='the name of the result file.')

    
    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    input_data_location = args.input_data_location
    finetune_save_dir = args.finetune_save_dir
    pretrain_model_df_file = args.pretrain_model_list_file
    result_name = args.result_name

    # Read the text file into a DataFrame
    # Read the text file into a DataFrame
    pretrain_model_df = pd.read_csv(pretrain_model_df_file, sep='\t', header=0)
    pretrain_model_df = pretrain_model_df.replace(r'\n|\$|\'', '', regex=True)
 

    #tasks to predict using encoder
    task_list_cat=['Benefit BINARY', 'Nivo Benefit BINARY', 'MSKCC BINARY', 'IMDC BINARY', 'Benefit ORDINAL', 'MSKCC ORDINAL', 'IMDC ORDINAL', 'ORR', 'Benefit', 'IMDC', 'MSKCC', 'Prior_2' ]

    #survival tasks
    task_list_survival=[ 'OS', 'NIVO OS', 'EVER OS', 'PFS']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f'Device: {device}')

    #get fine-tuning input data 
    print ('Getting fine-tuning input data')
    (X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)=get_finetune_input_data(input_data_location)



    # Initialize a list to collect all results
    all_results = []

    for index, row in pretrain_model_df.iterrows():
        
        pretrain_name = row['trial_name']  # Access the value for 'trial_name' for the current row
        pretrain_id = row['trial_number']  # Access the value for 'trial_number' for the current row
         
        #loading the VAE modesl developed with and without transfer leanring
        print (f'Predicting tasks using the latnet space of the VAE models with pre-train model name {pretrain_name} and trial ID: {pretrain_id}') 

        #path to pre-train and fine-tune models
        models_path=f'{finetune_save_dir}/{pretrain_name}/trial_{pretrain_id}'

        #finetune models files
        finetune_VAE_TL_file= f'{models_path}/finetune_VAE_TL_True_model.pth'
        finetune_VAE_TL=torch.load(finetune_VAE_TL_file)

        finetune_VAE_noTL_file= f'{models_path}/finetune_VAE_TL_False_model.pth'
        finetune_VAE_noTL=torch.load(finetune_VAE_noTL_file)


        latent_size= finetune_VAE_TL.latent_size
        num_hidden_layers= finetune_VAE_TL.num_hidden_layers

        ## visulizing the latnet space of the VAE models

        # # with transfer learning
        # result_png_file_TL= f'{models_path}/finetune_VAE_{pretrain_model_ID}_TL_latent_space.png'
        # visualize_latent_space_multiple_tasks(finetune_VAE_TL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file_TL)

        # # without transfer learning
        # result_png_file_TL= f'{models_path}/finetune_VAE_{pretrain_model_ID}_TL_latent_space.png'
        # visualize_latent_space_multiple_tasks(finetune_VAE_noTL, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, result_png_file_TL)



        #getting the recon loss of the models
        # Example usage
        # Assuming model is your trained VAE model
        losses_TL = compute_losses(finetune_VAE_TL, X_data_train, X_data_val, X_data_test, device)
        losses_noTL = compute_losses(finetune_VAE_TL, X_data_train, X_data_val, X_data_test, device)


        for task in task_list_cat:
            print (f'Predicting task: {task}')
            # predicting tasks using the latnet space of the VAE models with transfer learning 
            best_val_accuracy_TL, best_val_auc_TL, test_accuracy_TL, test_auc_TL= predict_task_from_latent_avg (finetune_VAE_TL, task, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10)

            # predicting tasks using the latnet space of the VAE models without transfer learning
            best_val_accuracy_noTL, best_val_auc_noTL, test_accuracy_noTL, test_auc_noTL= predict_task_from_latent_avg (finetune_VAE_noTL, task, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10)


            # Append the results to the list
            result_dict = {
                'Pretrain Model ID': pretrain_id,
                'Pretrain Model name': pretrain_name,
                'Latent Size': latent_size,
                'Num Hidden Layers': num_hidden_layers,
                'Validation Recon Loss TL': losses_TL['val_recon_loss'],
                'Validation KL Loss TL': losses_TL['val_kl_loss'],
                'Test Recon Loss TL':losses_TL['test_recon_loss'],
                'Test KL Loss TL': losses_TL['test_kl_loss'],
                'Validation Recon Loss NO TL': losses_noTL['val_recon_loss'],
                'Validation KL Loss NO TL': losses_noTL['val_kl_loss'],
                'Test Recon Loss NO TL':losses_noTL['test_recon_loss'],
                'Test KL Loss NO TL':losses_noTL['test_kl_loss'],
                'Task': task,
                'Type': 'Classification',
                'Best Val Accuracy TL': best_val_accuracy_TL,
                'Best Val AUC TL': best_val_auc_TL,
                'Test Accuracy TL': test_accuracy_TL,
                'Test AUC TL': test_auc_TL,
                'Best Val Accuracy NO TL': best_val_accuracy_noTL,
                'Best Val AUC NO TL': best_val_auc_noTL,
                'Test Accuracy NO TL': test_accuracy_noTL,
                'Test AUC NO TL': test_auc_noTL
            }
            all_results.append(result_dict)


        # predicting survival tasks using the latnet space of the VAE models
        for task in task_list_survival:
            print (f'Predicting survival task: {task}')

            if task=='OS' or task=='NIVO OS' or task=='EVER OS':
                task_event= 'OS_Event'
            elif task=='PFS':
                task_event= 'PFS_Event'
            else:
                task_event= None

            # predicting survival tasks using the latnet space of the VAE models with transfer learning 
            best_val_c_index_TL, best_test_c_index_TL, best_params_TL= predict_survival_from_latent_avg (finetune_VAE_TL, task, task_event, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10)

            # predicting survival tasks using the latnet space of the VAE models without transfer learning
            best_val_c_index_noTL, best_test_c_index_noTL, best_params_noTL= predict_survival_from_latent_avg (finetune_VAE_noTL, task, task_event, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test,y_data_test, batch_size=64, num_times=10)


            # Append the results to the list
            result_dict = {
                'Pretrain Model ID': pretrain_id,
                'Pretrain Model name': pretrain_name,
                'Latent Size': latent_size,
                'Num Hidden Layers': num_hidden_layers,
                'Validation Recon Loss TL': np.log(losses_TL['val_recon_loss']),
                'Validation KL Loss TL': np.log(losses_TL['val_kl_loss']),
                'Test Recon Loss TL':np.log(losses_TL['test_recon_loss']),
                'Test KL Loss TL': np.log(losses_TL['test_kl_loss']),
                'Validation Recon Loss NO TL': np.log(losses_noTL['val_recon_loss']),
                'Validation KL Loss NO TL': np.log(losses_noTL['val_kl_loss']),
                'Test Recon Loss NO TL':np.log(losses_noTL['test_recon_loss']),
                'Test KL Loss NO TL':np.log(losses_noTL['test_kl_loss']),
                'Task': task,
                'Type': 'Survival',
                'Best Val C-Index TL': best_val_c_index_TL,
                'Best Test C-Index TL': best_test_c_index_TL,
                'Best Val C-Index NO TL': best_val_c_index_noTL,
                'Best Test C-Index NO TL': best_test_c_index_noTL,
            }
            all_results.append(result_dict)
        


    # Convert the list of results to a pandas DataFrame
    results_df = pd.DataFrame(all_results)

    # Save the results to a CSV file
    results_df.to_csv(f'{finetune_save_dir}/{pretrain_name}/fine_tune_TL_noTL_{pretrain_name}_latent_eval_results.csv', index=False)

    print (f'Predictions are saved in: {finetune_save_dir}/{pretrain_name}/fine_tune_TL_noTL_{pretrain_name}_latent_eval_results.csv')









if __name__=='__main__':
    main()