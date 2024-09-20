
'''
evaluting pre-trained latnet space to predict new tasks

input: pre-trained latnet space, new tasks

output: evaluation results of the new tasks saved in csv file


'''


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

import torch
import torch.nn.functional as F

from pretrain.latent_task_predict import log_reg_multi_class, ridge_regression_predict



def generate_latent_space(X_data, vae_model, batch_size=128):
    if isinstance(X_data, pd.DataFrame):
        x_index = X_data.index
        X_data = torch.tensor(X_data.to_numpy(), dtype=torch.float32)
    Z = torch.tensor([])
    vae_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_model.to(device)
    with torch.inference_mode():
        for i in range(0, len(X_data), batch_size):
            # print(i, len(X_data))
            X_batch = X_data[i:i+batch_size].to(device)
            Z_batch = vae_model.transform(X_batch)
            Z_batch = Z_batch.cpu()
            Z = torch.cat((Z, Z_batch), dim=0)
        Z = Z.detach().numpy()
        Z = pd.DataFrame(Z, index=x_index)
    vae_model.to('cpu')
    return Z




def get_avg_latent_space(vae_model, X_data_train, X_data_val, X_data_test, model_folder, latent_passes=20):

    # Getting the latent space by averaging 
    Z_train = []
    Z_val = []
    Z_test = []

    for _ in range(latent_passes):  # Generate multiple latent representations
        
        # Convert DataFrames to Tensors
        latent_rep_train = torch.tensor(generate_latent_space(X_data_train, vae_model).values, dtype=torch.float32)
        Z_train.append(latent_rep_train)

        latent_rep_val = torch.tensor(generate_latent_space(X_data_val, vae_model).values, dtype=torch.float32)
        Z_val.append(latent_rep_val)

        latent_rep_test = torch.tensor(generate_latent_space(X_data_test, vae_model).values, dtype=torch.float32)
        Z_test.append(latent_rep_test)


    # Averaging the latent spaces
    Z_train = torch.mean(torch.stack(Z_train), dim=0)
    Z_val = torch.mean(torch.stack(Z_val), dim=0)
    Z_test = torch.mean(torch.stack(Z_test), dim=0)

    #now converting the latent space to dataframe
    Z_train = pd.DataFrame(Z_train, index=X_data_train.index)
    Z_val = pd.DataFrame(Z_val, index=X_data_val.index)
    Z_test = pd.DataFrame(Z_test, index=X_data_test.index)


    #save to csv
    Z_train.to_csv(f'{model_folder}/Z_train_avg_{latent_passes}.csv')
    Z_val.to_csv(f'{model_folder}/Z_val_avg_{latent_passes}.csv')
    Z_test.to_csv(f'{model_folder}/Z_test_avg_{latent_passes}.csv')

    return (Z_train, Z_val, Z_test)




def calculating_recon_loss(vae_model, X_data_train, X_data_val, X_data_test):
    """
    Calculate the reconstruction loss for the VAE model on training, validation, and test datasets.
    
    Parameters:
        vae_model: The trained VAE model.
        X_data_train: Pandas DataFrame containing training data.
        X_data_val: Pandas DataFrame containing validation data.
        X_data_test: Pandas DataFrame containing test data.
        
    Returns:
        recon_loss_train: Reconstruction loss on the training dataset.
        recon_loss_val: Reconstruction loss on the validation dataset.
        recon_loss_test: Reconstruction loss on the test dataset.
    """
    
    # Ensure model is on the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_model.to(device)
    
    # Convert the pandas DataFrames to PyTorch tensors and move to device
    X_train_tensor = torch.tensor(X_data_train.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_data_val.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_data_test.values, dtype=torch.float32).to(device)
    
    def compute_recon_loss(data_tensor, vae_model):
        vae_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            # Forward pass
            recon, mu, log_var = vae_model(data_tensor)
            
            # Compute reconstruction loss (assuming MSE loss for continuous data)
            recon_loss = F.mse_loss(recon, data_tensor, reduction='mean')
            
            # Compute KL divergence loss and normalize by the number of samples
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1) # / num_samples
            
            # # Take the mean over the batch and normalize by the latent dimensions
            kl_loss = kl_loss.mean() / vae_model.latent_size
            
            total_loss = recon_loss.item() + kl_loss.item()

            return recon_loss.item(), total_loss


    # Calculate reconstruction losses
    recon_loss_train, total_loss_train = compute_recon_loss(X_train_tensor, vae_model)
    recon_loss_val, total_loss_val = compute_recon_loss(X_val_tensor, vae_model)
    recon_loss_test, total_loss_test = compute_recon_loss(X_test_tensor, vae_model)
    
    return recon_loss_train, total_loss_train, recon_loss_val, total_loss_val, recon_loss_test, total_loss_test




def evalute_pretrain_latent_extra_task(vae_model, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, model_folder, latent_passes=20):

    #tasks to predict using encoder
    task_list_cat=[ 'Study ID', 'is Female', 'is Pediatric', 'Cohort Label v0','Smoking Status', 'Cancer Risk' ]

    task_list_num=[ 'BMI', 'Age' ]

    # getting the avegrae latent space for the model
    print ('Getting the avg latent space')
    (Z_train, Z_val, Z_test)=get_avg_latent_space(vae_model, X_data_train, X_data_val, X_data_test, model_folder, latent_passes=20)
    
    print ('Latent space shape')
    print (Z_train.shape, Z_val.shape, Z_test.shape)

    # evaluating the avg latent space for the reconstruction loss


    model_results = {}

    ## first, calcualting the Recon loss for train, val and test sets
    (recon_loss_train, total_loss_train, recon_loss_val, total_loss_val, recon_loss_test, total_loss_test)= calculating_recon_loss(vae_model, X_data_train, X_data_val, X_data_test)

    model_results['Total Loss Train'] = total_loss_train
    model_results['Recon Loss Train'] = recon_loss_train
    model_results['Total Loss Val'] = total_loss_val
    model_results['Recon Loss Val'] = recon_loss_val
    model_results['Total Loss Test'] = total_loss_test
    model_results['Recon Loss Test'] = recon_loss_test

    print ('Recon Loss Train:', recon_loss_train)
    print ('Recon Loss Val:', recon_loss_val)
    print ('Recon Loss Test:', recon_loss_test)
    

    # Second, predict the categorical tasks using the latnet space
    for task in task_list_cat:

        (val_accuracy, val_auc, test_accuracy, test_auc) = log_reg_multi_class(task, Z_train, y_data_train, Z_val, y_data_val, Z_test, y_data_test)

        # Store the results in the dictionary
        model_results[f'{task} Val Accuracy'] = val_accuracy
        model_results[f'{task} Test Accuracy'] = test_accuracy
        model_results[f'{task} Val AUC'] = val_auc
        model_results[f'{task} Test AUC'] = test_auc

        print(f'{task} Val Accuracy: {val_accuracy:.4f}')
        print(f'{task} Test Accuracy: {test_accuracy:.4f}')

    
    #now evaluting numercal task predictions
    for task in task_list_num:

        (val_mse, val_mae, val_r2, test_mse, test_mae, test_r2)= ridge_regression_predict(task, Z_train, y_data_train, Z_val, y_data_val, Z_test, y_data_test)

        # Store the results in the dictionary
        # model_results[f'{task} Val MSE'] = val_mse
        model_results[f'{task} Val MAE'] = val_mae
        # model_results[f'{task} Val R2'] = val_r2
        # model_results[f'{task} Test MSE'] = test_mse
        model_results[f'{task} Test MAE'] = test_mae
        # model_results[f'{task} Test R2'] = test_r2

        print(f'{task} Val MAE : {val_mae:.4f}')
        print(f'{task} Test MAE : {test_mae:.4f}')
        
    
    model_results_df = pd.DataFrame(model_results, index=[0])
    model_results_df.to_csv(f'{model_folder}/model_eval_results.csv', index=False)
    print ('Model results saved in csv file at', f'{model_folder}/model_eval_results.csv')



            