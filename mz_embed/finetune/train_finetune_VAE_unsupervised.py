# developing a fine-tune VAE with transfer learning from a pretrained model 
'''
This script is used to fine-tune a VAE model with transfer learning from a pretrained model.
The pretrained model is a VAE model trained. The pretrained model is loaded
and the encoder part of the model is used to initialize the encoder part of the fine-tuned model.



'''
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import subprocess
import threading

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter





#importing the VAE model and the pretrain VAE model
from models.models_VAE import VAE
from pretrain.train_pretrain_VAE import PretrainVAE


def get_finetune_input_data(data_location):

    #defining the input datasets
    
    finetune_X_train=f'{data_location}/X_Finetune_Discovery_Train.csv'
    finetune_y_train=f'{data_location}/y_Finetune_Discovery_Train.csv'

    finetune_X_val=f'{data_location}/X_Finetune_Discovery_Val.csv'
    finetune_y_val=f'{data_location}/y_Finetune_Discovery_Val.csv'

    finetune_X_test=f'{data_location}/X_Finetune_Test.csv'
    finetune_y_test=f'{data_location}/y_Finetune_Test.csv'

    
    #loading the data
    X_data_train = pd.read_csv(finetune_X_train, index_col=0)
    y_data_train = pd.read_csv(finetune_y_train, index_col=0)

    X_data_val = pd.read_csv(finetune_X_val, index_col=0)
    y_data_val = pd.read_csv(finetune_y_val, index_col=0)

    X_data_test = pd.read_csv(finetune_X_test, index_col=0)
    y_data_test = pd.read_csv(finetune_y_test, index_col=0)


    #returning the data
    return(X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test)



def _reset_params(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()  # Call the layer's reset_parameters method if it exists



def fine_tune_vae(pretrain_VAE, model_path, X_data_train, 
                  X_data_val,  
                  X_data_test, 
                  batch_size=64, num_epochs=10, learning_rate=1e-4, 
                  dropout_rate=0.2, l1_reg=0.0, weight_decay=0.0, patience=0,
                  transfer_learning=True, **kwargs):
    
    
    # Initialize TensorBoard logging directory for each trial
    log_dir = f'{model_path}_TL_{transfer_learning}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)  # Initialize SummaryWriter for TensorBoard
        
    # Step 0: Convert DataFrames to NumPy arrays
    X_data_train_np = X_data_train.to_numpy()
    X_data_val_np = X_data_val.to_numpy()
    X_data_test_np = X_data_test.to_numpy()

    # Step 1: Create DataLoader objects for training, validation, and testing datasets
    train_dataset = TensorDataset(torch.Tensor(X_data_train_np))
    val_dataset = TensorDataset(torch.Tensor(X_data_val_np))
    test_dataset = TensorDataset(torch.Tensor(X_data_test_np))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 2: Initialize the fine-tuning model with the same architecture as the pre-trained VAE
    fine_tune_VAE = PretrainVAE (input_size=pretrain_VAE.input_size, 
                        latent_size=pretrain_VAE.latent_size, 
                        num_hidden_layers=pretrain_VAE.num_hidden_layers, 
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        dropout_rate=dropout_rate,  # Pass dropout_rate here
                        l1_reg=l1_reg,  # Pass l1_reg_weight here
                        weight_decay=weight_decay,  # Pass weight_decay here
                        patience=patience,
                        activation=pretrain_VAE.activation, 
                        use_batch_norm=pretrain_VAE.use_batch_norm)
    
    # setting the device right
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fine_tune_VAE.to(device)
    
    # Step 3: Transfer learning logic
    if transfer_learning:
        # Load pre-trained weights into the fine-tuning VAE
        fine_tune_VAE.load_state_dict(pretrain_VAE.state_dict())
        print("Transfer learning: Initialized weights from the pre-trained VAE.")
    else:
        # Random initialization (no need to load pre-trained weights)
        fine_tune_VAE.apply(_reset_params)
        print("Random initialization: Initialized weights randomly.")


    # Optional: If you want to freeze the encoder during fine-tuning, uncomment below:
    # for param in fine_tune_VAE.encoder.parameters():
    #     param.requires_grad = False


    # Step 4: Set up optimizer with weight decay for L2 regularization
    optimizer = optim.Adam(fine_tune_VAE.parameters(), lr=fine_tune_VAE.learning_rate, weight_decay=fine_tune_VAE.weight_decay)

    # L1 regularization function
    def l1_penalty(model):
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return model.l1_reg * l1_loss
    
    
    # kl_annealing_epochs = kwargs.get('kl_annealing_epochs', 0.5*num_epochs)  # Number of epochs for KL annealing
    # kl_start_weight = kwargs.get('kl_start_weight', 0.0)  # Initial weight for KL divergence
    # kl_max_weight = kwargs.get('kl_max_weight', 1)  # Maximum weight for KL divergence
    kl_annealing_epochs=0


    best_val_loss = np.inf
    best_model=None
    
    # Training function (no y_batch since we only have x_batch)
    for epoch in range(num_epochs):

        fine_tune_VAE.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        # # KL-annealing schedule: linearly increase KL weight over the annealing period
        # kl_weight = kl_start_weight + (kl_max_weight - kl_start_weight) * min(1, epoch / kl_annealing_epochs)
        
        #fine_tune_VAE.kl_weight = kl_weight
        fine_tune_VAE.kl_weight = 2.0
        
        for x_batch in train_loader:
            x_batch = x_batch[0].to(device)  # Extract the actual data from the tuple
            #print (f' x_batch.shape {x_batch.shape}')
            
            optimizer.zero_grad()
            x_recon, mu, log_var = fine_tune_VAE(x_batch)
            
            #compute the loss
            recon_loss, kl_loss, loss = fine_tune_VAE.loss(x_batch, x_recon, mu, log_var)
            loss += l1_penalty(fine_tune_VAE)  # Add L1 regularization

            loss.backward()  # Backward pass (compute gradients)

            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(fine_tune_VAE.parameters(), max_norm=1.0)
        
            optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_kl_loss = train_kl_loss / len(train_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_recon', avg_train_recon_loss, epoch)
        writer.add_scalar('Loss/train_kl', avg_train_kl_loss, epoch)
        
        # Validation loop
        fine_tune_VAE.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch[0].to(device)  # Extract the actual data from the tuple
                x_recon, mu, log_var = fine_tune_VAE(x_batch)
                recon_loss, kl_loss, loss = fine_tune_VAE.loss(x_batch, x_recon, mu, log_var)
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_kl_loss = val_kl_loss / len(val_loader)
        

        # scheduler.step(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Log validation loss to TensorBoard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Loss/val_recon', avg_val_recon_loss, epoch)
        writer.add_scalar('Loss/val_kl', avg_val_kl_loss, epoch)
        

        if avg_val_loss < best_val_loss and epoch > kl_annealing_epochs:
            best_val_loss = avg_val_loss
            best_model = fine_tune_VAE

   
    # Close TensorBoard writer
    writer.close()

    # Return the fine-tuned model and validation loss for Optuna to minimize
    return best_model, best_val_loss, log_dir
