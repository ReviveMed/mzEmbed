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
from models.SupervisedVAE import SupervisedVAE
from pretrain.train_pretrain_VAE import PretrainVAE


def _reset_params(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()  # Call the layer's reset_parameters method if it exists



def fine_tune_vae(pretrain_VAE, model_path, X_data_train, X_data_val, y_data_train, y_data_val, task, task_type, num_classes, task_event, batch_size, num_epochs, learning_rate, dropout_rate, l1_reg, weight_decay, patience, transfer_learning):
    
    batch_size=X_data_train.shape[0]  # Set batch size to the entire dataset
    
    # Initialize TensorBoard logging directory for each trial
    log_dir = f'{model_path}_TL_{transfer_learning}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)  # Initialize SummaryWriter for TensorBoard
        
    # setting the device right
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Step 1: Create DataLoader objects for training, validation testing datasets
    # Convert pandas DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_data_train.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_data_val.values, dtype=torch.float32).to(device)
 
    
    if task_type == 'classification':
        
        #getting the y data and filling missing values with -1
        y_data_train_tensor = torch.tensor(y_data_train[task].fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32).to(device)
        y_data_val_tensor = torch.tensor(y_data_val[task].fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32).to(device)
        
        # Create TensorDataset
        train_dataset = TensorDataset(X_train_tensor, y_data_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_data_val_tensor)
            
    elif task_type == 'cox':
        # Convert pandas DataFrames to PyTorch tensors
        y_duration_train_tensor = torch.tensor(y_data_train[task].fillna(-1).values, dtype=torch.float32).to(device)
        y_event_train_tensor = torch.tensor(y_data_train[task_event].fillna(-1).values, dtype=torch.float32).to(device)
        y_duration_val_tensor = torch.tensor(y_data_val[task].fillna(-1).values, dtype=torch.float32).to(device)
        y_event_val_tensor = torch.tensor(y_data_val[task_event].fillna(-1).values, dtype=torch.float32).to(device)

        # Create TensorDataset
        train_dataset = TensorDataset(X_train_tensor, y_duration_train_tensor, y_event_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_duration_val_tensor, y_event_val_tensor)


    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    # Step 2: Initialize the fine-tuning model with the same architecture as the pre-trained VAE
    fine_tune_VAE = SupervisedVAE (task_type=task_type, 
                        num_classes=num_classes,
                        input_size=pretrain_VAE.input_size, 
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
    fine_tune_VAE.to(device)
    
    # Step 3: Transfer learning logic
    if transfer_learning:
        # Load pre-trained encoder weights
        fine_tune_VAE.encoder.load_state_dict(pretrain_VAE.encoder.state_dict())
        # Load pre-trained decoder weights
        fine_tune_VAE.decoder.load_state_dict(pretrain_VAE.decoder.state_dict())
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
    fine_tune_VAE.kl_weight = 2.0
    lambda_sup=1.0


    best_val_loss = np.inf
    best_model=None
    
    # Training function (no y_batch since we only have x_batch)
    for epoch in range(num_epochs):

        fine_tune_VAE.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        train_sup_loss = 0
        
        # # KL-annealing schedule: linearly increase KL weight over the annealing period
        # kl_weight = kl_start_weight + (kl_max_weight - kl_start_weight) * min(1, epoch / kl_annealing_epochs)
        
        #fine_tune_VAE.kl_weight = kl_weight

        
        for batch in train_loader:
            if task_type == 'classification':
                x_batch, y_batch = batch
                y_batch = y_batch.to(device)
            elif task_type == 'cox':
                x_batch, duration_batch, event_batch = batch
                duration_batch = duration_batch.to(device)
                event_batch = event_batch.to(device)
            
            x_batch = x_batch[0].to(device)  # Extract the actual data from the tuple
            
            optimizer.zero_grad()
            
            x_recon, mu, log_var, supervised_out = fine_tune_VAE(x_batch)
            
            # Compute loss
            if task_type == 'classification':
                recon_loss, kl_loss, sup_loss, loss = fine_tune_VAE.loss_function(x_batch, x_recon, mu, log_var, supervised_out, y_batch, lambda_sup=lambda_sup)
            elif task_type == 'cox':
                recon_loss, kl_loss, sup_loss, loss = fine_tune_VAE.loss_function(x_batch, x_recon, mu, log_var, supervised_out, y=duration_batch, event=event_batch, lambda_sup=lambda_sup)
            
            
            loss += l1_penalty(fine_tune_VAE)  # Add L1 regularization

            loss.backward()  # Backward pass (compute gradients)

            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(fine_tune_VAE.parameters(), max_norm=1.0)
        
            optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_sup_loss += sup_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_kl_loss = train_kl_loss / len(train_loader)
        avg_train_sup_loss = train_sup_loss / len(train_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_recon', avg_train_recon_loss, epoch)
        writer.add_scalar('Loss/train_kl', avg_train_kl_loss, epoch)
        writer.add_scalar('Loss/train_sup', avg_train_sup_loss, epoch)
        
        # Validation loop
        fine_tune_VAE.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_sup_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if task_type == 'classification':
                    x_batch, y_batch = batch
                    y_batch = y_batch.to(device)
                elif task_type == 'cox':
                    x_batch, duration_batch, event_batch = batch
                    duration_batch = duration_batch.to(device)
                    event_batch = event_batch.to(device)
                    
                x_batch = x_batch[0].to(device)  # Extract the actual data from the tuple
                x_recon, mu, log_var, supervised_out = fine_tune_VAE(x_batch)
                
                # Compute validation loss
                if task_type == 'classification':
                    recon_loss, kl_loss, sup_loss, loss = fine_tune_VAE.loss_function(x_batch, x_recon, mu, log_var, supervised_out, y_batch, lambda_sup=lambda_sup)
                elif task_type == 'cox':
                    recon_loss, kl_loss, sup_loss, loss = fine_tune_VAE.loss_function(x_batch, x_recon, mu, log_var, supervised_out, duration=duration_batch, event=event_batch, lambda_sup=lambda_sup)
               
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                val_sup_loss += sup_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_kl_loss = val_kl_loss / len(val_loader)
        avg_val_sup_loss = val_sup_loss / len(val_loader)
        

        # scheduler.step(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Log validation loss to TensorBoard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Loss/val_recon', avg_val_recon_loss, epoch)
        writer.add_scalar('Loss/val_kl', avg_val_kl_loss, epoch)
        writer.add_scalar('Loss/val_sup', avg_val_sup_loss, epoch)
        

        if avg_val_loss < best_val_loss and epoch > kl_annealing_epochs:
            best_val_loss = avg_val_loss
            best_model = fine_tune_VAE

   
    # Close TensorBoard writer
    writer.close()

    # Return the fine-tuned model and validation loss for Optuna to minimize
    return best_model, best_val_loss, log_dir
