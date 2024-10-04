
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

import copy
import os

# Make sure to disable non-deterministic operations if exact reproducibility is crucial
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed=None):
    """
    Sets the seed for generating random numbers and returns the seed used.
    
    Parameters:
    - seed (int): The seed value to use. If None, a random seed will be generated.

    Returns:
    - seed (int): The seed value used.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)  # Generate a random seed if none is provided

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed



# Define L1 regularization
def l1_regularization(model, l1_reg_weight):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_reg_weight * l1_norm



# Define the custom masked MSE loss for numerical targets
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        inputs = inputs.squeeze()  # Squeeze the inputs to match the shape of targets
        # Create a mask to ignore invalid targets (e.g., negative values or NaNs)
        valid_mask = torch.isfinite(targets)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Apply the mask to inputs and targets
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        # Calculate the masked MSE loss
        loss = self.criterion(inputs, targets)
        return loss.mean()





class FineTuneModel(nn.Module):
    def __init__(self, VAE_model, num_layers_to_retrain=1, add_post_latent_layers=False, num_post_latent_layers=1, post_latent_layer_size=128, dropout=0.2):
        super(FineTuneModel, self).__init__()
        self.latent_size = VAE_model.latent_size  # Keep the original latent size from the VAE
        self.encoder = VAE_model.encoder
        self.freeze_encoder_except_last(num_layers_to_retrain)
        self.latent_mode = False  # Toggle between raw input and latent mode

        # If additional layers are needed after latent space
        if add_post_latent_layers:
            layers = []
            input_size = self.latent_size  # Start with the original latent size
            for _ in range(num_post_latent_layers):
                layers.append(nn.Linear(input_size, post_latent_layer_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_size = post_latent_layer_size  # Update for next layer input size

            self.post_latent_layers = nn.Sequential(*layers)
            self.post_latent_layer_size = post_latent_layer_size  # Keep track of post-latent size
        else:
            self.post_latent_layers = None
            self.post_latent_layer_size = self.latent_size  # Use latent size if no post-latent layers
            
        # For regression task (MSE Loss)
        self.regressor_head = nn.Sequential(
            nn.Linear(self.post_latent_layer_size, 1)  
        )

    def freeze_encoder_except_last(self, num_layers_to_retrain):
        """Freeze all trainable (Linear) layers in the encoder except the specified number of layers."""
        
        # Get only the Linear layers from the encoder's Sequential block
        encoder_layers = [layer for layer in self.encoder.network.children() if isinstance(layer, nn.Linear)]
        # Exclude the last Linear layer (latent space)
        hidden_layers = encoder_layers #[:-1]

        # Get the total number of hidden layers (excluding latent)
        total_layers = len(hidden_layers)
                
        # If num_layers_to_retrain exceeds total layers, set it to total layers
        if num_layers_to_retrain > total_layers:
            print(f"num_layers_to_retrain exceeds total layers ({total_layers}). Setting num_layers_to_retrain to {total_layers}.")
            num_layers_to_retrain = total_layers

        # print(f"\nTotal hidden layers (excluding latent): {total_layers}")
        # print(f"Number of layers to retrain: {num_layers_to_retrain}")
        
        # Freeze all parameters in the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # # Print the state of each layer before unfreezing
        # for name, param in self.encoder.named_parameters():
        #     print(f"{name}: requires_grad = {param.requires_grad}")


        # Unfreeze the specified number of layers from the end of the hidden layers
        if num_layers_to_retrain > 0:
            for layer in hidden_layers[-num_layers_to_retrain:]:
                for param in layer.parameters():
                        param.requires_grad = True
        
        # Print the state of each layer after unfreezing
        # # print("\nAfter unfreezing:")
        # for name, param in self.encoder.named_parameters():
        #     print(f"{name}: requires_grad = {param.requires_grad}")


    def forward(self, x):
        if not self.latent_mode:  # If in raw input mode
            encoder_output = self.encoder(x)  # Get latent representation from encoder
            
            # Slice the first half of the encoder output to get `mu`
            mu = encoder_output[:, :self.latent_size]  # Get the first half for `mu`
            x = mu  # Use `mu` as the input to the regressor_head

        if self.post_latent_layers:
            x = self.post_latent_layers(x)

        # Pass the latent representation (or the output of post-latent layers) to the regressor_head
        return self.regressor_head(x)
    



def retrain_pretrain_num_task(VAE_model,model_path, X_train, y_data_train, X_val, y_data_val, num_layers_to_retrain=1, add_post_latent_layers=False, num_post_latent_layers=1, post_latent_layer_size=128, num_epochs=20, batch_size=32, learning_rate=1e-4, dropout=0.2, l1_reg_weight=0.0, l2_reg_weight=0.0, latent_passes=20, seed=None, patience=0):
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    # Initialize the fine-tuning model
    model = FineTuneModel(VAE_model, num_layers_to_retrain=num_layers_to_retrain, add_post_latent_layers=add_post_latent_layers, num_post_latent_layers=num_post_latent_layers, post_latent_layer_size=post_latent_layer_size, dropout=dropout)
    model = model.to(device)
    
    # Define the loss function
    criterion = MaskedMSELoss()  
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_reg_weight)

    # Convert pandas DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_data_train_tensor = torch.tensor(y_data_train.fillna(-1).values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_data_val_tensor = torch.tensor(y_data_val.fillna(-1).values, dtype=torch.float32).to(device)

    #print (X_train_tensor.shape, y_data_train_tensor.shape, X_val_tensor.shape, y_data_val_tensor.shape)

    # Create TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_data_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_data_val_tensor)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda worker_id: set_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Early Stopping Setup
    best_val_metric = -float('inf')  # Best validation metric, AUC for binary and F1-score for multi-class
    best_model = None  # Store the state of the best model
    patience_counter = 0  # Counter to track patience

    # DataFrame to store metrics per epoch
    metrics_per_epoch = pd.DataFrame(columns=['Epoch', 'Mean sq. error', 'Mean avg. error', 'R2', 'Validation Loss'])
    best_val_metrics_df = pd.DataFrame(columns=['Epoch', 'Mean sq. error', 'Mean avg. error', 'R2', 'Validation Loss'])

    # Initialize TensorBoard logging directory for each trial
    log_dir = model_path
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)  # Initialize SummaryWriter for TensorBoard
    
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        model.latent_mode = False  # Train using raw input data

        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # Latent averaging with fine-tuning of the last encoder layer
            latent_reps_train = []

            for _ in range(latent_passes):  # Generate multiple latent representations
                latent_rep = model.encoder(inputs.to(device))
                mu = latent_rep[:, :model.latent_size]  
                latent_reps_train.append(mu)
            latent_rep_train = torch.mean(torch.stack(latent_reps_train), dim=0)
            
            #print (latent_rep_train.shape)

            # Set the model to latent mode to ensure it processes latent representations correctly
            model.latent_mode = True

            # Forward pass using averaged latent representations
            outputs = model.forward(latent_rep_train)
            loss = criterion(outputs, labels.to(device))

            # Add L1 regularization
            l1_norm = l1_regularization(model, l1_reg_weight)
            loss = loss + l1_norm

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss=running_loss/len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}')
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation
        model.eval()
        #model.latent_mode = True  # Use precomputed latent representations for validation
        
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_preds_np=[]
        all_labels_np=[]

        # Validation loop (corrected)
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Latent averaging for validation
                latent_reps_val = []
                for _ in range(latent_passes):  # Multiple passes through the encoder
                    latent_rep = model.encoder(inputs.to(device))  # Ensure inputs are on the same device
                    mu = latent_rep[:, :model.latent_size]  # Slice out mu (first half of the output)
                    latent_reps_val.append(mu)
                latent_rep_val = torch.mean(torch.stack(latent_reps_val), dim=0)

                #print (latent_rep_val.shape)
                model.latent_mode = True  # Use precomputed latent representations for validation
                # Forward pass using averaged latent representations
                outputs = model(latent_rep_val)

                # Switch back to not latent mode
                model.latent_mode = False

                # Ensure predictions and labels have the same batch size
                #print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")  # Debugging step
            
                loss = criterion(outputs, labels.to(device))  # Ensure labels are on the same device

                # Add L1 regularization to validation loss as well
                val_loss += (loss + l1_regularization(model, l1_reg_weight)).item()
                
                # Move predictions and labels back to CPU
                outputs_np = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # Ensure valid_mask is applied to both predictions and labels
                valid_mask = (labels_np >= 0)  # Assuming you want to ignore invalid labels

                # Append predicted probabilities for all samples in the batch
                all_preds.extend(outputs_np[valid_mask])  # Append predictions for all batches
                all_labels.extend(labels_np[valid_mask])  # Use only valid labels
            
                
            # Convert lists to NumPy arrays
            all_preds_np = np.array(all_preds)  
            all_labels_np = np.array(all_labels)
            
            # Calculate regression metrics
            mse = mean_squared_error(all_labels_np, all_preds_np)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(all_labels_np, all_preds_np)
            r2 = r2_score(all_labels_np, all_preds_np)  # R-squared metric

            val_metric = mae  # Use MSE as the validation metric


        avg_val_loss = val_loss / len(val_loader)
        # Log validation loss to TensorBoard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
         # Log validation loss to TensorBoard
        writer.add_scalar('MSE val', mse, epoch)
        writer.add_scalar('MAE val', mae, epoch)
        
        
        # Create a DataFrame with the metrics for the current epoch
        metrics_df = pd.DataFrame({
            'Epoch': [epoch + 1],
            'Mean sq. error': [mse],
            'Mean avg. error': [mae],
            'R2': [r2],
            'Validation Loss': [val_loss / len(val_loader)]
        })

        # Concatenate with the existing DataFrame
        metrics_per_epoch = pd.concat([metrics_per_epoch, metrics_df], ignore_index=True)

        print(f'Validation Loss: {val_loss/len(val_loader)}, MSE: {mse}, MAE: {mae}')

        
        # saving the best model based on the validation metric
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model, f'{model_path}/best_model.pth')
            best_val_metrics_df=metrics_df
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # If no improvement for `patience` epochs, stop training
        if patience > 0:
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        
        
        
    writer.close()  # Close the TensorBoard writer
    print('Fine-tuning completed.')
    return best_val_metrics_df




