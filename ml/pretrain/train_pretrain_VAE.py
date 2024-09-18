'''
this funtion develop a pre-trained VAE models


'''

import os
import json
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import optuna




#importing my own functions and classes
from models.models_VAE import VAE  # Assuming the VAE class is in vae_model.py
from pretrain.latent_task_predict import log_reg_multi_class, ridge_regression_predict



#### setting the same seed for everything
random_seed = 42

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




class PretrainVAE(VAE):
    def __init__(self, **kwargs):
        # Extract VAE-specific arguments and initialize the parent VAE class
        vae_kwargs = {
            'input_size': int(kwargs.get('input_size', 1)),
            'latent_size': int(kwargs.get('latent_size', 1)),
            'num_hidden_layers': int(kwargs.get('num_hidden_layers', 1)),
            'dropout_rate': float(kwargs.get('dropout_rate', 0.2)),
            'activation': kwargs.get('activation', 'leakyrelu'),
            'use_batch_norm': kwargs.get('use_batch_norm', False),
            'act_on_latent_layer': kwargs.get('act_on_latent_layer', False),
            'verbose': kwargs.get('verbose', False)
        }
        super().__init__(**vae_kwargs)

        # Training parameters
        self.learning_rate = float(kwargs.get('learning_rate', 0.001))
        self.weight_decay = float(kwargs.get('weight_decay', 1e-5))
        self.num_epochs = int(kwargs.get('num_epochs', 10))
        self.batch_size = int(kwargs.get('batch_size', 64))
        self.patience = int(kwargs.get('patience', 5))  # Early stopping patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move model to device

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
    def init_layers(self):
        self.apply(self._init_weights)  # Initialize weights using a helper function

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)



def pretrain_vae(X_data_train, X_data_val, trial_name, trial_id, save_path, **kwargs):
    """
    Pretrains a Variational Autoencoder (VAE) on the provided data.
    Recon loss only
    
    Parameters:
        X_data_train: Panda data-frame, training input data
        X_data_val: Panda data-frame, validation input data
        **kwargs: Additional keyword arguments for the VAE model configuration
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Automatically infer the input size from X_data_train
    input_size = X_data_train.shape[1]
    kwargs['input_size'] = input_size
    
    # Convert the data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_data_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_data_val.values, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    
    # Initialize the pretrain VAE model with parsed kwargs
    vae = PretrainVAE(**kwargs).to(device)
    vae.init_layers()  # Initialize weights

    # Set early stopping parameters
    patience = int(kwargs.get('patience', 0))  # 0 means no early stopping
    best_val_loss = np.inf
    epochs_no_improve = 0

    # KL-Annealing parameters
    kl_annealing_epochs = kwargs.get('kl_annealing_epochs', 10)  # Number of epochs for KL annealing
    kl_start_weight = kwargs.get('kl_start_weight', 0.0)  # Initial weight for KL divergence
    kl_max_weight = kwargs.get('kl_max_weight', 1.0)  # Maximum weight for KL divergence

    
    # Training loop
    num_epochs = int(kwargs.get('num_epochs', 10))
    for epoch in range(num_epochs):
        vae.train()  # Set model to training mode
        train_loss = 0

        # KL-annealing schedule: linearly increase KL weight over the annealing period
        kl_weight = kl_start_weight + (kl_max_weight - kl_start_weight) * min(1, epoch / kl_annealing_epochs)

        for batch in train_loader:
            batch = batch[0].to(device)  # Move batch to device

            # Forward pass
            vae.optimizer.zero_grad()
            recon, mu, log_var = vae(batch)

            # Compute the loss
            vae.kl_weight = kl_weight
            loss = vae.loss(batch, recon, mu, log_var)

            # Backpropagation and optimization
            loss.backward()

            # Log gradient norms
            # total_norm = 0
            # for p in vae.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"Gradient norm: {total_norm}")

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

            # Update weights
            vae.optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')

        # Validation loop
        vae.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(vae.device)
                recon, mu, log_var = vae(batch)
                loss = vae.loss(batch, recon, mu, log_var)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

        # Early stopping logic
        if patience > 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    # Create a directory for this trial
    model_folder = os.path.join(save_path, trial_name, f"trial_{trial_id}")
    os.makedirs(model_folder, exist_ok=True)

    # Save the encoder state
    encoder_path = os.path.join(model_folder, f"encoder_state.pt")
    torch.save(vae.encoder.state_dict(), encoder_path)

    # Save the model hyperparameters
    hyperparameters = vae.get_hyperparameters()

    # Filter out non-serializable values
    serializable_hyperparameters = {k: v for k, v in hyperparameters.items() if isinstance(v, (int, float, str, bool, list, dict))}
    
    hyperparams_save_path = os.path.join(model_folder, 'model_hyperparameters.json')
    with open(hyperparams_save_path, 'w') as f:
        json.dump(serializable_hyperparameters, f, indent=4)

    ### evaluating pre-trained model on the latent task prediction and loss calculation
    
    
        

    print(f"Encoder saved to {encoder_path}")
    print(f"Hyperparameters saved to {hyperparams_save_path}")

    return avg_val_loss



def objective(trial, X_data_train, X_data_val, combined_params, trial_name, save_path, **kwargs):
    """
    Objective function for Optuna to minimize the loss using different hyperparameters.
    
    Parameters:
        trial: Optuna trial object to suggest hyperparameters.
        X_data_train: Pandas DataFrame, training input data.
        X_data_val: Pandas DataFrame, validation input data.
        combined_params: Dictionary containing the parameter ranges and default kwargs.
        trial_name: The name of the hyperparameter optimization run.
        save_path: Path to save the models and encoder state.
        **kwargs: Additional keyword arguments for the VAE model configuration.
        
    Returns:
        Validation loss for the current trial.
    """

    # Helper function to determine whether the parameter should be tuned or kept fixed
    def suggest_param(param_name, param_value):
        if isinstance(param_value, tuple):
            # Integer parameter with step size
            if len(param_value) == 3:
                min_val, max_val, step = param_value
                if isinstance(min_val, int) and isinstance(max_val, int):  # Ensure it's an integer range
                    return trial.suggest_int(param_name, min_val, max_val, step=step)
                else:
                    # If it's not an integer range, treat it as a float range
                    return trial.suggest_float(param_name, min_val, max_val)
                # Float parameter without step size
            elif len(param_value) == 2:
                if param_name in ['learning_rate', 'weight_decay']:
                    return trial.suggest_loguniform(param_name, param_value[0], param_value[1])
                else:
                    return trial.suggest_float(param_name, param_value[0], param_value[1])
        elif isinstance(param_value, list):
            # Categorical parameter (e.g., activation functions, optimizer)
            return trial.suggest_categorical(param_name, param_value)
        else:
            # Fixed value (not tuned)
            return param_value

    # Model hyperparameters
    latent_size = suggest_param('latent_size', combined_params.get('latent_size'))
    num_hidden_layers = suggest_param('num_hidden_layers', combined_params.get('num_hidden_layers'))
    dropout_rate = suggest_param('dropout_rate', combined_params.get('dropout_rate'))  # Only one dropout rate is needed here
    learning_rate = suggest_param('learning_rate', combined_params.get('learning_rate'))
    weight_decay = suggest_param('weight_decay', combined_params.get('weight_decay'))
    activation = suggest_param('activation', combined_params.get('activation', 'leakyrelu'))

    # Batch normalization (True or False)
    batch_norm = suggest_param('use_batch_norm', combined_params.get('use_batch_norm', False))

    # Training hyperparameters
    batch_size = suggest_param('batch_size', combined_params.get('batch_size', 64))  # Tune batch size
    patience = suggest_param('patience', combined_params.get('patience', 5))  # Tune patience
    num_epochs = suggest_param('num_epochs', combined_params.get('num_epochs', 20))  # Tune num_epochs

    # Update kwargs with all the hyperparameters
    kwargs.update({
        'latent_size': latent_size,
        'num_hidden_layers': num_hidden_layers,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'activation': activation,
        'use_batch_norm': batch_norm,
        'input_size': X_data_train.shape[1],  # Automatically determine input size
        'batch_size': batch_size,
        'patience': patience,
        'num_epochs': num_epochs
    })

    # Call the pretrain_vae function to train the model and return the validation loss
    avg_val_loss = pretrain_vae(X_data_train, X_data_val, trial_name, trial.number, save_path, **kwargs)

    return avg_val_loss




def optimize_hyperparameters(X_data_train, X_data_val, param_ranges, trial_name, save_path,  n_trials=50, **kwargs):
    """
    Function to run hyperparameter optimization using Optuna.
    
    Parameters:
        X_data_train: Pandas DataFrame, training input data.
        X_data_val: Pandas DataFrame, validation input data.
        param_ranges: Dictionary containing the parameter ranges for optimization.
        n_trials: Number of trials to run for Optuna optimization.
        **kwargs: Additional keyword arguments for the VAE model configuration.
        
    Returns:
        Best trial object containing optimal hyperparameters.
    """

    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_data_train, X_data_val, param_ranges, trial_name, save_path, **kwargs), n_trials=n_trials)

    print(f"Best trial: {study.best_trial}")
    print(f"Best hyperparameters: {study.best_trial.params}")
    
    return study.best_trial, study
