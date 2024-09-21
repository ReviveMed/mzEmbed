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
import subprocess
import threading



#importing my own functions and classes
from models.models_VAE import VAE  # Assuming the VAE class is in vae_model.py
from pretrain.eval_pretrained_VAE import evalute_pretrain_latent_extra_task


#### setting the same seed for everything
random_seed = 42

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import subprocess
import threading


#this is only needed for locally launching the dashboard
def launch_dashboard(trial_name, storage):
    """
    Launches the Optuna dashboard in a separate thread.
    
    Parameters:
        trial_name: Name of the Optuna trial.
        storage: Path to the persistent storage (e.g., sqlite:///<path_to_db>.db).
    """
    cmd = f"optuna-dashboard --host 0.0.0.0 --port 8081 {storage}"
    
    def run_dashboard():
        subprocess.run(cmd, shell=True)
    
    # Launch the dashboard in a new thread to run in the background
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.start()
    print(f"Optuna dashboard launched at http://localhost:8081 for trial '{trial_name}'.")




def _init_weights_xavier(layer,gain=1.0):
        if hasattr(layer, '.weight'):
            # nn.init.xavier_normal_(layer.weight.data,gain=gain)
            nn.init.xavier_uniform_(layer.weight.data,gain=gain)
        else:
            if hasattr(layer, 'children'):
                for child in layer.children():
                    _init_weights_xavier(child,gain=gain)



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
        self.l1_reg = float(kwargs.get('1_reg', 0))
        self.weight_decay = float(kwargs.get('weight_decay', 1e-5))
        self.noise_factor = float(kwargs.get('noise_factor', 0))
        self.num_epochs = int(kwargs.get('num_epochs', 50))
        self.batch_size = int(kwargs.get('batch_size', 94))
        self.patience = int(kwargs.get('patience', 5))  # Early stopping patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move model to device

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
    def init_layers(self):
        #self.apply(self._init_weights)  # Initialize weights using a helper function
        _init_weights_xavier(self.encoder,gain=nn.init.calculate_gain(self.activation))
        _init_weights_xavier(self.decoder,gain=nn.init.calculate_gain(self.activation))  
                
    # def _init_weights(self, layer):
    #     # if isinstance(layer, nn.Linear):
    #     #     nn.init.xavier_normal_(layer.weight)
    #     #     if layer.bias is not None:
    #     #         nn.init.zeros_(layer.bias)
        


def pretrain_vae(X_data_train, X_data_val,  X_data_test, y_data_train, y_data_val, y_data_test, trial_name, trial_id, save_path, **kwargs):
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

    # KL-Annealing parameters & epochs
    num_epochs = int(kwargs.get('num_epochs', 50))
    
    kl_annealing_epochs_default = 0.75 * num_epochs
    kl_annealing_epochs = kwargs.get('kl_annealing_epochs', kl_annealing_epochs_default)  # Number of epochs for KL annealing
    kl_start_weight = kwargs.get('kl_start_weight', 0.0)  # Initial weight for KL divergence
    kl_max_weight = kwargs.get('kl_max_weight', 1.0)  # Maximum weight for KL divergence

    
    # Training loop
    for epoch in range(num_epochs):
        vae.train()  # Set model to training mode
        train_loss = 0

        # KL-annealing schedule: linearly increase KL weight over the annealing period
        kl_weight = kl_start_weight + (kl_max_weight - kl_start_weight) * min(1, epoch / kl_annealing_epochs)
        vae.kl_weight = kl_weight

        for batch in train_loader:
            batch = batch[0].to(device)  # Move batch to device

            if vae.noise_factor > 0:
                # Inject noise into the batch
                noisy_batch = batch + vae.noise_factor * torch.randn_like(batch)
                # Optionally, clamp the values to be between 0 and 1 if your data is normalized
                #noisy_batch = torch.clamp(noisy_batch, 0., 1.)
            else:
                noisy_batch = batch

    
            # Forward pass
            vae.optimizer.zero_grad()
            recon, mu, log_var = vae(noisy_batch)

            # Compute the loss
            loss = vae.loss(noisy_batch, recon, mu, log_var)
            
            # Compute L1 regularization term in the training loop
            l1_norm = sum(p.abs().sum() for p in vae.parameters())
            
            # Add the L1 regularization to the loss
            loss = loss + vae.l1_reg * l1_norm

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
    #evlaute the model on the latent task
    if avg_val_loss is not None:
        evalute_pretrain_latent_extra_task(vae, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, model_folder)
    
        
    print(f"Encoder saved to {encoder_path}")
    print(f"Hyperparameters saved to {hyperparams_save_path}")

    return avg_val_loss



   

def objective(trial, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, combined_params, trial_name, save_path, **kwargs):
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

    # Case-by-case parsing based on parameter type (range, fixed, or categorical)
    

    # Latent size: integer range or fixed
    if isinstance(combined_params['latent_size'], tuple):
        min_val, max_val, step = combined_params['latent_size']
        latent_size = trial.suggest_int('latent_size', min_val, max_val, step=step)
    else:
        latent_size = combined_params['latent_size']

    # Number of hidden layers: integer range or fixed
    if isinstance(combined_params['num_hidden_layers'], tuple):
        min_val, max_val, step = combined_params['num_hidden_layers']
        num_hidden_layers = trial.suggest_int('num_hidden_layers', min_val, max_val, step=step)
    else:
        num_hidden_layers = combined_params['num_hidden_layers']

    # Dropout rate: float range or fixed
    if isinstance(combined_params['dropout_rate'], tuple):
        min_val, max_val, step = combined_params['dropout_rate']
        dropout_rate = trial.suggest_float('dropout_rate', min_val, max_val, step=step)
    else:
        dropout_rate = combined_params['dropout_rate']
        
    # Noise factor: float range or fixed
    if isinstance(combined_params['noise_factor'], tuple):
        min_val, max_val, step = combined_params['noise_factor']
        noise_factor = trial.suggest_float('noise_factor', min_val, max_val, step=step)
    else:
        noise_factor = combined_params['noise_factor']

    # Learning rate: loguniform or fixed
    if isinstance(combined_params['learning_rate'], tuple):
        learning_rate = trial.suggest_loguniform('learning_rate', combined_params['learning_rate'][0], combined_params['learning_rate'][1])
    else:
        learning_rate = combined_params['learning_rate']
        
    # Learning rate: loguniform or fixed
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

    # Update kwargs with all the hyperparameters
    kwargs.update({
        'latent_size': latent_size,
        'num_hidden_layers': num_hidden_layers,
        'dropout_rate': dropout_rate,
        'noise_factor': noise_factor,
        'learning_rate': learning_rate,
        'l1_reg': l1_reg,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'patience': patience,
        'num_epochs': num_epochs,
        'input_size': X_data_train.shape[1],  # Automatically determine input size
    })


    # Call the pretrain_vae function to train the model and return the validation loss
    avg_val_loss = pretrain_vae(X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, trial_name, trial.number, save_path, **kwargs)

    return avg_val_loss




# def optimize_hyperparameters(X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test,  param_ranges, trial_name, save_path,  n_trials=50, **kwargs):
#     """
#     Function to run hyperparameter optimization using Optuna.
    
#     Parameters:
#         X_data_train: Pandas DataFrame, training input data.
#         X_data_val: Pandas DataFrame, validation input data.
#         param_ranges: Dictionary containing the parameter ranges for optimization.
#         n_trials: Number of trials to run for Optuna optimization.
#         **kwargs: Additional keyword arguments for the VAE model configuration.
        
#     Returns:
#         Best trial object containing optimal hyperparameters.
#     """

    
#     study = optuna.create_study(direction='minimize')
#     study.optimize(lambda trial: objective(trial, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, param_ranges, trial_name, save_path, **kwargs), n_trials=n_trials)

#     print(f"Best trial: {study.best_trial}")
#     print(f"Best hyperparameters: {study.best_trial.params}")
    
#     return study.best_trial, study




def optimize_hyperparameters(X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test,  param_ranges, trial_name, save_path, n_trials=50, storage='mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB', **kwargs):
    """
    Function to run hyperparameter optimization using Optuna with dashboard support.
    
    Parameters:
        X_data_train: Pandas DataFrame, training input data.
        X_data_val: Pandas DataFrame, validation input data.
        X_data_test: Pandas DataFrame, test input data.
        y_data_train: Pandas DataFrame, training target data.
        y_data_val: Pandas DataFrame, validation target data.
        y_data_test: Pandas DataFrame, test target data.
        param_ranges: Dictionary containing the parameter ranges for optimization.
        trial_name: Name for the trial.
        save_path: Path to save the models and trial results.
        n_trials: Number of trials to run for Optuna optimization.
        storage: Path to the persistent storage for Optuna (e.g., sqlite:///<path_to_db>.db).
        **kwargs: Additional keyword arguments for the VAE model configuration.
        
    Returns:
        Best trial object containing optimal hyperparameters.
    """
    
    # Launch the Optuna dashboard in the background
    #launch_dashboard(trial_name, storage)
    
    # Create or load the existing Optuna study
    study = optuna.create_study(
        direction='minimize',
        study_name=trial_name,
        storage=storage,
        load_if_exists=True
    )

    # Optimize the hyperparameters using the objective function
    study.optimize(lambda trial: objective(trial, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, param_ranges, trial_name, save_path, **kwargs), n_trials=n_trials)

    # Output the best trial and hyperparameters
    print(f"Best trial: {study.best_trial}")
    print(f"Best hyperparameters: {study.best_trial.params}")
    
    return study.best_trial, study
