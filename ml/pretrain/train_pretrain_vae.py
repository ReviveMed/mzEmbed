'''
this funtion develop a pre-trained VAE models


'''

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
import os
import json

from models.models_VAE import VAE  # Assuming the VAE class is in vae_model.py


class ModelStorage:
    def __init__(self, **kwargs):
        """
        Stores model configurations and kwargs.

        Parameters:
            **kwargs: The keyword arguments to store (model hyperparameters and settings).
        """
        self.kwargs = kwargs

    def parse_kwargs(self):
        """
        Parse and return the model hyperparameters from the stored kwargs.
        """
        parsed_kwargs = {
            'input_size': self.kwargs.get('input_size', 1),
            'latent_size': self.kwargs.get('latent_size', 1),
            'num_hidden_layers': self.kwargs.get('num_hidden_layers', 1),
            'dropout_rate': self.kwargs.get('dropout_rate', 0.2),
            'activation': self.kwargs.get('activation', 'leakyrelu'),
            'use_batch_norm': self.kwargs.get('use_batch_norm', False),
            'act_on_latent_layer': self.kwargs.get('act_on_latent_layer', False),
            'learning_rate': self.kwargs.get('learning_rate', 1e-3),
            'epochs': self.kwargs.get('epochs', 100),
            'save_path': self.kwargs.get('save_path', './'),
            'model_name': self.kwargs.get('model_name', 'pretrained_vae.pth')
        }
        return parsed_kwargs

    def save_storage(self, file_path):
        """
        Save the stored kwargs (hyperparameters) to a JSON file.

        Parameters:
            file_path: The file path where the kwargs will be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(self.kwargs, f)
        print(f'Model kwargs saved to {file_path}')

    def load_storage(self, file_path):
        """
        Load the stored kwargs (hyperparameters) from a JSON file.

        Parameters:
            file_path: The file path from which the kwargs will be loaded.
        """
        with open(file_path, 'r') as f:
            self.kwargs = json.load(f)
        print(f'Model kwargs loaded from {file_path}')
        return self.kwargs




def pretrain_vae(X_data_train, y_data_train, X_data_val, y_data_val, 
                 X_data_test, y_data_test, **kwargs):
    """
    Pretrains a Variational Autoencoder (VAE) on the provided data.
    
    Parameters:
        X_data_train: numpy array, training input data
        y_data_train: numpy array, training labels (not used for VAE pretraining)
        X_data_val: numpy array, validation input data
        y_data_val: numpy array, validation labels (not used for VAE pretraining)
        X_data_test: numpy array, test input data
        y_data_test: numpy array, test labels (not used for VAE pretraining)
        **kwargs: Additional keyword arguments for the VAE model configuration
    """
    
    # Automatically infer the input size from X_data_train
    input_size = X_data_train.shape[1]
    
    # Parse the kwargs using ModelStorage and include the inferred input_size
    storage = ModelStorage(input_size=input_size, **kwargs)
    model_kwargs = storage.parse_kwargs()

    # Convert the data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_data_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_data_val.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_data_test.values, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the VAE model with parsed kwargs
    vae = VAE(**model_kwargs)
    vae.init_layers()  # Initialize weights
    
    # Define optimizer
    optimizer = optim.Adam(vae.parameters(), lr=model_kwargs['learning_rate'])

    # Training loop
    epochs = model_kwargs['epochs']
    for epoch in range(epochs):
        vae.train()  # Set model to training mode
        train_loss = 0
        for batch in train_loader:
            batch = batch[0]  # Extract the tensor

            # Forward pass
            optimizer.zero_grad()
            recon, mu, log_var = vae(batch)

            # Compute the loss
            loss = vae.loss(batch, recon, mu, log_var)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss/len(train_loader)}')

        # Validate on validation data
        vae.eval()  # Set to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0]
                recon, mu, log_var = vae(batch)
                loss = vae.loss(batch, recon, mu, log_var)
                val_loss += loss.item()

        print(f'Validation Loss: {val_loss/len(val_loader)}')

    # Save the trained model and model hyperparameters
    vae.save_state_to_path(model_kwargs['save_path'], model_kwargs['model_name'])
    storage.save_storage(os.path.join(model_kwargs['save_path'], 'model_hyperparameters.json'))

    print(f"Model saved to {os.path.join(model_kwargs['save_path'], model_kwargs['model_name'])}")

   