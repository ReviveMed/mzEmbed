# developing a fine-tune VAE with transfer learning from a pretrained model 
'''
This script is used to fine-tune a VAE model with transfer learning from a pretrained model.
The pretrained model is a VAE model trained. The pretrained model is loaded
and the encoder part of the model is used to initialize the encoder part of the fine-tuned model.



'''


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models_VAE import VAE


def _reset_params(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()  # Call the layer's reset_parameters method if it exists


def fine_tune_vae(pretrain_VAE, X_data_train, 
                  X_data_val,  
                  X_data_test, 
                  batch_size=64, num_epochs=10, learning_rate=1e-4, 
                  dropout_rate=0.2, l1_reg_weight=0.0, l2_reg_weight=0.0, 
                  transfer_learning=True):
    
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
    fine_tune_VAE = VAE(input_size=pretrain_VAE.input_size, 
                        hidden_size=pretrain_VAE.hidden_size, 
                        latent_size=pretrain_VAE.latent_size, 
                        num_hidden_layers=pretrain_VAE.num_hidden_layers, 
                        dropout_rate=dropout_rate,  # Pass dropout_rate here
                        activation=pretrain_VAE.activation, 
                        use_batch_norm=pretrain_VAE.use_batch_norm)
    
    
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
    optimizer = optim.Adam(fine_tune_VAE.parameters(), lr=learning_rate, weight_decay=l2_reg_weight)

    # L1 regularization function
    def l1_penalty(model):
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_reg_weight * l1_loss

    # Training function (no y_batch since we only have x_batch)
    def train_epoch(model, data_loader):
        model.train()
        total_loss = 0
        for x_batch in data_loader:
            x_batch = x_batch[0].to(device)  # Extract the actual data from the tuple
            
            optimizer.zero_grad()
            x_recon, mu, log_var = model(x_batch)
            loss = model.loss(x_batch, x_recon, mu, log_var)
            loss += l1_penalty(model)  # Add L1 regularization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    # Validation function (no y_batch)
    def validate(model, data_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch in data_loader:
                x_batch = x_batch[0].to(device)  # Extract the actual data from the tuple
                x_recon, mu, log_var = model(x_batch)
                loss = model.loss(x_batch, x_recon, mu, log_var)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
    # Step 5: Fine-tune the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fine_tune_VAE.to(device)

    for epoch in range(num_epochs):
        train_loss = train_epoch(fine_tune_VAE, train_loader)
        val_loss = validate(fine_tune_VAE, val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Step 6: Evaluate on the validation set during fine-tuning
    val_loss = validate(fine_tune_VAE, val_loader)
    print(f'Validation Loss: {val_loss:.4f}')

    # Optional: Evaluate on the test set after fine-tuning (for final evaluation)
    test_loss = validate(fine_tune_VAE, test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    
    # Return the fine-tuned model and validation loss for Optuna to minimize
    return fine_tune_VAE, val_loss
