
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from lifelines.utils import concordance_index


# Define the custom CoxPH loss function
class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk, duration, event):
        """Computes the negative partial log-likelihood.
        
        Args:
            risk: Predicted risk scores (log hazard ratios).
            duration: Observed durations.
            event: Event indicators (1 if event occurred, 0 if censored).
            
        Returns:
            Loss: The negative partial log-likelihood.
        """
        risk = risk.squeeze()  # Ensure risk has shape [batch_size]

        # Create a mask to exclude missing values
        valid_mask = (duration != -1) & (event != -1)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Apply the mask
        duration = duration[valid_mask]
        event = event[valid_mask]
        risk = risk[valid_mask]

        # Sort by duration in descending order
        idx = torch.argsort(duration, descending=True)
        duration = duration[idx]
        event = event[idx]
        risk = risk[idx]

        # Compute the cumulative sum of the exponential of the predicted risk scores
        exp_risk_sum = torch.cumsum(torch.exp(risk), dim=0)

        # Compute the log-likelihood for events
        log_likelihood = risk - torch.log(exp_risk_sum)

        # Only consider the events (not censored cases) in the loss
        log_likelihood = log_likelihood * event

        # Return the negative log-likelihood as the loss
        return -torch.mean(log_likelihood)





# Define the fine-tuning model with a CoxPH head
class FineTuneCoxModel(nn.Module):
    def __init__(self, VAE_model, num_layers_to_retrain=1, add_post_latent_layers=False, num_post_latent_layers=1, post_latent_layer_size=128, dropout=0.2):
        super(FineTuneCoxModel, self).__init__()
        self.latent_size=VAE_model.latent_size
        #latent_size = self.get_last_linear_out_features(encoder)
        self.encoder = VAE_model.encoder
        self.freeze_encoder_except_last(num_layers_to_retrain)
        self.latent_mode = False  # Add a flag to toggle between raw input and latent mode
        

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

        # CoxPH head
        self.cox_head = nn.Sequential(
            nn.Linear(self.post_latent_layer_size, 1)  # Output layer with 1 unit for CoxPH model (log hazard ratio)
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

        # Freeze all parameters in the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the specified number of layers from the end of the hidden layers
        if num_layers_to_retrain > 0:
            for layer in hidden_layers[-num_layers_to_retrain:]:
                for param in layer.parameters():
                        param.requires_grad = True

    
    def forward(self, x):
        if not self.latent_mode:  # If in raw input mode
            encoder_output = self.encoder(x)  # Get latent representation from encoder
            # Slice the first half of the encoder output to get `mu`
            # Assuming the encoder output shape is [batch_size, 2 * latent_size]
            mu = encoder_output[:, :self.latent_size]  # Get the first half for `mu`
            x = mu  # Use `mu` as the input to the classifier
        
        if self.post_latent_layers:
            x = self.post_latent_layers(x)
        
        # Pass the latent representation (`mu`) to the classifier
        return self.cox_head(x)







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




# Define the fine-tuning model function with L1 and L2 regularization for survival analysis
def fine_tune_cox_model(VAE_model, X_train, y_data_train, y_event_train, X_val, y_data_val, y_event_val, num_layers_to_retrain=1, add_post_latent_layers=False, num_post_latent_layers=1, post_latent_layer_size=128, num_epochs=20, batch_size=32, learning_rate=1e-4, dropout=0.2, l1_reg_weight=0.0, l2_reg_weight=0.0, latent_passes=10, seed=None, patience=5):
    
    # Set seed for reproducibility
    seed = set_seed(seed)

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #initialize the model
    model = FineTuneCoxModel(VAE_model, num_layers_to_retrain=num_layers_to_retrain, add_post_latent_layers=add_post_latent_layers, num_post_latent_layers=num_post_latent_layers, post_latent_layer_size=post_latent_layer_size, dropout=dropout)
    model=model.to(device)
    
    criterion = CoxPHLoss()  # Use CoxPH loss function for survival analysis
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_reg_weight)

    # Convert pandas DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_duration_train_tensor = torch.tensor(y_data_train.fillna(-1).values, dtype=torch.float32).to(device)
    y_event_train_tensor = torch.tensor(y_event_train.fillna(-1).values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_duration_val_tensor = torch.tensor(y_data_val.fillna(-1).values, dtype=torch.float32).to(device)
    y_event_val_tensor = torch.tensor(y_event_val.fillna(-1).values, dtype=torch.float32).to(device)

    # Create TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_duration_train_tensor, y_event_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_duration_val_tensor, y_event_val_tensor)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              worker_init_fn=lambda worker_id: set_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Early Stopping Setup
    best_c_index = -float('inf')  # Best validation AUC
    best_model = None  # Store the state of the best model
    patience_counter = 0  # Counter to track patience

    # DataFrame to store metrics per epoch
    metrics_per_epoch = pd.DataFrame(columns=['Epoch', 'C-index', 'Validation Loss'])


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        model.latent_mode = False  # Train using raw input data

        running_loss = 0.0
        for inputs, durations, events in train_loader:
            # Check for NaNs in inputs
            if torch.isnan(inputs).any():
                print("NaN found in inputs")
            if torch.isnan(durations).any():
                print("NaN found in durations")
            if torch.isnan(events).any():
                print("NaN found in events")

            optimizer.zero_grad() # Reset gradients before backpropagation

            # Latent averaging with fine-tuning of the last encoder layer
            latent_reps_train = []
            for _ in range(latent_passes):  # Generate multiple latent representations
                latent_rep = model.encoder(inputs) 
                mu=latent_rep[:, :model.latent_size]
                latent_reps_train.append(mu)
            latent_rep_train = torch.mean(torch.stack(latent_reps_train), dim=0)

            # Set the model to latent mode to ensure it processes latent representations correctly
            model.latent_mode = True
            # Forward pass using averaged latent representations
            outputs = model(latent_rep_train)

            loss = criterion(outputs, durations, events)
            #print(f"Outputs: {outputs}")

            # Add L1 regularization
            l1_norm = l1_regularization(model, l1_reg_weight)
            loss += l1_norm

            loss.backward() # Backward pass (compute gradients)

            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            optimizer.step()  # Update weights based on gradients
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # Validation
        model.eval()
        model.latent_mode = True  # Use precomputed latent representations for validation
        val_loss = 0.0
        all_durations = []
        all_events = []
        all_risks = []
        with torch.no_grad():
            for inputs, durations, events in val_loader:
                
                # Latent averaging for validation
                latent_reps_val = []
                for _ in range(latent_passes):  # Multiple passes through the encoder
                    latent_rep = model.encoder(inputs)
                    mu=latent_rep[:, :model.latent_size]
                    latent_reps_val.append(mu)
                latent_rep_val = torch.mean(torch.stack(latent_reps_val), dim=0)

                # Forward pass using averaged latent representations
                outputs = model(latent_rep_val)
                loss = criterion(outputs, durations, events)
                
                # Add L1 regularization to validation loss as well
                val_loss += (loss + l1_regularization(model, l1_reg_weight)).item()
                
                valid_mask = (durations != -1) & (events != -1)
                all_durations.extend(durations[valid_mask].cpu().numpy())
                all_events.extend(events[valid_mask].cpu().numpy())
                all_risks.extend(outputs[valid_mask].squeeze().cpu().numpy())

        # Calculate C-index
        c_index = concordance_index(all_durations, -np.array(all_risks), all_events)

        # Check early stopping condition
        # Early stopping logic (if patience > 0)
        if patience > 0:
            if c_index > best_c_index:
                best_c_index = c_index
                best_model = model # Save the best model state
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            # If no improvement for `patience` epochs, stop training
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

        # Create a DataFrame with the metrics for the current epoch
        metrics_df = pd.DataFrame({
            'Epoch': [epoch + 1],
            'C-index': [c_index],
            'Validation Loss': [val_loss / len(val_loader)]
        })

        # Concatenate with the existing DataFrame
        metrics_per_epoch = pd.concat([metrics_per_epoch, metrics_df], ignore_index=True)

        print(f'Validation Loss: {val_loss/len(val_loader)}, C-index: {c_index}')

    print('Fine-tuning completed.')
    return best_model, metrics_per_epoch
