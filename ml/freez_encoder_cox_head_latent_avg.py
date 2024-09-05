
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
    def __init__(self, encoder, dropout=0.2, task_layer_size=128):
        super(FineTuneCoxModel, self).__init__()
        latent_size = self.get_last_linear_out_features(encoder)
        self.encoder = encoder
        self.freeze_encoder_except_last()
        self.latent_mode = False  # Add a flag to toggle between raw input and latent mode
        
        self.cox_head = nn.Sequential(
            nn.Linear(latent_size, task_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_layer_size, 1)  # Output layer with 1 unit for CoxPH model (log hazard ratio)
        )

    def get_last_linear_out_features(self, module):
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Linear):
                return m.out_features
        raise ValueError("No nn.Linear layer found in the encoder")
    
    def freeze_encoder_except_last(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        latent_size = self.get_last_linear_out_features(self.encoder)
        for m in reversed(list(self.encoder.modules())):
            if isinstance(m, nn.Linear) and m.out_features == latent_size:
                for param in m.parameters():
                    param.requires_grad = True
                break

    def forward(self, x):
        if not self.latent_mode:  # If in raw input mode
            x = self.encoder(x)
            if isinstance(x, tuple):
                x = x[0]
        # If in latent mode, assume x is already a latent representation
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
def fine_tune_cox_model(encoder, X_train, y_duration_train, y_event_train, X_val, y_duration_val, y_event_val, num_epochs=20, batch_size=32, learning_rate=1e-4, dropout=0.2, task_layer_size=128, l1_reg_weight=0.0, l2_reg_weight=0.0, latent_passes=10, seed=None):
    
    # Set seed for reproducibility
    seed = set_seed(seed)

    model = FineTuneCoxModel(encoder, dropout, task_layer_size)
    
    criterion = CoxPHLoss()  # Use CoxPH loss function for survival analysis
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_reg_weight)

    # Convert pandas DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_duration_train_tensor = torch.tensor(y_duration_train.fillna(-1).values, dtype=torch.float32)
    y_event_train_tensor = torch.tensor(y_event_train.fillna(-1).values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_duration_val_tensor = torch.tensor(y_duration_val.fillna(-1).values, dtype=torch.float32)
    y_event_val_tensor = torch.tensor(y_event_val.fillna(-1).values, dtype=torch.float32)

    # Create TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_duration_train_tensor, y_event_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_duration_val_tensor, y_event_val_tensor)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              worker_init_fn=lambda worker_id: set_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # DataFrame to store metrics per epoch
    metrics_per_epoch = pd.DataFrame(columns=['Epoch', 'C-index', 'Validation Loss'])


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, durations, events in train_loader:
            optimizer.zero_grad()

            # Latent averaging with fine-tuning of the last encoder layer
            latent_reps_train = []
            for _ in range(latent_passes):  # Generate multiple latent representations
                latent_rep = model.encoder(inputs)[0]  # Assuming the first output is mu
                latent_reps_train.append(latent_rep)
            latent_rep_train = torch.mean(torch.stack(latent_reps_train), dim=0)

            # Forward pass using averaged latent representations
            outputs = model.cox_head(latent_rep_train)
            loss = criterion(outputs, durations, events)

            # Add L1 regularization
            l1_norm = l1_regularization(model, l1_reg_weight)
            loss += l1_norm

            loss.backward()
            optimizer.step()
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
                    latent_rep = model.encoder(inputs)[0]
                    latent_reps_val.append(latent_rep)
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
    return model, metrics_per_epoch
