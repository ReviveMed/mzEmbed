
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from lifelines.utils import concordance_index


# Make sure to disable non-deterministic operations if exact reproducibility is crucial
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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





def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)
    



# Define the fine-tuning model function with L1 and L2 regularization
def evaluate_model(model, latent_rep, y_duration_test, y_event_test, seed, batch_size=32):
    
    # Set seed for reproducibility
    seed = set_seed(seed)
    


    
    criterion = CoxPHLoss()  # Use CoxPH loss function for survival analysis
 
    # Convert pandas DataFrames to PyTorch tensors
    y_duration_test_tensor = torch.tensor(y_duration_test.fillna(-1).values, dtype=torch.float32)
    y_event_test_tensor = torch.tensor(y_event_test.fillna(-1).values, dtype=torch.float32)
   
    # Replace X_test_tensor with latent_rep for downstream evaluation
    # Create TensorDataset using the latent representation
    test_dataset = TensorDataset(latent_rep, y_duration_test_tensor, y_event_test_tensor)
 
    # Create DataLoader for training and validation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # DataFrame to store metrics per epoch
    metrics_test = pd.DataFrame(columns=['C-index'])

    
    # Validation
    model.eval()
    model.latent_mode = True  # Use precomputed latent representations for validation
    
    all_durations = []
    all_events = []
    all_risks = []

    with torch.no_grad():
        for inputs, durations, events in test_loader:
            
            # Forward pass using averaged latent representations
            outputs = model(inputs)
                        
            valid_mask = (durations != -1) & (events != -1)
            all_durations.extend(durations[valid_mask].cpu().numpy())
            all_events.extend(events[valid_mask].cpu().numpy())
            all_risks.extend(outputs[valid_mask].squeeze().cpu().numpy())

        # Calculate C-index
        c_index = concordance_index(all_durations, -np.array(all_risks), all_events)

        # Create a DataFrame with the metrics for the current epoch
        metrics_test = pd.DataFrame({
            'C-index': [c_index]
        })

        

        print(f'Test C-index: {c_index}')

    print('Fine-tuning completed.')
    return metrics_test
