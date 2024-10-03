
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from lifelines.utils import concordance_index

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


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
    
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    
    criterion = CoxPHLoss()  # Use CoxPH loss function for survival analysis
 
    # Convert pandas DataFrames to PyTorch tensors
    y_duration_test_tensor = torch.tensor(y_duration_test.fillna(-1).values, dtype=torch.float32).to(device)
    y_event_test_tensor = torch.tensor(y_event_test.fillna(-1).values, dtype=torch.float32).to(device)
   
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
    return metrics_test, all_durations, all_risks, all_events



def best_finetune_model_test_eval_cox(best_model, X_data_train, y_data_train, X_data_val, y_data_val, X_data_test, y_data_test, task, task_event, seed):

    ### Training data
    # Convert pandas DataFrames to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_data_train.values, dtype=torch.float32).to(device)

    latent_reps_train = []
    for _ in range(10):  # Run multiple passes
        latent_rep = best_model.encoder(X_train_tensor)
        mu=latent_rep[:, :best_model.latent_size]
        latent_reps_train.append(mu)

    latent_rep_train = torch.mean(torch.stack(latent_reps_train), dim=0)


    ### Validation data
    # Convert pandas DataFrames to PyTorch tensors
    X_val_tensor = torch.tensor(X_data_val.values, dtype=torch.float32).to(device)

    latent_reps_val = []
    for _ in range(10):  # Run multiple passes
        latent_rep = best_model.encoder(X_val_tensor)  
        mu=latent_rep[:, :best_model.latent_size]
        latent_reps_val.append(mu)

    latent_rep_val = torch.mean(torch.stack(latent_reps_val), dim=0)


    ### Test data
    # Convert pandas DataFrames to PyTorch tensors
    X_test_tensor = torch.tensor(X_data_test.values, dtype=torch.float32).to(device)

    latent_reps_test = []
    for _ in range(10):  # Run multiple passes
        latent_rep = best_model.encoder(X_test_tensor) 
        mu=latent_rep[:, :best_model.latent_size]
        latent_reps_test.append(mu)

    latent_rep_test = torch.mean(torch.stack(latent_reps_test), dim=0)


    metrics_train, all_durations, all_risks, all_events=evaluate_model(best_model, latent_rep_train, y_data_train[task], y_data_train[task_event], seed, batch_size=32)
  

    metrics_val, all_durations, all_risks, all_events=evaluate_model(best_model, latent_rep_val, y_data_val[task], y_data_val[task_event], seed, batch_size=32)


    metrics_test, all_durations, all_risks, all_events=evaluate_model(best_model, latent_rep_test, y_data_test[task], y_data_test[task_event], seed, batch_size=32)

    # Add an identifier column to each dataframe
    metrics_train['Dataset'] = 'Train'
    metrics_val['Dataset'] = 'Validation'
    metrics_test['Dataset'] = 'Test'

    # Concatenate the dataframes into a single dataframe
    combined_result_metrics = pd.concat([metrics_train, metrics_val, metrics_test], ignore_index=True)

    # Reorder the columns to make 'Dataset' the first column
    combined_result_metrics = combined_result_metrics[['Dataset'] + [col for col in combined_result_metrics.columns if col != 'Dataset']]

    
    return combined_result_metrics, latent_rep_train, latent_rep_val, latent_rep_test 




def best_model_latent_plot(latent_rep_train, latent_rep_val, latent_rep_test, y_data_train, y_data_val, y_data_test, task, task_event):

    # Assuming y_data_train, y_data_val, and y_data_test are pandas Series or arrays
    y_train = y_data_train[task].values
    y_val = y_data_val[task].values
    y_test = y_data_test[task].values

    # Convert latent representations to numpy arrays for visualization
    latent_rep_train_np = latent_rep_train.detach().cpu().numpy()
    latent_rep_val_np = latent_rep_val.detach().cpu().numpy()
    latent_rep_test_np = latent_rep_test.detach().cpu().numpy()

    # PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2)
    latent_rep_train_pca = pca.fit_transform(latent_rep_train_np)
    latent_rep_val_pca = pca.transform(latent_rep_val_np)
    latent_rep_test_pca = pca.transform(latent_rep_test_np)

    # UMAP for dimensionality reduction to 2D
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    latent_rep_train_umap = umap_reducer.fit_transform(latent_rep_train_np)
    latent_rep_val_umap = umap_reducer.transform(latent_rep_val_np)
    latent_rep_test_umap = umap_reducer.transform(latent_rep_test_np)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot PCA results
    axes[0, 0].scatter(latent_rep_train_pca[:, 0], latent_rep_train_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
    axes[0, 0].set_title('PCA: Training Data')
    axes[0, 0].set_xlabel('Component 1')
    axes[0, 0].set_ylabel('Component 2')
    axes[0, 0].colorbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])

    axes[0, 1].scatter(latent_rep_val_pca[:, 0], latent_rep_val_pca[:, 1], c=y_val, cmap='coolwarm', alpha=0.7)
    axes[0, 1].set_title('PCA: Validation Data')
    axes[0, 1].set_xlabel('Component 1')
    axes[0, 1].set_ylabel('Component 2')
    axes[0, 1].colorbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])

    axes[0, 2].scatter(latent_rep_test_pca[:, 0], latent_rep_test_pca[:, 1], c=y_test, cmap='coolwarm', alpha=0.7)
    axes[0, 2].set_title('PCA: Test Data')
    axes[0, 2].set_xlabel('Component 1')
    axes[0, 2].set_ylabel('Component 2')
    axes[0, 2].colorbar = plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2])

    # Plot UMAP results
    axes[1, 0].scatter(latent_rep_train_umap[:, 0], latent_rep_train_umap[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
    axes[1, 0].set_title('UMAP: Training Data')
    axes[1, 0].set_xlabel('Component 1')
    axes[1, 0].set_ylabel('Component 2')
    axes[1, 0].colorbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])

    axes[1, 1].scatter(latent_rep_val_umap[:, 0], latent_rep_val_umap[:, 1], c=y_val, cmap='coolwarm', alpha=0.7)
    axes[1, 1].set_title('UMAP: Validation Data')
    axes[1, 1].set_xlabel('Component 1')
    axes[1, 1].set_ylabel('Component 2')
    axes[1, 1].colorbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])

    axes[1, 2].scatter(latent_rep_test_umap[:, 0], latent_rep_test_umap[:, 1], c=y_test, cmap='coolwarm', alpha=0.7)
    axes[1, 2].set_title('UMAP: Test Data')
    axes[1, 2].set_xlabel('Component 1')
    axes[1, 2].set_ylabel('Component 2')
    axes[1, 2].colorbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])

    # Adjust layout
    plt.tight_layout()
    
    return fig, axes



def best_model_latent_plot_combined(latent_rep_train, latent_rep_val, latent_rep_test, y_data_train, y_data_val, y_data_test, task, task_event):


    # Assuming y_data_train, y_data_val, and y_data_test are pandas Series or arrays
    y_train = y_data_train[task].values
    y_val = y_data_val[task].values
    y_test = y_data_test[task].values

    # Convert latent representations to numpy arrays for visualization
    latent_rep_train_np = latent_rep_train.detach().cpu().numpy()
    latent_rep_val_np = latent_rep_val.detach().cpu().numpy()
    latent_rep_test_np = latent_rep_test.detach().cpu().numpy()

    # Concatenate latent representations and labels
    latent_rep_all = np.concatenate([latent_rep_train_np, latent_rep_val_np, latent_rep_test_np], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)

    # PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2)
    latent_rep_all_pca = pca.fit_transform(latent_rep_all)

    # UMAP for dimensionality reduction to 2D
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    latent_rep_all_umap = umap_reducer.fit_transform(latent_rep_all)

    # Create a combined figure for PCA and UMAP
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot combined PCA results
    scatter_pca = axes[0].scatter(latent_rep_all_pca[:, 0], latent_rep_all_pca[:, 1], c=y_all, cmap='coolwarm', alpha=0.7)
    axes[0].set_title('PCA: Combined Data')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].colorbar = plt.colorbar(scatter_pca, ax=axes[0])

    # Plot combined UMAP results
    scatter_umap = axes[1].scatter(latent_rep_all_umap[:, 0], latent_rep_all_umap[:, 1], c=y_all, cmap='coolwarm', alpha=0.7)
    axes[1].set_title('UMAP: Combined Data')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].colorbar = plt.colorbar(scatter_umap, ax=axes[1])

    # Adjust layout
    plt.tight_layout()
    return fig, axes
