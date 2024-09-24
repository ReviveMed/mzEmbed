
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


# Make sure to disable non-deterministic operations if exact reproducibility is crucial
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# Define the custom masked cross-entropy loss
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        valid_mask = (targets >= 0) & (targets < self.num_classes)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        loss = self.criterion(inputs, targets)
        return loss.mean()



# Define the custom masked BCE loss for binary classification
class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='none')  # No reduction, handle it manually

    def forward(self, inputs, targets):
        targets = targets.float().view(-1, 1)  # Ensure targets have shape [batch_size, 1]
        valid_mask = targets >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        loss = self.criterion(inputs, targets)
        return loss.mean()






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
def evaluate_model(model, latent_rep, y_data_test, num_classes, seed, batch_size=32):
    
    # Set seed for reproducibility
    seed = set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    if num_classes == 2:
        criterion = MaskedBCELoss()  # Use BCELoss for binary classification
    else:
        criterion = MaskedCrossEntropyLoss(num_classes=num_classes)  # Use CrossEntropyLoss for multi-class classification

    # Load the data
    y_data_test_tensor = torch.tensor(y_data_test.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32).to(device)
    
    # Replace X_test_tensor with latent_rep for downstream evaluation
    # Create TensorDataset using the latent representation
    test_dataset = TensorDataset(latent_rep, y_data_test_tensor)

    # Create DataLoader for evaluation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)


    # DataFrame to store metrics per epoch
    metrics = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Validation Loss'])
        
    # Validation

    model.eval() 
    model.latent_mode=True
 
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            if num_classes == 2:
                predicted_probs = outputs.squeeze()  # Binary classification probabilities
                predicted = (predicted_probs >= 0.5).float()  # Binary classification threshold
                valid_mask = labels >= 0  # Mask for valid labels
                all_preds.extend(predicted_probs[valid_mask].cpu().numpy())  # Use only valid predictions for AUC
                all_labels.extend(labels[valid_mask].cpu().numpy())  # Use only valid labels for AUC
                correct += ((predicted == labels) & valid_mask).sum().item()
            else:
                outputs_np = outputs.cpu().numpy()  # Convert outputs to numpy
                valid_mask = (labels >= 0) & (labels < num_classes)
                
                # Append predicted probabilities for all samples in the batch
                all_preds.extend(outputs_np[valid_mask.cpu().numpy()])  # Append predictions for all batches
                all_labels.extend(labels[valid_mask.to(device)].cpu().numpy())  # Use only valid labels
                # _, predicted = torch.max(outputs.data, 1)
                # valid_mask = (labels >= 0) & (labels < num_classes)
                # all_labels.extend(labels[valid_mask].cpu().numpy())
            
            total += valid_mask.sum().item()
            


    # Calculate metrics
        if num_classes == 2:
            accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
            precision = precision_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
            recall = recall_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
            f1 = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
            auc = roc_auc_score(all_labels, all_preds) * 100
        else:
            # Convert predicted probabilities to predicted class labels for multi-class classification
            all_preds_np = np.vstack(all_preds)  # Stack the list of predictions
            all_preds_class = np.argmax(all_preds_np, axis=1)  # Get the class with the highest probability
            
            accuracy = accuracy_score(all_labels, all_preds_class) * 100
            precision = precision_score(all_labels, all_preds_class, average='weighted') * 100
            recall = recall_score(all_labels, all_preds_class, average='weighted') * 100
            f1 = f1_score(all_labels, all_preds_class, average='weighted') * 100
        
            # Calculate AUC for multi-class classification
            auc = roc_auc_score(all_labels, all_preds_np, multi_class='ovr') * 100  # AUC for multi-class
    
    
    # Create a DataFrame with the metrics for the current epoch
    metrics_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'AUC': [auc],
        'Test Loss': [test_loss / len(test_loader)]
    })

    
    print(f'Validation Loss: {test_loss/len(test_loader)}, Accuracy: {accuracy}%, Precision: {precision}%, Recall: {recall}%, F1 Score: {f1}%, AUC: {auc}%')

    print('Fine-tuning completed.')
    return metrics_df





def evaluate_model_main(best_model, X_data_train, X_data_val, X_data_test, y_data_train, y_data_val, y_data_test, task, num_classes, seed):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### retreiving the latent space of the best models
    # training data
    X_train_tensor = torch.tensor(X_data_train.values, dtype=torch.float32).to(device)

    latent_reps_train = []
    for _ in range(10):  # Run multiple passes
        latent_rep = best_model.encoder(X_train_tensor) 
        mu = latent_rep[:, :best_model.latent_size]
        latent_reps_train.append(mu)

    latent_rep_train = torch.mean(torch.stack(latent_reps_train), dim=0)


    ### Validation data
    X_val_tensor = torch.tensor(X_data_val.values, dtype=torch.float32).to(device)

    latent_reps_val = []
    for _ in range(10):  # Run multiple passes
        latent_rep = best_model.encoder(X_val_tensor)
        mu = latent_rep[:, :best_model.latent_size] 
        latent_reps_val.append(mu)

    latent_rep_val = torch.mean(torch.stack(latent_reps_val), dim=0)


    ### Test data
    X_test_tensor = torch.tensor(X_data_test.values, dtype=torch.float32).to(device)

    latent_reps_test = []
    for _ in range(10):  # Run multiple passes
        latent_rep = best_model.encoder(X_test_tensor)
        mu = latent_rep[:, :best_model.latent_size]  
        latent_reps_test.append(mu)  # Assuming first output is mu
        
    latent_rep_test = torch.mean(torch.stack(latent_reps_test), dim=0)



    # # Evaluate on the test set
    best_model.eval()  # Set the model to evaluation mode
    best_model.latent_mode=True

    train_metrics = evaluate_model(
        best_model, 
        latent_rep_train, 
        y_data_train[task], 
        num_classes,
        seed,
        32
    )

    val_metrics = evaluate_model(
        best_model, 
        latent_rep_val, 
        y_data_val[task], 
        num_classes,
        seed,
        32
    )

    test_metrics = evaluate_model(
        best_model, 
        latent_rep_test, 
        y_data_test[task], 
        num_classes,
        seed,
        32
    )

    # Add an identifier column to each dataframe
    train_metrics['Dataset'] = 'Train'
    val_metrics['Dataset'] = 'Validation'
    test_metrics['Dataset'] = 'Test'

    # Concatenate the dataframes into a single dataframe
    combined_result_metrics = pd.concat([train_metrics, val_metrics, test_metrics], ignore_index=True)

    # Reorder the columns to make 'Dataset' the first column
    combined_result_metrics = combined_result_metrics[['Dataset'] + [col for col in combined_result_metrics.columns if col != 'Dataset']]



    return combined_result_metrics, latent_rep_train, latent_rep_val, latent_rep_test




def best_model_latent_plot(latent_rep_train, latent_rep_val, latent_rep_test, y_data_train, y_data_val, y_data_test, task):

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



def best_model_latent_plot_combined(latent_rep_train, latent_rep_val, latent_rep_test, y_data_train, y_data_val, y_data_test, task):


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
