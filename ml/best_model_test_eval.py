
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


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
    
    if num_classes == 2:
        criterion = MaskedBCELoss()  # Use BCELoss for binary classification
    else:
        criterion = MaskedCrossEntropyLoss(num_classes=num_classes)  # Use CrossEntropyLoss for multi-class classification

    # Load the data
    y_data_test_tensor = torch.tensor(y_data_test.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32)
    
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
            else:
                _, predicted = torch.max(outputs.data, 1)
                valid_mask = (labels >= 0) & (labels < num_classes)
                all_labels.extend(labels[valid_mask].cpu().numpy())
            
            total += valid_mask.sum().item()
            correct += ((predicted == labels) & valid_mask).sum().item()


    # Calculate metrics
    accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
    precision = precision_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
    recall = recall_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
    f1 = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
    auc = roc_auc_score(all_labels, all_preds) * 100

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