
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



# Define the fine-tuning model
class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0.2, task_layer_size=128):
        super(FineTuneModel, self).__init__()
        latent_size = self.get_last_linear_out_features(encoder)
        self.encoder = encoder
        self.freeze_encoder_except_last()
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(latent_size, task_layer_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(task_layer_size, 1),  # Output layer with 1 unit for binary classification
                nn.Sigmoid()  # Sigmoid for binary classification
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(latent_size, task_layer_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(task_layer_size, num_classes),
                nn.Softmax(dim=1)  # Softmax for multi-class classification
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
        x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        return self.classifier(x)





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




# Define the fine-tuning model function with L1 and L2 regularization
def fine_tune_model(encoder, X_train, y_data_train, X_val, y_data_val, num_classes, num_epochs=20, batch_size=32, learning_rate=1e-4, dropout=0.2, task_layer_size=128, l1_reg_weight=0.0, l2_reg_weight=0.0, seed=None):
    
    # Set seed for reproducibility
    seed = set_seed(seed)

    model = FineTuneModel(encoder, num_classes, dropout, task_layer_size)
    
    if num_classes == 2:
        criterion = MaskedBCELoss()  # Use BCELoss for binary classification
    else:
        criterion = MaskedCrossEntropyLoss(num_classes=num_classes)  # Use CrossEntropyLoss for multi-class classification

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_reg_weight)

    # Convert pandas DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_data_train_tensor = torch.tensor(y_data_train.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_data_val_tensor = torch.tensor(y_data_val.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32)

    # Create TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_data_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_data_val_tensor)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              worker_init_fn=lambda worker_id: set_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # DataFrame to store metrics per epoch
    metrics_per_epoch = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Validation Loss'])

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add L1 regularization
            l1_norm = l1_regularization(model, l1_reg_weight)
            loss += l1_norm

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Add L1 regularization to validation loss as well
                val_loss += (loss + l1_regularization(model, l1_reg_weight)).item()
                
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
            'Epoch': [epoch + 1],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1],
            'AUC': [auc],
            'Validation Loss': [val_loss / len(val_loader)]
        })

        # Concatenate with the existing DataFrame
        metrics_per_epoch = pd.concat([metrics_per_epoch, metrics_df], ignore_index=True)

        print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%, Precision: {precision}%, Recall: {recall}%, F1 Score: {f1}%, AUC: {auc}%')

    print('Fine-tuning completed.')
    return model, metrics_per_epoch
