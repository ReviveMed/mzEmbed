
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






# class FineTuneModel(nn.Module):
#     def __init__(self, VAE_model, num_classes, dropout=0.2, task_layer_size=128):
#         super(FineTuneModel, self).__init__()
#         self.latent_size = VAE_model.latent_size  # Use the latent size from the VAE model
#         print (self.latent_size)

#         self.encoder = VAE_model.encoder
#         self.freeze_encoder_except_last()
#         self.latent_mode = False  # Toggle between raw input and latent mode

#         if num_classes == 2:
#             self.classifier = nn.Sequential(
#                 nn.Linear(self.latent_size, task_layer_size),  # Use latent size here, not input size
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(task_layer_size, 1),  # Output layer with 1 unit for binary classification
#                 nn.Sigmoid()  # Sigmoid for binary classification
#             )
#         else:
#             self.classifier = nn.Sequential(
#                 nn.Linear(self.latent_size, task_layer_size),  # Use latent size here
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(task_layer_size, num_classes),
#                 nn.Softmax(dim=1)  # Softmax for multi-class classification
#             )

#     def freeze_encoder_except_last(self):
#         """Freeze all layers in the encoder except the last one."""
#         for name, param in self.encoder.named_parameters():
#             param.requires_grad = False

#         for m in reversed(list(self.encoder.modules())):
#             if isinstance(m, nn.Linear) and m.out_features == self.latent_size * 2:  # Considering mu and logvar
#                 for param in m.parameters():
#                     param.requires_grad = True
#                 break

#     def forward(self, x):
#         if not self.latent_mode:  # If in raw input mode
#             encoder_output = self.encoder(x)  # Get latent representation from encoder
            
#             # Slice the first half of the encoder output to get `mu`
#             # Assuming the encoder output shape is [batch_size, 2 * latent_size]
#             mu = encoder_output[:, :self.latent_size]  # Get the first half for `mu`
#             x = mu  # Use `mu` as the input to the classifier
           
#         # Pass the latent representation (`mu`) to the classifier
#         return self.classifier(x)






# def fine_tune_model(VAE_model, X_train, y_data_train, X_val, y_data_val, num_classes, add_post_latent_layers=False, num_post_latent_layers=1, post_latent_layer_size=128, num_epochs=20, batch_size=32, learning_rate=1e-4, dropout=0.2, l1_reg_weight=0.0, l2_reg_weight=0.0, latent_passes=10, seed=None):
    
#     # Set seed for reproducibility
#     if seed is not None:
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)

    
#     # Check if CUDA is available and set the device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # If your model is on the GPU, move it to the same device
    

#     # Initialize the fine-tuning model
#     model = FineTuneModel(VAE_model, num_classes, add_post_latent_layers=False, num_post_latent_layers=1, post_latent_layer_size=128, dropout=0.2)
#     model = model.to(device)
    
#     if num_classes == 2:
#         criterion = MaskedBCELoss()  # Use BCELoss for binary classification
#     else:
#         criterion = MaskedCrossEntropyLoss(num_classes=num_classes)  # Use CrossEntropyLoss for multi-class classification

#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_reg_weight)

#     # Convert pandas DataFrames to PyTorch tensors
#     X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
#     y_data_train_tensor = torch.tensor(y_data_train.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32).to(device)
#     X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
#     y_data_val_tensor = torch.tensor(y_data_val.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32).to(device)

#     # Create TensorDataset
#     train_dataset = TensorDataset(X_train_tensor, y_data_train_tensor)
#     val_dataset = TensorDataset(X_val_tensor, y_data_val_tensor)

#     # Create DataLoader for training and validation
#     train_loader = DataLoader(train_dataset, 
#                               batch_size=batch_size, 
#                               shuffle=True, 
#                               worker_init_fn=lambda worker_id: set_seed(seed))
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # DataFrame to store metrics per epoch
#     metrics_per_epoch = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Validation Loss'])

#     # Training loop
#     for epoch in range(num_epochs):
#         model.train()
#         model.latent_mode = False  # Train using raw input data
        
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
            
#             # Latent averaging with fine-tuning of the last encoder layer
#             latent_reps_train = []
#             for _ in range(latent_passes):  # Generate multiple latent representations
#                 latent_rep = model.encoder(inputs)
#                 mu = latent_rep[:, :model.latent_size]  
#                 latent_reps_train.append(mu)
#             latent_rep_train = torch.mean(torch.stack(latent_reps_train), dim=0)

#             # Forward pass using averaged latent representations
#             outputs = model.classifier(latent_rep_train)
#             loss = criterion(outputs, labels)

#             # Add L1 regularization
#             l1_norm = l1_regularization(model, l1_reg_weight)
#             loss = loss+ l1_norm

#             loss.backward()
#             optimizer.step()
#             running_loss = running_loss+ loss.item()
        
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

#         # Validation
#         model.eval()
#         model.latent_mode = True  # Use precomputed latent representations for validation
        
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         all_labels = []
#         all_preds = []
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 # Latent averaging for validation
#                 latent_reps_val = []
#                 for _ in range(latent_passes):  # Multiple passes through the encoder
#                     latent_rep = model.encoder(inputs)
#                     mu = latent_rep[:, :model.latent_size]  # Slice out mu (first half of the output)
#                     latent_reps_val.append(mu)
#                 latent_rep_val = torch.mean(torch.stack(latent_reps_val), dim=0)

#                 # Forward pass using averaged latent representations
#                 outputs = model(latent_rep_val)
#                 loss = criterion(outputs, labels)
                
#                 # Add L1 regularization to validation loss as well
#                 val_loss += (loss + l1_regularization(model, l1_reg_weight)).item()
                
#                 if num_classes == 2:
#                     predicted_probs = outputs.squeeze()  # Binary classification probabilities
#                     predicted = (predicted_probs >= 0.5).float()  # Binary classification threshold
#                     valid_mask = labels >= 0  # Mask for valid labels
#                     all_preds.extend(predicted_probs[valid_mask].cpu().numpy())  # Use only valid predictions for AUC
#                     all_labels.extend(labels[valid_mask].cpu().numpy())  # Use only valid labels for AUC
#                 else:
#                     _, predicted = torch.max(outputs.data, 1)
#                     valid_mask = (labels >= 0) & (labels < num_classes)
#                     all_labels.extend(labels[valid_mask].cpu().numpy())
                
#                 total += valid_mask.sum().item()
#                 correct += ((predicted == labels) & valid_mask).sum().item()

#         # Calculate metrics
#         accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
#         precision = precision_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
#         recall = recall_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
#         f1 = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)) * 100
#         auc = roc_auc_score(all_labels, all_preds) * 100

#         # Create a DataFrame with the metrics for the current epoch
#         metrics_df = pd.DataFrame({
#             'Epoch': [epoch + 1],
#             'Accuracy': [accuracy],
#             'Precision': [precision],
#             'Recall': [recall],
#             'F1 Score': [f1],
#             'AUC': [auc],
#             'Validation Loss': [val_loss / len(val_loader)]
#         })

#         # Concatenate with the existing DataFrame
#         metrics_per_epoch = pd.concat([metrics_per_epoch, metrics_df], ignore_index=True)

#         print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%, Precision: {precision}%, Recall: {recall}%, F1 Score: {f1}%, AUC: {auc}%')

#     print('Fine-tuning completed.')
#     return model, metrics_per_epoch









class FineTuneModel(nn.Module):
    def __init__(self, VAE_model, num_classes, num_layers_to_retrain=1, add_post_latent_layers=False, num_post_latent_layers=1, post_latent_layer_size=128, dropout=0.2):
        super(FineTuneModel, self).__init__()
        self.latent_size = VAE_model.latent_size  # Keep the original latent size from the VAE
        self.encoder = VAE_model.encoder
        self.freeze_encoder_except_last(num_layers_to_retrain)
        self.latent_mode = False  # Toggle between raw input and latent mode

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

        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(self.post_latent_layer_size, 1),  # Binary classification
                nn.Sigmoid()  # Sigmoid for binary classification
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.post_latent_layer_size, num_classes),
                nn.Softmax(dim=1)  # Softmax for multi-class classification
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

        # print(f"\nTotal hidden layers (excluding latent): {total_layers}")
        # print(f"Number of layers to retrain: {num_layers_to_retrain}")
        
        # Freeze all parameters in the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # # Print the state of each layer before unfreezing
        # for name, param in self.encoder.named_parameters():
        #     print(f"{name}: requires_grad = {param.requires_grad}")


        # Unfreeze the specified number of layers from the end of the hidden layers
        if num_layers_to_retrain > 0:
            for layer in hidden_layers[-num_layers_to_retrain:]:
                for param in layer.parameters():
                        param.requires_grad = True
        
        # Print the state of each layer after unfreezing
        # # print("\nAfter unfreezing:")
        # for name, param in self.encoder.named_parameters():
        #     print(f"{name}: requires_grad = {param.requires_grad}")


    def forward(self, x):
        if not self.latent_mode:  # If in raw input mode
            encoder_output = self.encoder(x)  # Get latent representation from encoder
            
            # Slice the first half of the encoder output to get `mu`
            mu = encoder_output[:, :self.latent_size]  # Get the first half for `mu`
            x = mu  # Use `mu` as the input to the classifier

        if self.post_latent_layers:
            x = self.post_latent_layers(x)

        # Pass the latent representation (or the output of post-latent layers) to the classifier
        return self.classifier(x)
    



def fine_tune_model(VAE_model, X_train, y_data_train, X_val, y_data_val, num_classes, num_layers_to_retrain=1, add_post_latent_layers=False, num_post_latent_layers=1, post_latent_layer_size=128, num_epochs=20, batch_size=32, learning_rate=1e-4, dropout=0.2, l1_reg_weight=0.0, l2_reg_weight=0.0, latent_passes=10, seed=None):
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    # Initialize the fine-tuning model
    model = FineTuneModel(VAE_model, num_classes, num_layers_to_retrain=num_layers_to_retrain, add_post_latent_layers=add_post_latent_layers, num_post_latent_layers=num_post_latent_layers, post_latent_layer_size=post_latent_layer_size, dropout=dropout)
    model = model.to(device)
    
    if num_classes == 2:
        criterion = MaskedBCELoss()  # Use BCELoss for binary classification
    else:
        criterion = MaskedCrossEntropyLoss(num_classes=num_classes)  # Use CrossEntropyLoss for multi-class classification

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=l2_reg_weight)

    # Convert pandas DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_data_train_tensor = torch.tensor(y_data_train.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_data_val_tensor = torch.tensor(y_data_val.fillna(-1).values, dtype=torch.long if num_classes > 2 else torch.float32).to(device)

    #print (X_train_tensor.shape, y_data_train_tensor.shape, X_val_tensor.shape, y_data_val_tensor.shape)

    # Create TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_data_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_data_val_tensor)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda worker_id: set_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # DataFrame to store metrics per epoch
    metrics_per_epoch = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Validation Loss'])

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        model.latent_mode = False  # Train using raw input data

        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # Latent averaging with fine-tuning of the last encoder layer
            latent_reps_train = []

            
            for _ in range(latent_passes):  # Generate multiple latent representations
                latent_rep = model.encoder(inputs.to(device))
                mu = latent_rep[:, :model.latent_size]  
                latent_reps_train.append(mu)
            latent_rep_train = torch.mean(torch.stack(latent_reps_train), dim=0)
            
            #print (latent_rep_train.shape)

            # Set the model to latent mode to ensure it processes latent representations correctly
            model.latent_mode = True

            # Forward pass using averaged latent representations
            outputs = model.forward(latent_rep_train)
            loss = criterion(outputs, labels.to(device))

            # Add L1 regularization
            l1_norm = l1_regularization(model, l1_reg_weight)
            loss = loss + l1_norm

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')


    # Validation
    print  ('Validation started')
    model.eval()
    #model.latent_mode = True  # Use precomputed latent representations for validation
    
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    # Validation loop (corrected)
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Latent averaging for validation
            latent_reps_val = []
            for _ in range(latent_passes):  # Multiple passes through the encoder
                latent_rep = model.encoder(inputs.to(device))  # Ensure inputs are on the same device
                mu = latent_rep[:, :model.latent_size]  # Slice out mu (first half of the output)
                latent_reps_val.append(mu)
            latent_rep_val = torch.mean(torch.stack(latent_reps_val), dim=0)

            #print (latent_rep_val.shape)
            model.latent_mode = True  # Use precomputed latent representations for validation
            # Forward pass using averaged latent representations
            outputs = model(latent_rep_val)

            # Switch back to not latent mode
            model.latent_mode = False

            # Ensure predictions and labels have the same batch size
            #print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")  # Debugging step
        
            loss = criterion(outputs, labels.to(device))  # Ensure labels are on the same device

            # Add L1 regularization to validation loss as well
            val_loss += (loss + l1_regularization(model, l1_reg_weight)).item()
            
            if num_classes == 2:
                predicted_probs = outputs.squeeze()  # Binary classification probabilities
                predicted = (predicted_probs >= 0.5).float()  # Binary classification threshold
                valid_mask = labels >= 0  # Mask for valid labels
                all_preds.extend(predicted_probs[valid_mask.to(device)].cpu().numpy())  # Use only valid predictions for AUC
                all_labels.extend(labels[valid_mask.to(device)].cpu().numpy())  # Use only valid labels for AUC
                # Ensure labels and predicted are on the same device
                correct += ((predicted == labels.to(device)) & valid_mask.to(device)).sum().item()
            else:
                outputs_np = outputs.cpu().numpy()  # Convert outputs to numpy
                valid_mask = (labels >= 0) & (labels < num_classes)
                
                # Append predicted probabilities for all samples in the batch
                all_preds.extend(outputs_np[valid_mask.cpu().numpy()])  # Append predictions for all batches
                all_labels.extend(labels[valid_mask.to(device)].cpu().numpy())  # Use only valid labels
                #_, predicted = torch.max(outputs.data, 1)  # For multi-class classification
                # valid_mask = (labels >= 0) & (labels < num_classes)
                # all_preds.extend(predicted[valid_mask.to(device)].cpu().numpy())  # Use predictions
                # all_labels.extend(labels[valid_mask.to(device)].cpu().numpy())  # Use only valid labels

            
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