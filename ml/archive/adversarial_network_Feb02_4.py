# %%
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
#%% Load the data

test_adversarial_dir = '/Users/jonaheaton/rcc4_mzlearn4-5/test_adversarial'
output_dir = os.path.join(test_adversarial_dir, 'adversarial_network_Feb02_3')
os.makedirs(output_dir, exist_ok=True)
batch_size = 64
num_epochs = 200


# create data loaders
class CustomDataset(Dataset):
    def __init__(self, X, y_benefit, y_gender):
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y_benefit = torch.tensor(y_benefit.to_numpy(), dtype=torch.float32)
        self.y_gender = torch.tensor(y_gender.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_benefit[idx], self.y_gender[idx]


# Load the data
X_train = pd.read_csv(os.path.join(test_adversarial_dir, 'X_train.csv'),index_col=0)
y_benefit_train = pd.read_csv(os.path.join(test_adversarial_dir, 'y_train.csv'),index_col=0)
y_gender_train = pd.read_csv(os.path.join(test_adversarial_dir, 'sex_train.csv'),index_col=0)
input_dim = X_train.shape[1]    
latent_dim = 8                                    

# create a validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_benefit_train, y_benefit_val = train_test_split(X_train, y_benefit_train, test_size=0.2, random_state=42,
                                                                  stratify=y_benefit_train)

y_gender_val =  y_gender_train.loc[X_val.index]
y_gender_train = y_gender_train.loc[X_train.index]


# create a dataloader
train_dataset = CustomDataset(X_train, y_benefit_train, y_gender_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(X_val, y_benefit_val, y_gender_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# %%

X_test = pd.read_csv(os.path.join(test_adversarial_dir, 'X_test.csv'),index_col=0)
y_benefit_test = pd.read_csv(os.path.join(test_adversarial_dir, 'y_test.csv'),index_col=0)
y_gender_test = pd.read_csv(os.path.join(test_adversarial_dir, 'sex_test.csv'),index_col=0)

# create a dataloader
test_dataset = CustomDataset(X_test, y_benefit_test, y_gender_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# %%
# Define the networks
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, dropout_rate=0.2):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

class BinaryClassifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim=16, dropout_rate=0.2):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim=16, dropout_rate=0.2,num_classes=2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


class Classifier_MultiLayer(nn.Module):
    def __init__(self, latent_dim, hidden_dim=16, dropout_rate=0.2,num_classes=2,num_hidden_layers=1):
        super(Classifier_MultiLayer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        for _ in range(num_hidden_layers-1):
            self.network.add_module(nn.Linear(hidden_dim, hidden_dim))
            self.network.add_module(nn.ReLU())
            self.network.add_module(nn.Dropout(dropout_rate))
        self.network.add_module(nn.Linear(hidden_dim, num_classes))
        self.network.add_module(nn.Softmax(dim=1))

    def forward(self, x):
        return self.network(x)
# %%


# Help on creating a adverarial network with the classifier
# https://xebia.com/blog/fairness-in-machine-learning-with-pytorch/

# this is conceptually similar to 
# Adversarial deconfounding autoencoder for learning robust gene expression embeddings


# Instantiate the networks
feature_extractor = FeatureExtractor(input_dim=input_dim, latent_dim=latent_dim)
benefit_classifier = BinaryClassifier(latent_dim=latent_dim)
gender_classifier = Classifier(latent_dim=latent_dim)

# compute the gender imbalance
genders = train_loader.dataset.y_gender.numpy()
gender_weights = np.array([1/np.sum(genders==0),1/np.sum(genders==1)])

# Define the loss function
criterion = nn.BCELoss()
GC_criterion = nn.CrossEntropyLoss(weight=torch.tensor(gender_weights, dtype=torch.float32))
adv_factor = 3

# Define the optimizers
optimizer_FE_BC = torch.optim.Adam(list(feature_extractor.parameters()) + list(benefit_classifier.parameters()))
optimizer_GC = torch.optim.Adam(gender_classifier.parameters())

# Training loop
for epoch in range(num_epochs):
    total_loss_BC = 0.0
    total_loss_GC = 0.0

    for batch in train_loader:
        # Unpack the batch
        X, y_benefit, y_gender = batch

        # zero the gradients
        optimizer_FE_BC.zero_grad()
        optimizer_GC.zero_grad()

        # Forward pass through the primary networks
        latent = feature_extractor(X)
        benefit_pred = benefit_classifier(latent)

        # Detach the latent variable from the computation graph
        latent2 = latent.detach()
        latent2.requires_grad_()

        # Forward pass through the Gender Classifier
        gender_pred = gender_classifier(latent2)

        # Compute the losses
        loss_BC = criterion(benefit_pred, y_benefit)
        loss_GC = GC_criterion(gender_pred, y_gender.view(-1).long())

        # Backward pass and optimize the primary classifiers
        (loss_BC - adv_factor*loss_GC).backward(retain_graph=True)
        optimizer_FE_BC.step()

        ######## Unsure if this part is necessary
        # Recompute the gender predictions and the gender loss
        # Recompute the latent space
        latent = feature_extractor(X)

        # Detach the latent variable from the computation graph
        latent2 = latent.detach()
        latent2.requires_grad_()
        gender_pred = gender_classifier(latent2)
        loss_GC = GC_criterion(gender_pred, y_gender.view(-1).long())
        ##########

        # Backward pass and optimize the adversarial classifiers
        loss_GC.backward()
        optimizer_GC.step()

        # Accumulate the losses
        total_loss_BC += loss_BC.item()
        total_loss_GC += loss_GC.item()

    for batch in train_loader:
        # Unpack the batch
        X, y_benefit, y_gender = batch


        optimizer_GC.zero_grad()
        latent = feature_extractor(X)
        # Detach the latent variable from the computation graph
        latent2 = latent.detach()
        latent2.requires_grad_()

        # Forward pass through the Gender Classifier
        gender_pred = gender_classifier(latent2)
        loss_GC = GC_criterion(gender_pred, y_gender.view(-1).long())
        ##########

        # Backward pass and optimize the adversarial classifiers
        loss_GC.backward()
        optimizer_GC.step()



    # Print the losses at each epoch
    print(f"Epoch {epoch+1}: Benefit Classifier Loss: {total_loss_BC}, Gender Classifier Loss: {total_loss_GC}")
    # print(f"Epoch {epoch+1}: Benefit Loss = {loss_BC.item()}, Gender Loss = {loss_GC.item()}")

# save the models
torch.save(feature_extractor, output_dir+'/feature_extractor.pth')
torch.save(benefit_classifier, output_dir+'/benefit_classifier.pth')
torch.save(gender_classifier, output_dir+'/gender_classifier.pth')

# %% Train a new Gender Classifier to see if the latent representation really removes useful information

gender_classifier_2 = Classifier(latent_dim=latent_dim)
optimizer_GC_2 = torch.optim.Adam(gender_classifier_2.parameters())

# Training loop
for epoch in range(2*num_epochs):
    total_loss_GC_2 = 0.0

    for batch in train_loader:
        # Unpack the batch
        X, y_benefit, y_gender = batch

        # Forward pass through the networks
        latent = feature_extractor(X)
        latent2 = latent.detach()
        latent2.requires_grad_()
        gender_pred = gender_classifier_2(latent2) 
        # gender_pred = gender_classifier_2(latent)  # Use the new gender classifier

        # Compute the loss for Gender Classifier
        loss_GC_2 = GC_criterion(gender_pred, y_gender.view(-1).long())

        # Backward pass and optimization for Gender Classifier
        optimizer_GC_2.zero_grad()
        loss_GC_2.backward()
        optimizer_GC_2.step()

        # Accumulate the losses
        total_loss_GC_2 += loss_GC_2.item()

    # Print the losses at each epoch
    print(f"Epoch {epoch+1}: Gender Classifier 2 Loss: {total_loss_GC_2}")


torch.save(feature_extractor, output_dir+'/feature_extractor_2.pth')
torch.save(gender_classifier_2, output_dir+'/gender_classifier_2.pth')


# %% Evaluation
from torchmetrics import Accuracy

# Instantiate accuracy metrics
benefit_accuracy = Accuracy(task="binary", average='macro')
gender_accuracy = Accuracy(task="binary", average='macro')
gender_accuracy_2 = Accuracy(task="binary", average='macro')

# Evaluation loop
with torch.no_grad():
    for batch in train_loader:
        # Unpack the batch
        X, y_benefit, y_gender = batch

        # Forward pass through the networks
        latent = feature_extractor(X)
        benefit_pred = benefit_classifier(latent)
        gender_pred = gender_classifier(latent)
        gender_pred_2 = gender_classifier_2(latent)
        gender_pred = torch.argmax(gender_pred, dim=1)
        gender_pred_2 = torch.argmax(gender_pred_2, dim=1)
        # Compute the accuracy for Benefit Classifier
        benefit_accuracy.update(benefit_pred, y_benefit)

        # Compute the accuracy for Gender Classifier
        gender_accuracy.update(gender_pred, y_gender.view(-1))

        # Compute the accuracy for Gender Classifier 2
        gender_accuracy_2.update(gender_pred_2,y_gender.view(-1))

    # Get the overall accuracy scores
    benefit_accuracy_score = benefit_accuracy.compute()
    gender_accuracy_score = gender_accuracy.compute()
    gender_accuracy_2_score = gender_accuracy_2.compute()

# Print the accuracy scores
print(f"Benefit Classifier Accuracy: {benefit_accuracy_score}")
print(f"Gender Classifier Accuracy: {gender_accuracy_score}")    # Evaluation
print(f"Gender 2 Classifier Accuracy: {gender_accuracy_2_score}")    # Evaluation


# %% Run evaluation on the Validation Set
# Instantiate accuracy metrics
benefit_accuracy = Accuracy(task="binary", average='macro')
gender_accuracy = Accuracy(task="binary", average='macro')
gender_accuracy_2 = Accuracy(task="binary", average='macro')

# Evaluation loop
with torch.no_grad():
    for batch in val_loader:
        # Unpack the batch
        X, y_benefit, y_gender = batch

        # Forward pass through the networks
        latent = feature_extractor(X)
        benefit_pred = benefit_classifier(latent)
        gender_pred = gender_classifier(latent)
        gender_pred_2 = gender_classifier_2(latent)
        gender_pred = torch.argmax(gender_pred, dim=1)
        gender_pred_2 = torch.argmax(gender_pred_2, dim=1)
        # Compute the accuracy for Benefit Classifier
        benefit_accuracy.update(benefit_pred, y_benefit)

        # Compute the accuracy for Gender Classifier
        gender_accuracy.update(gender_pred, y_gender.view(-1))

        # Compute the accuracy for Gender Classifier 2
        gender_accuracy_2.update(gender_pred_2,y_gender.view(-1))

    # Get the overall accuracy scores
    benefit_accuracy_score = benefit_accuracy.compute()
    gender_accuracy_score = gender_accuracy.compute()
    gender_accuracy_2_score = gender_accuracy_2.compute()

# Print the accuracy scores
print(f"Validation Benefit Classifier Accuracy: {benefit_accuracy_score}")
print(f"Validation Gender Classifier Accuracy: {gender_accuracy_score}")    # Evaluation
print(f"Validation Gender 2 Classifier Accuracy: {gender_accuracy_2_score}")    # Evaluation





# %% Run evaluation on the test set




# Instantiate accuracy metrics
benefit_accuracy = Accuracy(task="binary", average='macro')
gender_accuracy = Accuracy(task="binary", average='macro')
gender_accuracy_2 = Accuracy(task="binary", average='macro')

# Evaluation loop
with torch.no_grad():
    for batch in test_loader:
        # Unpack the batch
        X, y_benefit, y_gender = batch

        # Forward pass through the networks
        latent = feature_extractor(X)
        benefit_pred = benefit_classifier(latent)
        gender_pred = gender_classifier(latent)
        gender_pred_2 = gender_classifier_2(latent)
        gender_pred = torch.argmax(gender_pred, dim=1)
        gender_pred_2 = torch.argmax(gender_pred_2, dim=1)
        # Compute the accuracy for Benefit Classifier
        benefit_accuracy.update(benefit_pred, y_benefit)

        # Compute the accuracy for Gender Classifier
        gender_accuracy.update(gender_pred, y_gender.view(-1))

        # Compute the accuracy for Gender Classifier 2
        gender_accuracy_2.update(gender_pred_2,y_gender.view(-1))

    # Get the overall accuracy scores
    benefit_accuracy_score = benefit_accuracy.compute()
    gender_accuracy_score = gender_accuracy.compute()
    gender_accuracy_2_score = gender_accuracy_2.compute()

# Print the accuracy scores
print(f"Test Benefit Classifier Accuracy: {benefit_accuracy_score}")
print(f"Test Gender Classifier Accuracy: {gender_accuracy_score}")    # Evaluation
print(f"Test Gender 2 Classifier Accuracy: {gender_accuracy_2_score}")    # Evaluation


# %%
# gender_classifier_2 = torch.load(output_dir+'/gender_classifier_2.pth')
# feature_extractor = torch.load(output_dir+'/feature_extractor.pth')
# # with torch.no_grad():
# #     train_latent = feature_extractor(train_loader.dataset.X)

# # gender = train_loader.dataset.y_gender
# # benefit = train_loader.dataset.y_benefit

# with torch.no_grad():
#     val_latent = feature_extractor(val_loader.dataset.X)
# gender = val_loader.dataset.y_gender
# benefit = val_loader.dataset.y_benefit

# with torch.no_grad():
#     test_latent = feature_extractor(test_loader.dataset.X)
# gender = test_loader.dataset.y_gender
# benefit = test_loader.dataset.y_benefit
# %%
# plot the umap of the latent representations

# import umap
# import matplotlib.pyplot as plt
# import seaborn as sns

# %%
# %%

# reducer = umap.UMAP()
# embedding = reducer.fit_transform(val_latent)

# umap_df = pd.DataFrame(embedding, columns=['UMAP1','UMAP2'])
# umap_df['gender'] = gender
# umap_df['benefit'] = benefit

# sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='gender')
# # %%

# sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='benefit')

# %%
