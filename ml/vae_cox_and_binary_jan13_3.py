# %%
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error
import json
from torch.utils.data import TensorDataset
from lifelines.utils import concordance_index
from loss import CoxPHLoss
# ! pip install scikit-survival
# from sksurv.metrics import concordance_index_censored

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, AUROC
# %%
###########

task_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination/hilic_pos_2024_feb_02_read_norm/subset all_studies with align score 0.3 from Merge_Jan25_align_80_40_default/num_cohorts_thresh_0.5/combat_Multi'
output_path =  os.path.join(task_dir,'vae_cox_0')
finetune_output_path = os.path.join(output_path,'finetune_0')
os.makedirs(output_path,exist_ok=True)

X_pretrain = pd.read_csv(os.path.join(task_dir,'X_pretrain.csv'),index_col=0)
# X_train = pd.read_csv(os.path.join(task_dir,'X_train.csv'),index_col=0)
# X_val = pd.read_csv(os.path.join(task_dir,'X_val.csv'),index_col=0)

# y_pretrain = pd.read_csv(os.path.join(task_dir,'y_pretrain.csv'),index_col=0)
# y_train = pd.read_csv(os.path.join(task_dir,'y_train.csv'),index_col=0)
# y_val = pd.read_csv(os.path.join(task_dir,'y_val.csv'),index_col=0)
X_finetune = pd.read_csv(os.path.join(task_dir,'X_finetune.csv'),index_col=0)
y_finetune = pd.read_csv(os.path.join(task_dir,'y_finetune.csv'),index_col=0)

baseline_select = y_finetune[y_finetune['study_week'] == 'baseline'].index.tolist()
nivo_select = y_finetune[y_finetune['Treatment'] == 'NIVOLUMAB'].index.tolist()

finetune_classification_col = 'Benefit'
finetune_survival_col = 'PFS' 
finetune_survival_censor_col = 'PFS_Event' # 1 = event, 0 = censored

y_finetune_survival = y_finetune[finetune_survival_col].copy()
y_finetune_censor = y_finetune[finetune_survival_censor_col].copy()
y_finetune_classification =  y_finetune[finetune_classification_col].copy()

# y_train_classification
y_classification_name_dict = {
    'CB': 1
    ,'NCB': 0
    ,'ICB': 0.5
}

y_finetune_classification = y_finetune_classification.map(y_classification_name_dict)
# y_finetune_classification.dropna(inplace=True)
# y_train_classification[y_train_classification=='ICB'] == 0.5

keep_idx = list(set(baseline_select).intersection(set(y_finetune_classification.index.tolist())).intersection(set(nivo_select)))
print('number of samples used for fine-tuning:',    len(keep_idx))
X_finetune = X_finetune.loc[keep_idx]
y_finetune_survival = y_finetune_survival.loc[keep_idx]
y_finetune_censor = y_finetune_censor.loc[keep_idx].astype(int)
y_finetune_classification = y_finetune_classification.loc[keep_idx]#.astype(int)

# Train test split
finetune_val_size = 0.2
finetune_val_random_state = 42
X_train, X_val, _, _, = train_test_split(X_finetune, 
                                          y_finetune_classification,  test_size=finetune_val_size, 
                                          random_state=finetune_val_random_state, stratify=y_finetune_classification)

y_train_survival = y_finetune_survival.loc[X_train.index.tolist()]
y_train_censor = y_finetune_censor.loc[X_train.index.tolist()]
y_train_classification = y_finetune_classification.loc[X_train.index.tolist()]

y_val_survival = y_finetune_survival.loc[X_val.index.tolist()]
y_val_censor = y_finetune_censor.loc[X_val.index.tolist()]
y_val_classification = y_finetune_classification.loc[X_val.index.tolist()]

y_train_classification[y_train_classification==0.5] = np.nan
y_val_classification[y_val_classification==0.5] = np.nan




# %% Load the Test Data
X_test = pd.read_csv(os.path.join(task_dir,'X_test.csv'),index_col=0)
y_test = pd.read_csv(os.path.join(task_dir,'y_test.csv'),index_col=0)
baseline_test_select = y_test[y_test['study_week'] == 'baseline'].index.tolist()

y_test_survival = y_test[finetune_survival_col].copy()
y_test_censor = y_test[finetune_survival_censor_col].copy()
y_test_classification =  y_test[finetune_classification_col].copy()
y_test_classification = y_test_classification.map(y_classification_name_dict)
y_test_classification=y_test_classification.loc[baseline_test_select]
# y_test_classification.dropna(inplace=True)

X_test = X_test.loc[y_test_classification.index.tolist()]
y_test_survival = y_test_survival.loc[y_test_classification.index.tolist()]
y_test_censor = y_test_censor.loc[y_test_classification.index.tolist()]
y_test_classification = y_test_classification.loc[y_test_classification.index.tolist()]

y_test_classification[y_test_classification==0.5] = np.nan

# TODO:
# 1. classification loss function to handle unblanced data
# 2. classification loss function to handle censored data
# 3. Coxph loss function vs regression loss function

# %%


pretrain_epochs = 2500 
hidden_dim = 128
latent_dim = 64
batch_size = 256
finetune_epochs = 2500  

###########

# 1. Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# KLD loss
def kld_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# 2. Split the pretraining data
input_dim = X_pretrain.shape[1]
pretrain_data = torch.tensor(X_pretrain.to_numpy(), dtype=torch.float32)

# 3. Pretrain the VAE model
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.BCELoss(reduction='sum')
# use MSE loss
criterion = nn.MSELoss(reduction='sum')
vae_model_loc = f'{output_path}/vae.pth'

if not os.path.exists(vae_model_loc):
    print('Pretraining VAE...')
    model.train()

    pretrain_size = int(0.8 * len(pretrain_data))
    pretrain_val_size = len(pretrain_data) - pretrain_size
    train_data, val_data = random_split(pretrain_data, [pretrain_size, pretrain_val_size])

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    epochs_list = []
    best_val_loss = np.inf
    best_epoch = 0

    for epoch in range(pretrain_epochs):
        for batch in DataLoader(train_data, batch_size=batch_size,shuffle=True):
            recon, mu, logvar = model(batch)
            loss = criterion(recon, batch) + kld_loss(mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print the training and validation loss every 10 epochs
        if epoch % 5 == 0:
            model.eval()
            train_loss = 0
            val_loss = 0
            with torch.no_grad():
                # Evaluate on training data
                for batch in DataLoader(train_data, batch_size=batch_size,shuffle=True):
                    recon, mu, logvar = model(batch)
                    train_loss += (criterion(recon, batch) + kld_loss(mu, logvar))*batch.shape[0]/len(train_data)
                # Evaluate on validation data
                for batch in DataLoader(val_data, batch_size=batch_size,shuffle=True):
                    recon, mu, logvar = model(batch)
                    val_loss += (criterion(recon, batch) + kld_loss(mu, logvar))*batch.shape[0]/len(val_data)
            print('Epoch: ', epoch ,'Train loss:', train_loss.item(), 'Validation loss:', val_loss.item())
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            epochs_list.append(epoch)
            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{output_path}/vae.pth')
                torch.save(optimizer.state_dict(), f'{output_path}/vae_optimizer.pth')
                print('Best Model saved!')

            model.train()

    # 4. Evaluate the VAE model
    # Load the best model
    model.load_state_dict(torch.load(f'{output_path}/vae.pth'))

    model.eval()
    val_loss = 0
    train_loss = 0
    with torch.no_grad():
        for batch in DataLoader(val_data, batch_size=batch_size,shuffle=True):
            recon, mu, logvar = model(batch)
            val_loss += (criterion(recon, batch) + kld_loss(mu, logvar))*batch.shape[0]/len(val_data)
        for batch in DataLoader(train_data, batch_size=batch_size,shuffle=True):
            recon, mu, logvar = model(batch)
            train_loss += (criterion(recon, batch) + kld_loss(mu, logvar))*batch.shape[0]/len(train_data)
    print('Final Train loss:', train_loss.item(), 'Final Validation loss:', val_loss.item())


    # 5. Save the pretrained model and its hyperparameters and validation results
    torch.save(model.state_dict(), f'{output_path}/vae.pth')
    with open(f'{output_path}/vae.json', 'w') as f:
        json.dump({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'pretrain_epochs': pretrain_epochs,
            'validation_loss': loss.item(),
            'epochs_list': epochs_list,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, f)
else:
    print('Already pretrained VAE.')

print('Loading pretrained VAE...')
model.load_state_dict(torch.load(f'{output_path}/vae.pth'))
# model.eval()
# set the model to train mode to allow finetuning
model.train()
for param in model.parameters():
    param.requires_grad = True
    

# 6. Create a CoxPH model head
class CoxPHHead(nn.Module):
    def __init__(self, input_dim):
        super(CoxPHHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LeakyReLU(),
            # apply dropout to prevent overfitting
            nn.Dropout(0.2),
            nn.Linear(int(input_dim/2), 1))

    def forward(self, x):
        return self.fc(x)

coxph_head = CoxPHHead(latent_dim)

# Create a Binary Classification head
class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LeakyReLU(),
            # apply dropout to prevent overfitting
            nn.Dropout(0.2),
            nn.Linear(int(input_dim/2), 1),
        )

    def forward(self, x):
        return self.fc(x)

binary_classification_head = BinaryClassificationHead(latent_dim)

def binary_classification_loss_ignore_nan(output, target, class_weights):
    # Create a mask of non-nan values
    mask = ~torch.isnan(target)
    
    # Use the mask to select non-nan values
    output = output[mask]
    target = target[mask]
    
    # Calculate the weighted binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(output, target, weight=class_weights[target.long()], reduction='mean')
    
    return loss

# Create a DataLoader that includes censoring information

# class_weights = torch.tensor([1, 1], dtype=torch.float32)
# Determine the class weights from the training data, y_train_classification
weight_0 = 1 / (y_train_classification == 0).sum()
weight_1 = 1 / (y_train_classification == 1).sum()
class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)

batch_size = 128
train_dataset = TensorDataset(torch.tensor(X_train.to_numpy(),dtype=torch.float32),
                              torch.tensor(y_train_survival.to_numpy()), 
                              torch.tensor(y_train_classification.to_numpy()), 
                              torch.tensor(y_train_censor.to_numpy()))
val_dataset = TensorDataset(torch.tensor(X_val.to_numpy(),dtype=torch.float32),
                                torch.tensor(y_val_survival.to_numpy()), 
                                torch.tensor(y_val_classification.to_numpy()), 
                                torch.tensor(y_val_censor.to_numpy()))

test_dataset = TensorDataset(torch.tensor(X_test.to_numpy(),dtype=torch.float32),
                                torch.tensor(y_test_survival.to_numpy()),
                                torch.tensor(y_test_classification.to_numpy()),
                                torch.tensor(y_test_censor.to_numpy()))


train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# useful Deep learning survival analysis article, and multi-task learning
# https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/#deeplearning_sa

# Adjust the CoxPH loss function to handle censored data
# Is this the right loss function to use?
# see https://jmlr.org/papers/volume20/18-424/18-424.pdf
# https://www.nature.com/articles/s41598-021-86327-7
# The Nature paper has Survival Forests and Survival SVMs
#pycox package may be useful
# DeepSurv : https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1
# def coxph_loss(output, target, censor):
#     output = output.reshape(-1)
#     target = target.reshape(-1)
#     censor = censor.reshape(-1)
#     exp_output = torch.exp(output)
#     risk_set = exp_output.ge(exp_output.view(-1, 1)).type(torch.float32)
#     log_risk = torch.log(torch.sum(risk_set * exp_output.view(-1, 1), dim=0) + 1e-7)
#     uncensored_loss = torch.mul(output - log_risk, target)
#     loss = torch.sum(torch.mul(uncensored_loss, censor))
#     return torch.neg(loss)


# def cox_regression_loss(output, time, event):
#     """
#     Compute Cox Regression Loss.
    
#     Parameters:
#     output (Tensor): The model's output, expected to be the log hazard ratio.
#     time (Tensor): The survival time.
#     event (Tensor): The event indicator, 1 if the event of interest has occurred, 0 otherwise.
    
#     Returns:
#     Tensor: The computed Cox Regression Loss.
#     """
#     # Get risk scores
#     risk_scores = torch.exp(output)

#     # Get the cases where the event of interest has occurred
#     event_cases = event == 1

#     # Compute the total risk scores for all individuals at risk at each time
#     total_risk_scores = torch.empty(time.shape[0])
#     for t in torch.unique(time):
#         at_risk = time >= t
#         total_risk_scores[time == t] = risk_scores[at_risk].sum()

#     # Compute the loss
#     loss = -torch.sum(torch.log(risk_scores[event_cases] / total_risk_scores[event_cases]))

#     return loss

coxph_loss = CoxPHLoss()


# Decide the loss weights
reconstruction_loss_weight = 0
coxph_loss_weight = 1
classification_loss_weight = 1

# Initialize running averages for loss normalization
reconstruction_loss_avg = 0.1
coxph_loss_avg = 0.1
classification_loss_avg = 0.1
beta = 0.9  # decay factor for the running averages

os.makedirs(finetune_output_path,exist_ok=True)

train_loss_dict = {}
val_loss_dict = {}
train_loss_dict['reconstruction_loss'] = []
train_loss_dict['coxph_loss'] = []
train_loss_dict['classification_loss'] = []
train_loss_dict['joint_loss'] = []
val_loss_dict['reconstruction_loss'] = []
val_loss_dict['coxph_loss'] = []
val_loss_dict['classification_loss'] = []
val_loss_dict['joint_loss'] = []
best_train_joint_loss = np.inf
best_val_joint_loss = np.inf
best_epoch = 0
finetune_learning_rate = 1e-4
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(coxph_head.parameters()) + list(binary_classification_head.parameters()), 
    lr=finetune_learning_rate)

# Fine-tune the VAE + CoxPH + Binary Classification model
for epoch in range(finetune_epochs):
    for (batch, target_survival, target_classification, target_censor) in train_loader:
        recon, mu, logvar = model(batch)
        output_survival = coxph_head(mu)
        output_classification = binary_classification_head(mu)
        reconstruction_loss = criterion(recon, batch) - kld_loss(mu, logvar)
        coxph_loss_val = coxph_loss(output_survival, target_survival, target_censor)
        # classification_loss = nn.BCELoss()(output_classification.squeeze(1).float(), target_classification.float())
        
        #TODO update the classification loss to properly weight the not-nan values in the batch shape
        classification_loss = binary_classification_loss_ignore_nan(
            output_classification.squeeze(1).float(), target_classification.float(), class_weights)

        if (torch.isnan(coxph_loss_val)) or torch.isinf(coxph_loss_val):
            print('coxph_loss_val is nan or inf')

        # Update running averages
        if not torch.isnan(reconstruction_loss):
            reconstruction_loss_avg = beta * reconstruction_loss_avg + (1 - beta) * reconstruction_loss.item()
        if not torch.isnan(coxph_loss_val):
            coxph_loss_avg = beta * coxph_loss_avg + (1 - beta) * coxph_loss_val.item()
        if not torch.isnan(classification_loss):
            classification_loss_avg = beta * classification_loss_avg + (1 - beta) * classification_loss.item()
        
        # Normalize losses
        reconstruction_loss /= reconstruction_loss_avg
        coxph_loss_val /= coxph_loss_avg
        classification_loss /= classification_loss_avg
        
        joint_loss = reconstruction_loss_weight * reconstruction_loss + coxph_loss_weight * coxph_loss_val + classification_loss_weight * classification_loss
        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()
    # print loss every 10 epochs
    if epoch % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, finetune_epochs, joint_loss.item()))

    if epoch % 10 == 0:
        model.eval()
        coxph_head.eval()
        binary_classification_head.eval()
        with torch.no_grad():
            for (batch, target_survival, target_classification, target_censor) in train_loader:
                recon, mu, logvar = model(batch)
                output_survival = coxph_head(mu)
                output_classification = binary_classification_head(mu)
                reconstruction_loss += (criterion(recon, batch) - kld_loss(mu, logvar)) * batch.shape[0] / len(train_dataset)
                coxph_loss_val += (coxph_loss(output_survival, target_survival, target_censor)) * batch.shape[0] / len(train_dataset)
                # classification_loss += (nn.BCELoss()(output_classification.squeeze(1).float(), target_classification.float())) * batch.shape[0] / len(train_dataset)
                
                #TODO update the classification loss to properly weight the not-nan values in the batch shape
                classification_loss += binary_classification_loss_ignore_nan(
                    output_classification.squeeze(1).float(), target_classification.float(), class_weights) * batch.shape[0] / len(train_dataset)
                


            # Normalize losses
            reconstruction_loss /= reconstruction_loss_avg
            coxph_loss_val /= coxph_loss_avg
            classification_loss /= classification_loss_avg
            
            # train_loss_dict['joint_loss'].append((reconstruction_loss + coxph_loss_val + classification_loss).item())
            train_loss_dict['reconstruction_loss'].append(reconstruction_loss.item())
            train_loss_dict['coxph_loss'].append(coxph_loss_val.item())
            train_loss_dict['classification_loss'].append(classification_loss.item())


            train_joint_loss = reconstruction_loss_weight * reconstruction_loss + coxph_loss_weight * coxph_loss_val + classification_loss_weight * classification_loss
            train_loss_dict['joint_loss'].append(train_joint_loss.item())

            print('Epoch: ', epoch ,'Train Joint Loss',train_joint_loss.item()  , 'Train reconstruction loss:', reconstruction_loss.item(), 'Train CoxPH loss:', coxph_loss_val.item(), 'Train classification loss:', classification_loss.item())


            for (batch, target_survival, target_classification, target_censor) in val_loader:
                recon, mu, logvar = model(batch)
                output_survival = coxph_head(mu)
                output_classification = binary_classification_head(mu)
                reconstruction_loss += (criterion(recon, batch) - kld_loss(mu, logvar)) * batch.shape[0] / len(val_dataset)
                coxph_loss_val += (coxph_loss(output_survival, target_survival, target_censor)) * batch.shape[0] / len(val_dataset)
                # classification_loss += (nn.BCELoss()(output_classification.squeeze(1).float(), target_classification.float())) * batch.shape[0] / len(val_dataset)
                classification_loss += binary_classification_loss_ignore_nan(
                    output_classification.squeeze(1).float(), target_classification.float(), class_weights) * batch.shape[0] / len(val_dataset)

            # Normalize losses
            reconstruction_loss /= reconstruction_loss_avg
            coxph_loss_val /= coxph_loss_avg
            classification_loss /= classification_loss_avg

            val_loss_dict['reconstruction_loss'].append(reconstruction_loss.item())
            val_loss_dict['coxph_loss'].append(coxph_loss_val.item())
            val_loss_dict['classification_loss'].append(classification_loss.item())

            #TODO Also keep track of the Accuracy, AUC, and Concordance Index

            val_joint_loss = reconstruction_loss_weight * reconstruction_loss + coxph_loss_weight * coxph_loss_val + classification_loss_weight * classification_loss
            val_loss_dict['joint_loss'].append(val_joint_loss.item())
            if (val_joint_loss.item() < best_val_joint_loss) and (train_joint_loss.item() < best_train_joint_loss) and (epoch > 50):
                best_val_joint_loss = val_joint_loss.item()
                best_train_joint_loss = train_joint_loss.item()
                best_epoch = epoch
                torch.save(model.state_dict(), f'{finetune_output_path}/vae_cox.pth')
                torch.save(coxph_head.state_dict(), f'{finetune_output_path}/coxph_head.pth')
                torch.save(binary_classification_head.state_dict(), f'{finetune_output_path}/binary_classification_head.pth')
                print('Best Model saved!')


            print('Epoch: ', epoch ,'Val Joint Loss',joint_loss.item()  , 'Val reconstruction loss:', reconstruction_loss.item(), 'Val CoxPH loss:', coxph_loss_val.item(), 'Val classification loss:', classification_loss.item())
        
        model.train()
        coxph_head.train()
        binary_classification_head.train()

# load the best models
print('loading the best model from epoch', best_epoch, 'with joint validation loss', best_val_joint_loss, 'and joint train loss', best_train_joint_loss)
model.load_state_dict(torch.load(f'{finetune_output_path}/vae_cox.pth'))
coxph_head.load_state_dict(torch.load(f'{finetune_output_path}/coxph_head.pth'))
binary_classification_head.load_state_dict(torch.load(f'{finetune_output_path}/binary_classification_head.pth'))

# Evaluate this final model using the Training data
model.eval()
coxph_head.eval()
binary_classification_head.eval()
train_accuracy = Accuracy(task='binary',average='weighted')
train_auc = AUROC(task='binary',average='weighted')
concor_index_train_avg = 0
with torch.no_grad():
    for (batch, target_survival, target_classification, target_censor) in train_loader:
        _, mu, _ = model(batch)
        output_survival = coxph_head(mu)
        output_classification = binary_classification_head(mu)
        # Compute the concordance index as a measure of how well the model predicts the order of event times
        concor_index_train = concordance_index(
            event_times=target_survival.numpy(),
            predicted_scores=output_survival.numpy(),
            event_observed=target_censor.numpy()
        )
        concor_index_train_avg += concor_index_train * batch.shape[0] / len(train_dataset)
        # Compute the accuracy of the binary classification
        output_classification_prob = torch.sigmoid(output_classification.squeeze(1).float())
        mask = ~torch.isnan(target_classification.float())
        train_accuracy(output_classification_prob[mask], target_classification.float()[mask])
        train_auc(output_classification_prob[mask], target_classification.float()[mask])
print('Train Concordance Index:',concor_index_train_avg)
print('Train Accuracy:', train_accuracy.compute().item())
print('Train AUC:', train_auc.compute().item())

# Evaluate this final model using the Validation data
model.eval()
coxph_head.eval()
binary_classification_head.eval()
val_accuracy = Accuracy(task='binary',average='weighted')
val_auc = AUROC(task='binary',average='weighted')
concor_index_val_avg = 0
with torch.no_grad():
    for (batch, target_survival, target_classification, target_censor) in val_loader:
        _, mu, _ = model(batch)
        output_survival = coxph_head(mu)
        output_classification = binary_classification_head(mu)
        # Compute the concordance index as a measure of how well the model predicts the order of event times
        concor_index_val = concordance_index(
            event_times=target_survival.numpy(),
            predicted_scores=output_survival.numpy(),
            event_observed=target_censor.numpy()
        )
        concor_index_val_avg += concor_index_val * batch.shape[0] / len(val_dataset)
        # Compute the accuracy of the binary classification
        output_classification_prob = torch.sigmoid(output_classification.squeeze(1).float())
        mask = ~torch.isnan(target_classification.float())
        val_accuracy(output_classification_prob[mask], target_classification.float()[mask])
        val_auc(output_classification_prob[mask], target_classification.float()[mask])
print('Val Concordance Index:',concor_index_val_avg)
print('Val Accuracy:', val_accuracy.compute().item())
print('Val AUC:', val_auc.compute().item())


# Evaluate this final model using the Test data
model.eval()
coxph_head.eval()
binary_classification_head.eval()
test_accuracy = Accuracy(task='binary')
test_auc = AUROC(task='binary')
concor_index_test_avg = 0
with torch.no_grad():
    for (batch, target_survival, target_classification, target_censor) in test_loader:
        _, mu, _ = model(batch)
        output_survival = coxph_head(mu)
        output_classification = binary_classification_head(mu)
        mask = ~torch.isnan(target_survival.float())
        if sum(mask) == 0:
            continue
        # Compute the concordance index as a measure of how well the model predicts the order of event times
        concor_index_test = concordance_index(
            event_times=target_survival.numpy()[mask],
            predicted_scores=output_survival.numpy()[mask],
            event_observed=target_censor.numpy()[mask]
        )
        concor_index_test_avg += concor_index_test * batch.shape[0] / len(test_dataset)
        # Compute the accuracy of the binary classification
        output_classification_prob = torch.sigmoid(output_classification.squeeze(1).float())
        mask = ~torch.isnan(target_classification.float())
        test_accuracy(output_classification_prob[mask], target_classification.float()[mask])
        test_auc(output_classification_prob[mask], target_classification.float()[mask])
print('Test Concordance Index:',concor_index_test_avg)
print('Test Accuracy:', test_accuracy.compute().item())
print('Test AUC:', test_auc.compute().item())




# 9. Save the final model and its hyperparameters and test results
torch.save(model.state_dict(), f'{finetune_output_path}/vae_cox.pth')
torch.save(coxph_head.state_dict(), f'{finetune_output_path}/coxph_head.pth')
with open(f'{finetune_output_path}/vae_cox.json', 'w') as f:
    json.dump({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'num_finetune_samples': X_finetune.shape[0],
        'finetune_classification_col': finetune_classification_col,
        'finetune_survival_col': finetune_survival_col,
        'finetune_survival_censor_col': finetune_survival_censor_col,
        'val_random_state': finetune_val_random_state,
        'val_size': finetune_val_size,
        'finetune_learning_rate': finetune_learning_rate,
        'reconstruction_loss_weight': reconstruction_loss_weight,
        'coxph_loss_weight': coxph_loss_weight,
        'classification_loss_weight': classification_loss_weight,
        'init_reconstruction_loss_avg': reconstruction_loss_avg,
        'init_coxph_loss_avg': coxph_loss_avg,
        'init_classification_loss_avg': classification_loss_avg,
        'avg_decay_factor': beta,
        'finetune_epochs': finetune_epochs,
        'best_epoch': best_epoch,
        'best_train_joint_loss': best_train_joint_loss,
        'best_val_joint_loss': best_val_joint_loss,
        'train_concordance_index':concor_index_train_avg,
        'train_accuracy': train_accuracy.compute().item(),
        'train_auc': train_auc.compute().item(),
        'val_concordance_index':concor_index_val_avg,
        'val_accuracy': val_accuracy.compute().item(),
        'val_auc': val_auc.compute().item(),
        'test_concordance_index':concor_index_test_avg,
        'test_accuracy': test_accuracy.compute().item(),
        'test_auc': test_auc.compute().item(),
        'train_loss_dict': train_loss_dict,
        'val_loss_dict': val_loss_dict
    }, f)



# %%
