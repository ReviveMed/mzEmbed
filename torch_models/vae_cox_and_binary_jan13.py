import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error
import json
from torch.utils.data import TensorDataset
from lifelines.utils import concordance_index, concordance_index_censored

# 1. Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# 2. Split the pretraining data
pretrain_data = torch.tensor(X_pretrain)
train_size = int(0.8 * len(pretrain_data))
val_size = len(pretrain_data) - train_size
train_data, val_data = random_split(pretrain_data, [train_size, val_size])

# 3. Pretrain the VAE model
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss(reduction='sum')

for epoch in range(epochs):
    for batch in DataLoader(train_data, batch_size=batch_size):
        recon, mu, logvar = model(batch)
        loss = criterion(recon, batch) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Evaluate the VAE model
model.eval()
with torch.no_grad():
    recon, _, _ = model(torch.tensor(val_data))
    loss = criterion(recon, torch.tensor(val_data))
print('Validation loss:', loss.item())

# 5. Save the pretrained model and its hyperparameters and validation results
torch.save(model.state_dict(), 'vae.pth')
with open('vae.json', 'w') as f:
    json.dump({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'validation_loss': loss.item()
    }, f)


# 6. Create a CoxPH model head
class CoxPHHead(nn.Module):
    def __init__(self, input_dim):
        super(CoxPHHead, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

coxph_head = CoxPHHead(latent_dim)

# Create a Binary Classification head
class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

binary_classification_head = BinaryClassificationHead(latent_dim)


# Create a DataLoader that includes censoring information
train_dataset = TensorDataset(X_train, y_train_coxph, y_train_classification, censor_train)
test_dataset = TensorDataset(X_test, y_test_coxph, y_test_classification, censor_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Adjust the CoxPH loss function to handle censored data
def coxph_loss(output, target, censor):
    output = output.reshape(-1)
    target = target.reshape(-1)
    censor = censor.reshape(-1)
    exp_output = torch.exp(output)
    risk_set = exp_output.ge(exp_output.view(-1, 1)).type(torch.float32)
    log_risk = torch.log(torch.sum(risk_set * exp_output.view(-1, 1), dim=0) + 1e-10)
    uncensored_loss = torch.mul(output - log_risk, target)
    loss = torch.sum(torch.mul(uncensored_loss, censor))
    return torch.neg(loss)

# # Fine-tune the VAE + CoxPH + Binary Classification model
# for epoch in range(epochs):
#     for (batch, target_coxph, target_classification, censor) in train_loader:
#         recon, mu, logvar = model(batch)
#         output_coxph = coxph_head(mu)
#         output_classification = binary_classification_head(mu)
#         reconstruction_loss = criterion(recon, batch) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         coxph_loss = coxph_loss(output_coxph, target_coxph, censor)
#         classification_loss = nn.BCELoss()(output_classification, target_classification)
#         joint_loss = reconstruction_loss + coxph_loss + classification_loss
#         optimizer.zero_grad()
#         joint_loss.backward()
#         optimizer.step()

# Decide the loss weights
reconstruction_loss_weight = 1
coxph_loss_weight = 1
classification_loss_weight = 1

# Initialize running averages for loss normalization
reconstruction_loss_avg = 0.1
coxph_loss_avg = 0.1
classification_loss_avg = 0.1
beta = 0.9  # decay factor for the running averages

# Fine-tune the VAE + CoxPH + Binary Classification model
for epoch in range(epochs):
    for (batch, target_coxph, target_classification, censor) in train_loader:
        recon, mu, logvar = model(batch)
        output_coxph = coxph_head(mu)
        output_classification = binary_classification_head(mu)
        reconstruction_loss = criterion(recon, batch) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        coxph_loss = coxph_loss(output_coxph, target_coxph, censor)
        classification_loss = nn.BCELoss()(output_classification, target_classification)
        
        # Update running averages
        reconstruction_loss_avg = beta * reconstruction_loss_avg + (1 - beta) * reconstruction_loss.item()
        coxph_loss_avg = beta * coxph_loss_avg + (1 - beta) * coxph_loss.item()
        classification_loss_avg = beta * classification_loss_avg + (1 - beta) * classification_loss.item()
        
        # Normalize losses
        reconstruction_loss /= reconstruction_loss_avg
        coxph_loss /= coxph_loss_avg
        classification_loss /= classification_loss_avg
        
        joint_loss = reconstruction_loss_weight * reconstruction_loss + coxph_loss_weight * coxph_loss + classification_loss_weight * classification_loss
        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()



# Evaluate this final model
model.eval()
coxph_head.eval()
binary_classification_head.eval()
with torch.no_grad():
    for (batch, target_coxph, target_classification, censor) in test_loader:
        _, mu, _ = model(batch)
        output_coxph = coxph_head(mu)
        output_classification = binary_classification_head(mu)
        # Compute the concordance index as a measure of how well the model predicts the order of event times
        concordance_index = concordance_index_censored(target_coxph['event'], target_coxph['time'], output_coxph.numpy())
        # Compute the accuracy of the binary classification
        accuracy = ((output_classification > 0.5).numpy() == target_classification).mean()
print('Test Concordance Index:', concordance_index[0])
print('Test Accuracy:', accuracy)



# 9. Save the final model and its hyperparameters and test results
torch.save(model.state_dict(), 'vae_cox.pth')
torch.save(coxph_head.state_dict(), 'coxph_head.pth')
with open('vae_cox.json', 'w') as f:
    json.dump({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'test_concordance_index': concordance_index[0],
        'test_accuracy': accuracy
    }, f)

