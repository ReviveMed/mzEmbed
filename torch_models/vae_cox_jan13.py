import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error
import json

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

# CoxPH loss function
def coxph_loss(output, target):
    output = output.reshape(-1)
    target = target.reshape(-1)
    exp_output = torch.exp(output)
    risk_set = exp_output.ge(exp_output.view(-1, 1)).type(torch.float32)
    log_risk = torch.log(torch.sum(risk_set * exp_output.view(-1, 1), dim=0) + 1e-10)
    loss = torch.sum(torch.mul(output - log_risk, target))
    return torch.neg(loss)

# 7. Fine-tune the VAE + CoxPH model
for epoch in range(epochs):
    for batch, target in zip(DataLoader(X_train, batch_size=batch_size), DataLoader(y_train, batch_size=batch_size)):
        recon, mu, logvar = model(batch)
        output = coxph_head(mu)
        reconstruction_loss = criterion(recon, batch) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        coxph_loss = coxph_loss(output, target)
        joint_loss = reconstruction_loss + coxph_loss
        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()

# 8. Evaluate this final model
model.eval()
coxph_head.eval()
with torch.no_grad():
    _, mu, _ = model(torch.tensor(X_test))
    output = coxph_head(mu)
    # Compute the concordance index as a measure of how well the model predicts the order of event times
    concordance_index = concordance_index_censored(y_test['event'], y_test['time'], output.numpy())
print('Test Concordance Index:', concordance_index[0])



# 9. Save the final model and its hyperparameters and test results
torch.save(model.state_dict(), 'vae_cox.pth')
torch.save(coxph_head.state_dict(), 'coxph_head.pth')
with open('vae_cox.json', 'w') as f:
    json.dump({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'test_concordance_index': concordance_index[0]
    }, f)

