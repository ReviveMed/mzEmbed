import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from models.PretrainVAE import PretrainVAE



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




class SupervisedVAE(PretrainVAE):
    def __init__(self, task_type='classification', num_classes=2, **kwargs):
        super().__init__(**kwargs)
        
        self.task_type = task_type
        self.num_classes = num_classes
        
        # Classification head for binary or multi-class classification
        self.classification_head = nn.Linear(self.latent_size, num_classes)
        
        # Cox regression head (for survival analysis)
        self.cox_head = nn.Linear(self.latent_size, 1)  # Single output for Cox
        
        
    def forward(self, x):
        
        # Assuming the encoder returns a single vector with the first half as mu and the second half as logvar
        encoded = self.encoder(x)  # Single output from encoder
        
        # Debugging: Print the shape of the encoder output
        print(f"Encoded shape: {encoded.shape}")

        # Reshape the encoded vector to be [batch_size, latent_dim]
        if len(encoded.shape) == 1:  # If it's a 1D tensor
            encoded = encoded.view(1, -1)  # Reshape to [1, latent_size]

        # Split the encoded vector into mu and logvar
        mu, logvar = torch.chunk(encoded, 2, dim=1)  # Split the vector along the feature dimension
    
        
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        # Supervised output
        if self.task_type == 'classification':
            if self.classification_head.out_features == 1:
                # Binary classification
                supervised_out = torch.sigmoid(self.classification_head(z))  # Sigmoid for binary
            else:
                # Multi-class classification
                supervised_out = self.classification_head(z)  # Raw logits for CrossEntropyLoss
      
        elif self.task_type == 'cox':
            supervised_out = self.cox_head(z)

        return recon_x, mu, logvar, supervised_out


    def loss_function(self, x, recon_x, mu, logvar, supervised_out, y, duration=None,event=None, lambda_sup=1.0):
        """
        Compute the total loss including:
        - VAE reconstruction loss
        - KL divergence loss
        - Supervised loss based on the number of classes in y
        """
        # VAE losses
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Supervised loss based on the number of classes
        if self.task_type == 'classification':
            
            if self.num_classes == 2:
                # Binary classification (MaskedBCELoss)
                supervised_loss_fn = MaskedBCELoss()
                supervised_loss = supervised_loss_fn(supervised_out, y)
            else:
                # Multi-class classification (MaskedCrossEntropyLoss)
                supervised_loss_fn = MaskedCrossEntropyLoss(num_classes=self.num_classes)
                supervised_loss = supervised_loss_fn(supervised_out, y)
        
        elif self.task_type == 'cox':
            # CoxPH loss for survival analysis
            supervised_loss_fn = CoxPHLoss()
            supervised_loss = supervised_loss_fn(supervised_out, duration, event)

        # Total loss = reconstruction + KL + supervised loss
        total_loss = recon_loss + self.kl_weight * kl_loss + lambda_sup * supervised_loss
        
        return recon_loss, kl_loss, supervised_loss, total_loss
