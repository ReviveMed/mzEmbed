import torch
from torch import nn
import torch.nn.functional as F

from models.models_VAE import VAE

class ConditionalVAE(VAE):
    def __init__(self, pretrained_vae, condition_size, **kwargs):
        
        # Extract VAE-specific arguments and initialize the parent VAE class
        vae_kwargs = {
            'input_size': int(kwargs.get('input_size', 1)),
            'dropout_rate': float(kwargs.get('dropout_rate', 0.2)),
            'activation': kwargs.get('activation', 'leakyrelu'),
            'use_batch_norm': kwargs.get('use_batch_norm', False),
            'act_on_latent_layer': kwargs.get('act_on_latent_layer', False),
            'verbose': kwargs.get('verbose', False)
        }
        
        super(ConditionalVAE, self).__init__(**vae_kwargs)
        
        # Training parameters
        self.learning_rate = float(kwargs.get('learning_rate', 1e-5))
        self.l1_reg = float(kwargs.get('l1_reg', 0))
        self.weight_decay = float(kwargs.get('weight_decay', 1e-5))
        self.noise_factor = float(kwargs.get('noise_factor', 0))
        self.num_epochs = int(kwargs.get('num_epochs', 50))
        self.batch_size = int(kwargs.get('batch_size', 94))
        self.patience = int(kwargs.get('patience', 0))  # Early stopping patience
        self.kl_weight = int(kwargs.get('kl_weight', 1))  # KL weight for the loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move model to device
    

        self.latent_size = pretrained_vae.latent_size
        self.num_hidden_layers = pretrained_vae.num_hidden_layers
        self.condition_size = condition_size

       # Use the pre-trained encoder and decoder
        self.encoder = pretrained_vae.encoder  # Pre-trained encoder for input x
        self.decoder = pretrained_vae.decoder    # Pre-trained decoder

        # Condition network to produce gamma and beta for FiLM modulation
        self.condition_network = nn.Sequential(
            nn.Linear(self.condition_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * self.latent_size)  # # Output gamma_mu, beta_mu, gamma_log_var, beta_log_var
        )

        # Condition encoder to process condition vector c
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_size, 128),  # Map from 2 to 128 (or latent size)
            nn.ReLU(),
            nn.Linear(128, self.latent_size)  # Output size is latent_size (e.g., 490)
        )


        # Condition decoder to process condition vector c
        self.condition_decoder = nn.Sequential(
            nn.Linear(self.latent_size, 128),  # Map from latent_size (e.g., 490) to 128
            nn.ReLU(),
            nn.Linear(128, self.condition_size)  # Output size should match condition size (2)
        )


        # Layers to compute final mu and log_var
        self.fc_mu = nn.Linear(self.latent_size * 2, self.latent_size)
        self.fc_log_var = nn.Linear(self.latent_size * 2, self.latent_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x, c, c_mask):
        """
        Forward pass for Conditional VAE with FiLM conditioning.
        """
        # Make sure input tensors are moved to the same device as the model
        x = x.to(self.device)
        c = c.to(self.device)
        c_mask = c_mask.to(self.device)
        
        # Encode x with pre-trained encoder
        h_x = self.encoder(x)
        mu_x, log_var_x = h_x.chunk(2, dim=1)

        # Handle missing values in c
        c_masked = c * c_mask  # Apply mask
        c_masked = c_masked.to(x.dtype)

        # Pass c through the condition network to get gamma and beta
        gamma_beta = self.condition_network(c_masked)
        gamma_mu, beta_mu, gamma_log_var, beta_log_var = gamma_beta.chunk(4, dim=1)

        # Apply FiLM modulation to mu_x and log_var_x
        mu = gamma_mu * mu_x + beta_mu
        log_var = gamma_log_var * log_var_x + beta_log_var

        
        # Reparameterization trick
        z = self.reparameterize(mu, log_var)

        # Decode z with pre-trained decoder
        x_recon = self.decoder(z)

        return x_recon, mu, log_var
    
    
    def loss(self, x, x_recon, mu, log_var):
        """
        Loss function for Conditional VAE with FiLM conditioning.
        """
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss



    def forward_c(self, x, c, c_mask):
        """
        Forward pass for Conditional VAE with missing value handling.

        Parameters:
        - x: Input data tensor.
        - c: Condition vector tensor.
        - c_mask: Mask tensor indicating missing values in c (1 for observed, 0 for missing).
        """
        # Encode x with pre-trained encoder
        h_x = self.encoder(x)
        mu_x, log_var_x = h_x.chunk(2, dim=1)

        # Encode c with condition encoder
        # Apply mask to c before encoding
        c_encoded = self.condition_encoder(c * c_mask)  # Element-wise multiplication

        # Combine mu_x and c_encoded to get final mu
        combined = torch.cat([mu_x, c_encoded], dim=1)
        mu = self.fc_mu(combined)
        log_var = log_var_x  # Assume log_var is the same as log_var_x

        # Reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode z with pre-trained decoder
        x_recon = self.decoder(z)

        # Decode c with condition decoder
        c_decoded = self.condition_decoder(c_encoded)  # Apply mask

        return x_recon, mu, log_var, c_decoded


    def loss_c(self, x, x_recon, mu, log_var, c, c_decoded, c_mask):
        """
        Loss function for Conditional VAE. Combines reconstruction loss, KL divergence,
        and a loss for the decoded condition vector c_decoded.
        
        Parameters:
        - x: Original input data.
        - x_recon: Reconstructed input.
        - mu: Mean of the latent space.
        - log_var: Log variance of the latent space.
        - c: Original condition vector.
        - c_decoded: Processed/reconstructed condition vector.
        - c_mask: Mask indicating observed/missing values in the condition vector.
        
        Returns:
        - total_loss: Total loss including reconstruction, KL, and condition losses.
        - recon_loss: Reconstruction loss for x.
        - kl_loss: KL divergence loss.
        - c_loss: Condition vector reconstruction or transformation loss.
        """
        # Reconstruction loss for x
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Condition vector reconstruction or transformation loss
        # You can also use a different loss here depending on the task.
        c_loss = F.mse_loss(c_decoded * c_mask, c * c_mask, reduction='mean')  # Apply mask

        # Total loss: Combine reconstruction loss, KL divergence, and condition loss
        total_loss = recon_loss + kl_loss + c_loss

        return total_loss, recon_loss, kl_loss, c_loss
    
    
    def forward_x(self, x):
        """
        Forward pass for Conditional VAE with missing value handling.

        Parameters:
        - x: Input data tensor.
        """
        
        # Encode x with pre-trained encoder
        h_x = self.encoder(x)
        mu_x, log_var_x = h_x.chunk(2, dim=1)

        # Reparameterization trick
        z_x = self.reparameterize(mu_x, log_var_x)

        # Decode z_combined with pre-trained decoder
        x_recon = self.decoder(z_x)

        return x_recon, mu_x, log_var_x

