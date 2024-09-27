import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from models.models_VAE import VAE



def _init_weights_xavier(layer,gain=1.0):
        if hasattr(layer, '.weight'):
            # nn.init.xavier_normal_(layer.weight.data,gain=gain)
            nn.init.xavier_uniform_(layer.weight.data,gain=gain)
        else:
            if hasattr(layer, 'children'):
                for child in layer.children():
                    _init_weights_xavier(child,gain=gain)



class PretrainVAE(VAE):
    def __init__(self, **kwargs):
        # Extract VAE-specific arguments and initialize the parent VAE class
        vae_kwargs = {
            'input_size': int(kwargs.get('input_size', 1)),
            'latent_size': int(kwargs.get('latent_size', 1)),
            'num_hidden_layers': int(kwargs.get('num_hidden_layers', 1)),
            'dropout_rate': float(kwargs.get('dropout_rate', 0.2)),
            'activation': kwargs.get('activation', 'leakyrelu'),
            'use_batch_norm': kwargs.get('use_batch_norm', False),
            'act_on_latent_layer': kwargs.get('act_on_latent_layer', False),
            'verbose': kwargs.get('verbose', False)
        }
        super().__init__(**vae_kwargs)

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
    

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
    def init_layers(self):
        #self.apply(self._init_weights)  # Initialize weights using a helper function
        _init_weights_xavier(self.encoder,gain=nn.init.calculate_gain(self.activation))
        _init_weights_xavier(self.decoder,gain=nn.init.calculate_gain(self.activation))  
                
    def loss(self, x, x_recon, mu, log_var):

        # Reconstruction loss    
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        #print('recon_loss', recon_loss) 
        
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        #print('kl_loss', kl_loss)
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss 

        # print('kl_loss', kl_loss)
        return recon_loss, kl_loss, total_loss
    