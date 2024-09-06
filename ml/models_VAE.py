# %%
import torch
from torch.autograd import Function
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp
import os
from misc import save_json, load_json
import numpy as np

# from torchmetrics import AUROC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.base import BaseEstimator
import traceback
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

from lifelines.utils import concordance_index

####################################################################################
# Helper functions




def _reset_params(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                _reset_params(child)



def _init_weights_xavier(layer,gain=1.0):
    if hasattr(layer, '.weight'):
        # nn.init.xavier_normal_(layer.weight.data,gain=gain)
        nn.init.xavier_uniform_(layer.weight.data,gain=gain)
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                _init_weights_xavier(child,gain=gain)

def get_reg_penalty_slow(model,l1_lambda,l2_lambda):
    #this is probably slower, but more memory efficient 
    reg_loss = torch.tensor(0, dtype=torch.float32)
    for param in model.parameters():
        reg_loss += l1_lambda * torch.sum(torch.abs(param)) + l2_lambda * torch.sum(param**2)
    return reg_loss

def get_reg_penalty(model,l1_lambda,l2_lambda):
    if (l1_lambda == 0) and (l2_lambda == 0):
        return torch.tensor(0, dtype=torch.float32)
    all_params = torch.cat([x.view(-1) for x in model.parameters()])
    if l1_lambda == 0:
        return l2_lambda * torch.sum(all_params.pow(2))
    elif l2_lambda == 0:
        return l1_lambda * torch.sum(all_params.abs())
    return l1_lambda * torch.sum(all_params.abs()) + l2_lambda * torch.sum(all_params.pow(2))


def _nan_cleaner(y_output, y_true, ignore_nan=True, label_type='class'):
        if ignore_nan:
            if label_type == 'continuous':
                mask = ~torch.isnan(y_true) 
            elif label_type == 'class':
                mask = ~torch.isnan(y_true) & (y_true != -1)
            if mask.sum().item() == 0:
                return None, None
            return y_output[mask], y_true[mask]
        else:
            return y_output, y_true


#########################################################
### Basic Multi-layer Perceptron
class Dense_Layers(nn.Module):
    def __init__(self,**kwargs):
    # def __init__(self, input_size, hidden_size, output_size, 
    #     num_hidden_layers=1, dropout_rate=0.2, activation='leakyrelu',
    #     use_batch_norm=False, act_on_output_layer=False):
        super(Dense_Layers, self).__init__()

        input_size = kwargs.get('input_size', 1)
        hidden_size = kwargs.get('hidden_size', 1)
        output_size = kwargs.get('output_size', 1)
        num_hidden_layers = kwargs.get('num_hidden_layers', 1)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        activation = kwargs.get('activation', 'leakyrelu')
        use_batch_norm = kwargs.get('use_batch_norm', False)
        act_on_output_layer = kwargs.get('act_on_output_layer', False)
        verbose = kwargs.get('verbose', False)
        
        # print the unusued kwargs
        for key, value in kwargs.items():
            if key not in ['input_size', 'hidden_size', 'output_size', 'num_hidden_layers', 'dropout_rate', 'activation', 'use_batch_norm', 'act_on_output_layer','verbose']:
                if verbose: print(f'Warning: {key} is not a valid argument for Dense_Layers')

        if activation == 'leakyrelu' or activation == 'leaky_relu':
            activation_func = nn.LeakyReLU()
        elif activation == 'relu':
            activation_func = nn.ReLU()
        elif activation == 'tanh':
            activation_func = nn.Tanh()
        elif activation == 'sigmoid':
            activation_func = nn.Sigmoid()
        elif activation == 'elu':
            activation_func = nn.ELU()
        else:
            raise ValueError('activation must be one of "leakyrelu", "relu", "tanh", or "sigmoid", user gave: {}'.format(activation))

        if num_hidden_layers < 1:
            # raise ValueError('num_hidden_layers must be at least 1')
            self.network = nn.Sequential(
                nn.Linear(input_size, output_size))
        else:
            # this if else statement is a little hacky, but its needed for backwards compatibility
            if use_batch_norm:
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    activation_func,
                    nn.Dropout(dropout_rate)
                )
                for _ in range(num_hidden_layers-1):
                    self.network.add_module('hidden_layer', nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        activation_func,
                        nn.Dropout(dropout_rate)
                    ))
                self.network.add_module('output_layer', nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                ))


            else:
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    activation_func,
                    nn.Dropout(dropout_rate)
                )
                for _ in range(num_hidden_layers-1):
                    self.network.add_module('hidden_layer', nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        activation_func,
                        nn.Dropout(dropout_rate)
                    ))
                self.network.add_module('output_layer', nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                ))
            
        if act_on_output_layer:
            self.network.add_module('output_activation', activation_func)
            self.network.add_module('output_dropout', nn.Dropout(dropout_rate))


    def forward(self, x):
        return self.network(x)





############ Autoencoder Models ############



### Variational Autoencoder
class VAE(nn.Module):
    # def __init__(self, input_size, hidden_size, latent_size,
    #              num_hidden_layers=1, dropout_rate=0,
    #              activation='leaky_relu', use_batch_norm=False, act_on_latent_layer=False):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()

        input_size = kwargs.get('input_size', 1)
        hidden_size = kwargs.get('hidden_size', 1)
        latent_size = kwargs.get('latent_size', 1)
        num_hidden_layers = kwargs.get('num_hidden_layers', 1)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        activation = kwargs.get('activation', 'leakyrelu')
        use_batch_norm = kwargs.get('use_batch_norm', False)
        act_on_latent_layer = kwargs.get('act_on_latent_layer', False)
        verbose = kwargs.get('verbose', False)
        # print the unusued kwargs
        for key, value in kwargs.items():
            if key not in ['input_size', 'hidden_size', 'latent_size', 'num_hidden_layers', 'dropout_rate', 'activation', 'use_batch_norm', 'act_on_latent_layer']:
                if verbose: print(f'Warning: {key} is not a valid argument for VAE')

        self.goal = 'encode'
        self.kind = 'VAE'
        self.file_id = 'VAE'
        self.latent_size = latent_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        if activation == 'leakyrelu':
            self.activation = 'leaky_relu'
        else:
            self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.act_on_latent_layer = act_on_latent_layer
        # self.kl_weight = 1.0/np.sqrt(num_hidden_layers)
        self.kl_weight = 1.0
        self.latent_weight = 0.5

        self.encoder = Dense_Layers(input_size=input_size, 
                                    hidden_size=hidden_size, 
                                    output_size=2*latent_size, 
                                    num_hidden_layers=num_hidden_layers,
                                    dropout_rate = dropout_rate,
                                    activation = activation,
                                    use_batch_norm=use_batch_norm, 
                                    act_on_output_layer=act_on_latent_layer)
        
        self.decoder = Dense_Layers(input_size=latent_size, 
                                    hidden_size = hidden_size, 
                                    output_size = input_size,
                                    num_hidden_layers = num_hidden_layers, 
                                    dropout_rate = dropout_rate,
                                    activation = activation,
                                    use_batch_norm = use_batch_norm)


        # self.encoder = Dense_Layers(input_size, hidden_size, 2*latent_size, 
        #                             num_hidden_layers, dropout_rate, activation,
        #                             use_batch_norm,act_on_output_layer=act_on_latent_layer)
        
        # self.decoder = Dense_Layers(latent_size, hidden_size, input_size,
        #                             num_hidden_layers, dropout_rate, activation,
        #                             use_batch_norm)

    def init_layers(self):
        # Weight Initialization: Proper weight initialization can help mitigate 
        # the vanishing/exploding gradients problem. 
        # Techniques like Xavier/Glorot or He initialization can be used
        _init_weights_xavier(self.encoder,gain=nn.init.calculate_gain(self.activation))
        _init_weights_xavier(self.decoder,gain=nn.init.calculate_gain(self.activation))  

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def transform(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        return mu
    
    def transform_2(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        return log_var


    def loss(self, x, x_recon, mu, log_var):

        # Reconstruction loss    
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        print('recon_loss', recon_loss) 
        
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        print('kl_loss', kl_loss)
        
        # Latent size penalty
        latent_size_penalty = (self.latent_size * self.latent_weight)
    
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss 
        total_loss_with_penalty = torch.add(total_loss, latent_size_penalty)

        # print('kl_loss', kl_loss)
        return total_loss_with_penalty
    

    
    def forward_to_loss(self, x,y=None):
        if y is None:
            y = x
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return self.loss(y, x_recon, mu, log_var)

    def transform_with_loss(self, x, y=None):
        if y is None:
            y = x
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return mu, self.loss(y, x_recon, mu, log_var)

    def generate(self, z):
        return self.decoder(z)
    
    # def reset_weights(self):
    #     self.encoder.apply(self._reset_weights)
    #     self.decoder.apply(self._reset_weights)

    def score(self, x, y=None):
        if y is None:
            y = x
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return {'Reconstruction MSE' : F.mse_loss(x_recon, y, reduction='mean').item()}

    def reset_params(self):
        _reset_params(self)

    def get_file_ids(self):
        return [self.file_id]


    def save_state_to_path(self, save_path, save_name=None):
        if save_name is None:
            if os.path.isfile(save_path):
                save_name = os.path.basename(save_path)
                save_path = os.path.dirname(save_path)
            else:
                save_name = self.file_id + '_state.pt'
        torch.save(self.state_dict(), os.path.join(save_path, save_name))
        pass

    def load_state_from_path(self, load_path, load_name=None):
        if load_name is None:
            # check if the load_path is to a file
            if os.path.isfile(load_path):
                load_name = os.path.basename(load_path)
                load_path = os.path.dirname(load_path)
            else:
                load_name = self.file_id + '_state.pt'
        self.load_state_dict(torch.load(os.path.join(load_path, load_name)))
        pass


    def get_hyperparameters(self):
        # cycle through all the attributes of the class and save them
        hyperparameters = {}
        for key, value in self.__dict__.items():
            # skip if it is a private attribute
            if key[0] == '_':
                continue
            
            # skip if it is a method
            if callable(value):
                continue
            hyperparameters[key] = value
        return hyperparameters

    def get_info(self):
        return self.get_hyperparameters()

    def save_info(self, save_path, save_name=None):
        # save_json(self.get_hyperparameters(), save_path)
        # pass
        if save_name is None:
            if os.path.isfile(save_path):
                save_name = os.path.basename(save_path)
                save_path = os.path.dirname(save_path)
            else:
                save_name = self.file_id + '_info.json'
        save_json(self.get_info(), os.path.join(save_path, save_name))   

