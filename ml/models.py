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

from lifelines.utils import concordance_index

####################################################################################
# Helper functions
def get_model(model_kind, input_size, **kwargs):

    if model_kind == 'NA':
        model = DummyModel()
    elif model_kind == 'VAE':
        model = VAE(input_size = input_size, **kwargs)
    elif model_kind == 'AE':
        model = AE(input_size = input_size, **kwargs)
    elif model_kind == 'TGEM_Encoder':
        model = TGEM_Encoder(input_size = input_size, **kwargs)
    else:
        raise ValueError('model_kind not recognized')
    return model


# def get_model2(model_kind, **kwargs):

#     if model_kind == 'NA':
#         model = Dummy_Head()
#     elif model_kind == 'VAE':
#         model = VAE(input_size = input_size, **kwargs)
#     elif model_kind == 'AE':
#         model = AE(input_size = input_size, **kwargs)
#     elif model_kind == 'TGEM_Encoder':
#         model = TGEM_Encoder(input_size = input_size, **kwargs)
#     elif model_kind == 'TGEM':
#         model = TGEM(input_size = input_size, **kwargs)
#     elif model_kind == 'Binary_Head':
#         model = Binary_Head(input_size = input_size, **kwargs)
#     elif model_kind == 'MultiClass_Head':
#         model = MultiClass_Head(input_size = input_size, **kwargs)
#     else:
#         raise ValueError('model_kind not recognized')
#     return model


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


def _nan_cleaner(y_output, y_true, ignore_nan=True):
        if ignore_nan:
            mask = ~torch.isnan(y_true)
            if mask.sum().item() == 0:
                return None, None
            return y_output[mask], y_true[mask]
        else:
            return y_output, y_true


# create a dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Identity()
        self.goal = 'NA'
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

    def loss(self, y_pred, y_true):
        return torch.tensor(0, dtype=torch.float32)

    def reset_params(self):
        pass

    def get_hyperparameters(self):
        return {'model_kind': 'NA'}

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
            raise ValueError('activation must be one of "leakyrelu", "relu", "tanh", or "sigmoid"')

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

############ Grad Reverse ############

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


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
        # print the unusued kwargs
        for key, value in kwargs.items():
            if key not in ['input_size', 'hidden_size', 'latent_size', 'num_hidden_layers', 'dropout_rate', 'activation', 'use_batch_norm', 'act_on_latent_layer']:
                print(f'Warning: {key} is not a valid argument for VAE')

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

    def loss(self, x, x_recon, mu, log_var):
        # taking the average over the batch helps with stability
        # recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # return torch.div(torch.add(recon_loss, kl_loss), x.size(0))

        # recon_loss2 = F.mse_loss(x_recon, x, reduction='mean')
        # print('recon_loss2', recon_loss2)
        # kl_loss2 = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum() / (mu.size(0) * mu.size(1))
        # print('kl_loss2', kl_loss2)
        # return torch.add(recon_loss, self.kl_weight*kl_loss)
    
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        # print('recon_loss', recon_loss) 
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # print('kl_loss', kl_loss)
        return torch.add(recon_loss, self.kl_weight*kl_loss)
    
    def forward_to_loss(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return self.loss(x, x_recon, mu, log_var)

    def transform_with_loss(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return mu, self.loss(x, x_recon, mu, log_var)

    def generate(self, z):
        return self.decoder(z)
    
    # def reset_weights(self):
    #     self.encoder.apply(self._reset_weights)
    #     self.decoder.apply(self._reset_weights)

    def reset_params(self):
        _reset_params(self)

    def get_file_ids(self):
        return [self.file_id]

    # def get_hyperparameters(self):
    #     return {'input_size': self.input_size,
    #             'hidden_size': self.hidden_size,
    #             'latent_size': self.latent_size,
    #             'num_hidden_layers': self.num_hidden_layers,
    #             'dropout_rate': self.dropout_rate,
    #             'activation': self.activation,
    #             'use_batch_norm': self.use_batch_norm,
    #             'act_on_latent_layer': self.act_on_latent_layer,
    #             'model_kind': 'VAE'}
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
    
    
### Autoencoder
class AE(nn.Module):
    # def __init__(self, input_size, hidden_size, latent_size,
    #              num_hidden_layers=1, dropout_rate=0,
    #              activation='leaky_relu', use_batch_norm=False, act_on_latent_layer=False):
    def __init__(self, **kwargs):
        super(AE, self).__init__()

        input_size = kwargs.get('input_size')
        hidden_size = kwargs.get('hidden_size')
        latent_size = kwargs.get('latent_size')
        num_hidden_layers = kwargs.get('num_hidden_layers', 1)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        activation = kwargs.get('activation', 'leakyrelu')
        use_batch_norm = kwargs.get('use_batch_norm', False)
        act_on_latent_layer = kwargs.get('act_on_latent_layer', False)
        # print the unusued kwargs
        for key, value in kwargs.items():
            if key not in ['input_size', 'hidden_size', 'latent_size', 'num_hidden_layers', 'dropout_rate', 'activation', 'use_batch_norm', 'act_on_latent_layer']:
                print(f'Warning: {key} is not a valid argument for AE')

        self.goal = 'encode'
        self.kind = 'AE'
        self.latent_size = latent_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.file_id = 'AE'
        if activation == 'leakyrelu':
            self.activation = 'leaky_relu'
        else:
            self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.act_on_latent_layer = act_on_latent_layer
        self.encoder = Dense_Layers(input_size=input_size, 
                                    hidden_size=hidden_size, 
                                    output_size=latent_size, 
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

    def init_layers(self):
        # Weight Initialization: Proper weight initialization can help mitigate 
        # the vanishing/exploding gradients problem. 
        # Techniques like Xavier/Glorot or He initialization can be used
        _init_weights_xavier(self.encoder,gain=nn.init.calculate_gain(self.activation))
        _init_weights_xavier(self.decoder,gain=nn.init.calculate_gain(self.activation))  

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def transform(self, x):
        return self.encoder(x)
    
    def loss(self, x, x_recon):
        # taking the average over the batch helps with stability
        # return F.mse_loss(x_recon, x, reduction='sum')
        return F.mse_loss(x_recon, x, reduction='mean')
    
    def forward_to_loss(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return self.loss(x, x_recon)

    def transform_with_loss(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, self.loss(x, x_recon)

    def generate(self, z):
        return self.decoder(z)

    def reset_params(self):
        _reset_params(self)
    
    def get_file_ids(self):
        return [self.file_id]

    # def get_hyperparameters(self):
    #     return {'input_size': self.input_size,
    #             'hidden_size': self.hidden_size,
    #             'latent_size': self.latent_size,
    #             'num_hidden_layers': self.num_hidden_layers,
    #             'dropout_rate': self.dropout_rate,
    #             'activation': self.activation,
    #             'use_batch_norm': self.use_batch_norm,
    #             'act_on_latent_layer': self.act_on_latent_layer,
    #             'model_kind': 'AE'}

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
    


#########################################################
####### Heads #######


class MultiHead(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList([h for h in heads if h.weight > 0])
        self.heads_names = [head.name for head in self.heads]
        self.kind = 'MultiHead'
        self.name = 'MultiHead'
        self.file_id = 'MultiHead'
        self.weight = 1.0

        # check that all heads have unique names
        if len(set(self.heads_names)) != len(self.heads_names):
            raise ValueError('All heads must have unique names')
        
        # check that all heads have the same input size
        input_sizes = [head.input_size for head in self.heads]
        if len(set(input_sizes)) > 1:
            raise ValueError('All heads must have the same input size')
        if len(set(input_sizes)) == 1:
            self.input_size = input_sizes[0]

    def forward(self, x):
        outputs = {f'{head.kind}_{head.name}': head(x) for head in self.heads}
        return outputs

    def predict(self, x):
        outputs = {f'{head.kind}_{head.name}': head.predict(x) for head in self.heads}
        return outputs

    def update_class_weights(self, y_data):
        for head in self.heads:
            if head.goal == 'classify':
                head.update_class_weights(y_data)
        pass

    # def multi_loss(self, outputs, y_true):
    def multi_loss(self, outputs, y_true):
        losses = {f'{head.kind}_{head.name}': head.loss(outputs[f'{head.kind}_{head.name}'], y_true) for head in self.heads}
        return losses
    
    def score(self, outputs, y_true):
        if isinstance(outputs, dict):
            scores = {f'{head.kind}_{head.name}': head.score(outputs[f'{head.kind}_{head.name}'], y_true) for head in self.heads}
        else:
            scores = {f'{head.kind}_{head.name}': head.score(outputs, y_true) for head in self.heads}
        return scores

    def reset_params(self):
        for head in self.heads:
            head.reset_params()

    def joint_loss(self, outputs, y_true):
        losses = self.multi_loss(outputs, y_true)
        joint_loss = sum([head.weight * losses[f'{head.kind}_{head.name}'] for head in self.heads])
        return joint_loss
    
    def loss(self, outputs, y_true):
        return self.multi_loss(outputs, y_true)
    
    def save_state_to_path(self, save_path, save_name=None):
        if save_name is None:
            for head in self.heads:
                head.save_state_to_path(save_path)
        else:
            torch.save(self.state_dict(), os.path.join(save_path, save_name))
        pass

    def load_state_from_path(self, load_path, multihead=False):
        if multihead:
            self.load_state_dict(torch.load(load_path))
        else:
            for head in self.heads:
                head.load_state_from_path(load_path)
        pass

    def get_info(self):
        info = {}
        for head in self.heads:
            info[f'{head.kind}_{head.name}'] = head.get_info()
        return info

    def save_info(self, save_path, save_name=None):
        if save_name is None:
            for head in self.heads:
                head.save_info(save_path)
        else:
            save_json(self.get_info(), os.path.join(save_path, save_name))
        pass

    def get_file_ids(self):
        return [head.file_id for head in self.heads]


class Head(nn.Module):
    def __init__(self,**kwargs):
        super(Head, self).__init__()
        self.goal = 'NA'
        self.kind = 'Head'
        self.name = 'NA'
        self.network = nn.Identity()
        self.y_idx = 0
        self.loss_func = lambda x, y: torch.tensor(0, dtype=torch.float32)
        self.eval_func = lambda x, y: torch.tensor(0, dtype=torch.float32)
        self.weight = 1.0
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', 1)
        self.architecture = {}
        self.score_func_dict = {}
        self.file_id = self.kind + '_' + self.name

    def define_architecture(self,**kwargs):
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', 1)
        pass

    def assign_weight(self, weight):
        self.weight = weight
        pass

    def reset_params(self):
        _reset_params(self)

    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        return self.forward(x)

    # def define_loss(self):
    #     pass

    def loss(self, y_output, y_true):
        if self.weight == 0:
            return torch.tensor(0, dtype=torch.float32)
        return self.loss_func(y_output, y_true)
    
    # def define_score(self):
    #     pass

    def score(self, y_output, y_true):
        return self.score_func(y_output, y_true)

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

    def get_architecture(self):
        return self.architecture
    
    def get_kwargs(self):
        return self.architecture

    def get_file_ids(self):
        return [self.file_id]

    def save_info(self, save_path, save_name=None):
        if save_name is None:
            if os.path.isfile(save_path):
                save_name = os.path.basename(save_path)
                save_path = os.path.dirname(save_path)
            else:
                save_name = self.file_id + '_info.json'
        save_json(self.get_info(), os.path.join(save_path, save_name))                
        pass



#########################################################
class Dummy_Head(Head):
    def __init__(self, name='Dummy', y_idx=0, **kwargs):
        super(Dummy_Head, self).__init__()
        self.goal = 'NA'
        self.kind = 'Dummy'
        self.name = name
        self.y_idx = y_idx
        self.weight = 0.0
        self.network = nn.Identity()

    def update_class_weights(self, y_data):
        pass


#########################################################
    
class Binary_Head(Head):
    def __init__(self, **kwargs):
    # def __init__(self, name='Binary', y_idx=0, weight=1.0, kind='Binary', **kwargs):
        super(Binary_Head, self).__init__()
        self.goal = 'classify'
        self.kind = 'Binary'
        assert self.kind == kwargs.get('kind', 'Binary')
        self.name = kwargs.get('name', 'Binary')
        self.y_idx = kwargs.get('y_idx', 0)
        self.weight = kwargs.get('weight', 1.0)
        self.num_examples_per_class = None
        self.class_weight = None
        self.pos_class_weight = None
        self.architecture = kwargs.get('architecture', kwargs)
        self.loss_reduction = 'mean'
        # self.network = nn.Identity()
        
        self.network = Dense_Layers(**self.architecture)
        self.input_size = self.architecture.get('input_size', 1)
        self.output_size = self.architecture.get('output_size', 1)
        
        self.file_id = self.kind + '_' + self.name

        self.loss_func = nn.BCEWithLogitsLoss(reduction=self.loss_reduction,
                                                    pos_weight=self.pos_class_weight)
        self.score_func_dict = {'AUROC (micro)': lambda y_score, y_true:
                                roc_auc_score(y_true.numpy(), y_score.numpy(), average='micro')}
                                # AUROC(task='binary', average='weighted')}
        #TODO: is the score function aggregrationg results?

    def define_architecture(self, **kwargs):
        self.architecture = kwargs
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', 1)
        self.network = Dense_Layers(**kwargs)
        pass


    def update_class_weights(self, y_data):
        # self.loss_func = nn.BCEWithLogitsLoss(weight=class_weight,
                                                # reduction=reduction)
        # pos weight = # of negative samples / # of positive samples <- this is wrong, we want the reverse
        # class_weight = [1/(# of negative samples), 1/(# of positive samples)]
        # so pos_weight = (1/neg_weight)/(1/pos_weight) = pos_weight/neg_weight
        y_true = y_data[:,self.y_idx]
        # remove nans 
        y_true = y_true[~torch.isnan(y_true)]
        y_int = y_true.int()
        self.num_sample_per_class = torch.bincount(y_int)
        # print(self.num_sample_per_class)
        self.class_weight= 1/torch.bincount(y_int)
        self.pos_class_weight = self.num_sample_per_class[1]/self.num_sample_per_class[0]
        # self.pos_class_weight = self.class_weight[1]/self.class_weight[0]
        pass


    def loss(self, y_logits, y_data, ignore_nan=True):
        if self.weight == 0:
            return torch.tensor(0, dtype=torch.float32)
        y_true = y_data[:,self.y_idx]
        
        y0, y1 = _nan_cleaner(y_logits, y_true, ignore_nan)
        if y0 is None:
            return torch.tensor(0, dtype=torch.float32)
        return self.loss_func(y0.squeeze(), y1.squeeze())

    def logits_to_proba(self, y_logits):
        return F.sigmoid(y_logits)

    def predict_proba(self, x):
        return F.sigmoid(self.network(x))
    
    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) > threshold).float()


    def score(self, y_logits, y_data,ignore_nan=True):
        #TODO: make sure the fix for this issue is more correct
        # if y_data.shape[1] < self.y_idx:
            # return {k: 0 for k, v in self.score_func_dict.items()}
        if self.weight == 0:
            return {k: 0 for k, v in self.score_func_dict.items()}            
        try:
            y_true = y_data[:,self.y_idx]        
            logits, targets = _nan_cleaner(y_logits.detach(), y_true.detach(), ignore_nan)
            if logits is None:
                return torch.tensor(0, dtype=torch.float32)
            probs = self.logits_to_proba(logits)
            return {k: v(probs.squeeze(), targets.squeeze()) for k, v in self.score_func_dict.items()}
            # return {k: v(y0.squeeze(), y1.squeeze()).compute().item() for k, v in self.score_func_dict.items()}
        except IndexError as e:
            print(f'when calculate score get IndexError: {e}')
            traceback.print_exc()
            return {k: 0 for k, v in self.score_func_dict.items()}
#########################################################

class MultiClass_Head(Head):
    def __init__(self, **kwargs):
    # def __init__(self, name='MultiClass', y_idx=0, weight=1.0, num_classes=3, kind='MultiClass', **kwargs):
        super(MultiClass_Head, self).__init__()
        self.goal = 'classify'
        self.kind = 'MultiClass'
        assert self.kind == kwargs.get('kind', 'MultiClass')
        self.name = kwargs.get('name', 'MultiClass')
        self.y_idx = kwargs.get('y_idx', 0)
        self.weight = kwargs.get('weight', 1.0)
        self.num_classes = kwargs.get('num_classes', 3)
        self.architecture = kwargs.get('architecture', kwargs)
        
        if 'output_size' in self.architecture:
            if self.architecture['output_size'] != self.num_classes:
                raise ValueError(f'Output layer has {kwargs["output_size"]} features, but should have {self.num_classes} features')
        else:
            self.architecture['output_size'] = self.num_classes
        
        self.file_id = self.kind + '_' + self.name
        self.input_size = self.architecture.get('input_size', 1)
        self.output_size = self.architecture.get('output_size', self.num_classes)
        self.class_weight = None
        self.loss_reduction = 'mean'
        self.label_smoothing = 0
        # self.network = nn.Identity()
        self.network = Dense_Layers(**self.architecture)
        

        self.loss_func = nn.CrossEntropyLoss(reduction=self.loss_reduction, 
                                             weight=self.class_weight, 
                                             label_smoothing=self.label_smoothing)
        self.score_func_dict = {'AUROC (ovo, macro)': lambda y_score, y_true:
                                roc_auc_score(y_true.numpy(), y_score.numpy(), average='macro', multi_class='ovo'),
                                'ACC': lambda y_score, y_true: accuracy_score(y_true.numpy(), (torch.argmax(y_score, dim=1)).numpy()),}
                                # 'ACC': lambda y_score, y_true: accuracy_score(y_true.numpy(), y_score.numpy()),}
        
        # larger classes are given MORE weight when average="weighted"



    def define_architecture(self, **kwargs):
        if 'output_size' in kwargs:
            if kwargs['output_size'] != self.num_classes:
                raise ValueError(f'Output layer has {kwargs["output_size"]} features, but should have {self.num_classes} features')
        else:
            kwargs['output_size'] = self.num_classes
        
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', self.num_classes)
        self.architecture = kwargs
        self.network = Dense_Layers(**kwargs)
        # check that the output layer has the correct number of classes
        pass

    def update_class_weights(self, y_data):
        y_true = y_data[:,self.y_idx]
        # remove nans 
        y_true = y_true[~torch.isnan(y_true)]
        y_int = y_true.int()
        self.class_weight= 1/torch.bincount(y_int)
        if self.num_classes != len(self.class_weight):
            raise ValueError(f'num_classes {self.num_classes} does not match the number of unique classes {len(self.class_weight)} in the data')
    
        pass

    def loss(self, y_logits, y_data, ignore_nan=True):
        if self.weight == 0:
            return torch.tensor(0, dtype=torch.float32)
        y_true = y_data[:,self.y_idx]
        y0, y1 = _nan_cleaner(y_logits, y_true, ignore_nan)
        if y0 is None:
            return torch.tensor(0, dtype=torch.float32)
        return self.loss_func(y0, y1.long())

    def logits_to_proba(self, y_logits):
        return F.softmax(y_logits, dim=1)
    
    def predict_proba(self, x):
        return F.softmax(self.network(x), dim=1)
    
    def predict(self, x):
        return torch.argmax(self.predict_proba(x), dim=1)
    
    def score(self, y_logits, y_data,ignore_nan=True):
        if self.weight == 0:
            return {k: 0 for k, v in self.score_func_dict.items()}
        
        #TODO: make sure the fix for this issue is more correct
        # if y_data.shape[1] < self.y_idx:
            # return {k: 0 for k, v in self.score_func_dict.items()}
        try:
            y_true = y_data[:,self.y_idx]
            logits, targets = _nan_cleaner(y_logits.detach(), y_true.detach(), ignore_nan)
            if logits is None:
                return torch.tensor(0, dtype=torch.float32)
            probs = self.logits_to_proba(logits)
            return {k: v(probs, targets.long()) for k, v in self.score_func_dict.items()}
        except IndexError as e:
            print(f'when calculate score get IndexError: {e}')
            traceback.print_exc()
            return {k: 0 for k, v in self.score_func_dict.items()}

#########################################################


class Ordinal_Head(Head):
    def __init__(self, **kwargs):
        super(Ordinal_Head, self).__init__()
        self.goal = 'classify'
        self.kind = 'Ordinal'
        assert self.kind == kwargs.get('kind', 'Ordinal')
        self.name = kwargs.get('name', 'Ordinal')
        self.y_idx = kwargs.get('y_idx', 0)
        self.weight = kwargs.get('weight', 1.0)
        self.num_classes = kwargs.get('num_classes', 3)
        self.architecture = kwargs.get('architecture', kwargs)

        if 'output_size' in self.architecture:
            if self.architecture['output_size'] != self.num_classes-1:
                raise ValueError(f'Output layer has {kwargs["output_size"]} features, but should have {self.num_classes} features')
        else:
            self.architecture['output_size'] = self.num_classes-1

        self.loss_reduction = 'mean'
        self.network = Dense_Layers(**self.architecture)
        self.input_size = self.architecture.get('input_size', 1)
        self.output_size = self.architecture.get('output_size', self.num_classes-1)

        # indepdent biases for each class
        self.biases = nn.Parameter(torch.zeros(self.num_classes - 1))

        self.file_id = self.kind + '_' + self.name
        self.loss_func = nn.BCEWithLogitsLoss(reduction=self.loss_reduction)

        # self.score_func_dict = {'AUROC (ovo, macro)': lambda y_score, y_true:
        #                 roc_auc_score(y_true.numpy(), y_score.numpy(), average='macro', multi_class='ovo')}
        self.score_func_dict = {'ACC': lambda y_score, y_true: accuracy_score(y_true.numpy(), y_score.numpy()),}


    def define_architecture(self, **kwargs):
        self.architecture = kwargs
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', self.num_classes-1)
        self.network = Dense_Layers(**kwargs)
        pass

    def update_class_weights(self, y_data):
        #TODO: how to use this for the ordinal loss function?
        y_true = y_data[:,self.y_idx]
        # remove nans 
        y_true = y_true[~torch.isnan(y_true)]
        y_int = y_true.int()
        self.class_weight= 1/torch.bincount(y_int)
        if self.num_classes != len(self.class_weight):
            raise ValueError(f'num_classes {self.num_classes} does not match the number of unique classes {len(self.class_weight)} in the data')
    
        pass

    def forward(self, x):
        output = self.network(x)
        output += self.biases
        return output
    
    def loss(self, y_logits, y_data, ignore_nan=True):
        if self.weight == 0:
            return torch.tensor(0, dtype=torch.float32)
        y_true = y_data[:,self.y_idx]
        y0, y1 = _nan_cleaner(y_logits, y_true, ignore_nan)
        if y0 is None:
            return torch.tensor(0, dtype=torch.float32)
        y1_ordinal = transform_labels_to_binary(y1, self.num_classes)
        return self.loss_func(y0, y1_ordinal)
    
    def logits_to_proba(self, y_logits):
        return F.sigmoid(y_logits)
    
    def predict_proba(self, x):
        return self.logits_to_proba(self.forward(x))
    
    def proba_to_labels(self, probs, threshold=0.5):
        return transform_probs_to_labels(probs, threshold)

    def logits_to_labels(self, y_logits, threshold=0.5):
        return self.proba_to_labels(self.logits_to_proba(y_logits), threshold)

    def predict(self, x, threshold=0.5):
        return self.probs_to_labels(self.predict_proba(x), threshold).float()

    def score(self, y_logits, y_data,ignore_nan=True):
        if self.weight == 0:
            return {k: 0 for k, v in self.score_func_dict.items()}
        
        #TODO: make sure the fix for this issue is more correct
        # if y_data.shape[1] < self.y_idx:
            # return {k: 0 for k, v in self.score_func_dict.items()}
        try:
            y_true = y_data[:,self.y_idx]
            logits, targets = _nan_cleaner(y_logits.detach(), y_true.detach(), ignore_nan)
            if logits is None:
                return torch.tensor(0, dtype=torch.float32)
            probs = self.logits_to_proba(logits)
            labels = self.proba_to_labels(probs)
            return {k: v(labels, targets.long()) for k, v in self.score_func_dict.items()}
        except IndexError as e:
            print(f'when calculate score get IndexError: {e}')
            traceback.print_exc()
            return {k: 0 for k, v in self.score_func_dict.items()}
    

#########################################################


class Regression_Head(Head):
    # def __init__(self, name='Regression', y_idx=0, weight=1.0, kind='Regression', **kwargs):
    def __init__(self, **kwargs):
        super(Regression_Head, self).__init__()
        self.goal = 'regress'
        self.kind = 'Regression'
        assert self.kind == kwargs.get('kind', 'Regression')
        self.name = kwargs.get('name', 'Regression')
        self.y_idx = kwargs.get('y_idx', 0)
        self.weight = kwargs.get('weight', 1.0)
        self.architecture = kwargs.get('architecture', kwargs)
        self.loss_reduction = 'mean'
        self.network = Dense_Layers(**self.architecture)
        self.input_size = self.architecture.get('input_size', 1)
        self.output_size = self.architecture.get('output_size', 1)
        self.file_id = self.kind + '_' + self.name
        self.loss_func = nn.MSELoss(reduction=self.loss_reduction)
        self.score_func_dict = {'MSE': lambda y_score, y_true: F.mse_loss(y_score, y_true),
                                'MAE': lambda y_score, y_true: F.l1_loss(y_score, y_true)}
        # self.network = nn.Identity()
        # self.network = Dense_Layers(**kwargs)

    def define_architecture(self, **kwargs):
        self.architecture = kwargs
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', 1)
        self.network = Dense_Layers(**kwargs)
        pass

    def predict(self, x):
        return self.network(x)

    def loss(self, y_output, y_data, ignore_nan=True):
        if self.weight == 0:
            return torch.tensor(0, dtype=torch.float32)
        y_true = y_data[:,self.y_idx]
        y0, y1 = _nan_cleaner(y_output, y_true, ignore_nan)
        if y0 is None:
            return torch.tensor(0, dtype=torch.float32)
        return self.loss_func(y0.squeeze(), y1.squeeze())

    def score(self, y_output, y_data, ignore_nan=True):
        if self.weight == 0:
            return {k: 0 for k, v in self.score_func_dict.items()}
        try:
            y_true = y_data[:,self.y_idx]
            y0, y1 = _nan_cleaner(y_output.detach(), y_true.detach(), ignore_nan)
            if y0 is None:
                return torch.tensor(0, dtype=torch.float32)
            return {k: v(y0.squeeze(), y1.squeeze()) for k, v in self.score_func_dict.items()}
        except IndexError as e:
            print(f'when calculate score get IndexError: {e}')
            traceback.print_exc()
            return {k: 0 for k, v in self.score_func_dict.items()}


class Cox_Head(Head):
    def __init__(self, **kwargs):
        super(Cox_Head, self).__init__()
        self.goal = 'survival'
        self.kind = 'Cox'
        assert self.kind == kwargs.get('kind', 'Cox')
        self.name = kwargs.get('name', 'Cox')
        self.y_idx = kwargs.get('y_idx', [0,1])
        assert len(self.y_idx) == 2
        self.weight = kwargs.get('weight', 1.0)
        self.architecture = kwargs.get('architecture', kwargs)
        self.loss_reduction = 'mean'
        self.network = Dense_Layers(**self.architecture)
        self.input_size = self.architecture.get('input_size', 1)
        self.output_size = self.architecture.get('output_size', 1)
        self.file_id = self.kind + '_' + self.name
        self.loss_func = CoxPHLoss()
        self.score_func_dict = {'Concordance Index': 
                                lambda y_score, y_true, y_event: concordance_index(y_true, y_score, y_event)}
        # self.network = nn.Identity()
        # self.network = Dense_Layers(**kwargs)

    def define_architecture(self, **kwargs):
        self.architecture = kwargs
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', 1)
        self.network = Dense_Layers(**kwargs)
        pass

    def forward(self, x):
        return self.network(x)

    def loss(self, y_output, y_data, ignore_nan=True):
        if self.weight == 0:
            return torch.tensor(0, dtype=torch.float32)
        y_true = y_data[:,self.y_idx[0]]
        y_event = y_data[:,self.y_idx[1]]
        
        if ignore_nan:
            mask = ~torch.isnan(y_true)
            if mask.sum().item() == 0:
                return torch.tensor(0, dtype=torch.float32)
            y_true = y_true[mask]
            y_event = y_event[mask]
            y_output = y_output[mask]
        # return self.loss_func(y_output.squeeze(), y_true.squeeze(), y_event.squeeze())
        return self.loss_func(y_output, y_true, y_event)

    def predict_risk(self, x):
        # which is the correct output layer?
        # return F.sigmoid(self.network(x))
        return torch.exp(self.network(x))

    def predict(self, x):
        # return self.predict_risk(x)
        return self.network(x)

    def score(self, y_output, y_data, ignore_nan=True):
        if self.weight == 0:
            return {k: 0 for k, v in self.score_func_dict.items()}
        try:
            y_true = y_data[:,self.y_idx[0]]
            y_event = y_data[:,self.y_idx[1]]
            mask = ~torch.isnan(y_true)
            if mask.sum().item() == 0:
                return {k: 0 for k, v in self.score_func_dict.items()}
            y0 = -1*torch.exp(y_output.detach()[mask]) 
            y1 = y_true.detach()[mask]
            y2 = y_event.detach()[mask]

            return {k: v(y0.squeeze(), y1.squeeze(), y2.squeeze()) for k, v in self.score_func_dict.items()}
        except IndexError as e:
            print(f'when calculate score get IndexError: {e}')
            traceback.print_exc()
            return {k: 0 for k, v in self.score_func_dict.items()}


class Decoder_Head(Head):
    def __init__(self, name='Decoder', y_idx=0, weight=1.0, kind='Decoder', **kwargs):
        super(Decoder_Head, self).__init__()
        self.goal = 'decode'
        self.kind = 'Decoder'
        assert self.kind == kind
        self.name = name
        self.y_idx = y_idx
        self.weight = weight
        self.architecture = kwargs
        self.loss_reduction = 'mean'
        self.network = Dense_Layers(**kwargs)
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', 1)
        self.file_id = self.kind + '_' + self.name
        self.loss_func = nn.MSELoss(reduction=self.loss_reduction)
        self.score_func_dict = {'MSE': lambda y_score, y_true: F.mse_loss(y_score, y_true),
                                'MAE': lambda y_score, y_true: F.l1_loss(y_score, y_true)}
        # self.network = nn.Identity()
        # self.network = Dense_Layers(**kwargs)

    def define_architecture(self, **kwargs):
        self.architecture = kwargs
        self.input_size = kwargs.get('input_size', 1)
        self.output_size = kwargs.get('output_size', 1)
        self.network = Dense_Layers(**kwargs)
        pass

    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        return self.network(x)

    def loss(self, y_output, y_data, ignore_nan=True):
        if self.weight == 0:
            return torch.tensor(0, dtype=torch.float32)
        y_true = y_data[:,self.y_idx]
        y0, y1 = _nan_cleaner(y_output, y_true, ignore_nan)
        if y0 is None:
            return torch.tensor(0, dtype=torch.float32)
        return self.loss_func(y0.squeeze(), y1.squeeze())

    def score(self, y_output, y_data, ignore_nan=True):
        if self.weight == 0:
            return {k: 0 for k, v in self.score_func_dict.items()}
        try:
            y_true = y_data[:,self.y_idx]
            y0, y1 = _nan_cleaner(y_output.detach(), y_true.detach(), ignore_nan)
            if y0 is None:
                return torch.tensor(0, dtype=torch.float32)
            return {k: v(y0.squeeze(), y1.squeeze()) for k,
                    v in self.score_func_dict.items()}
        except IndexError as e:
            print(f'when calculate score get IndexError: {e}')
            traceback.print_exc()
            return {k: 0 for k, v in self.score_func_dict.items()}

#########################################################
#########################################################

def get_encoder(**kwargs):
    kind = kwargs.get('kind','UNKOWN')
    if kind == 'AE':
        encoder = AE(**kwargs)
    elif kind == 'VAE':
        encoder = VAE(**kwargs)
    else:
        raise ValueError(f'Encoder kind {kind} not recognized')
    return encoder


def get_head(**kwargs):
    kind = kwargs.get('kind','UNKOWN')
    if kind == 'NA' or kind == 'Dummy':
        head = Dummy_Head()
    elif kind == 'Binary':
        head = Binary_Head(**kwargs)
    elif kind == 'MultiClass':
        head = MultiClass_Head(**kwargs)
    # elif kind == 'MultiHead':
    #     head = MultiHead(**kwargs)
    elif kind == 'Regression':
        head = Regression_Head(**kwargs)
    elif kind == 'Cox':
        head = Cox_Head(**kwargs)
    elif kind == 'Ordinal':
        head = Ordinal_Head(**kwargs)
    elif kind == 'Decoder':
        raise NotImplementedError('Decoder head not tested')
        head = Decoder_Head(**kwargs)
    else:
        raise ValueError(f'Head kind {kind} not recognized')
    return head


def create_model_wrapper(model_info_path, model_state_path=None, is_encoder=True):
    model_info = load_json(model_info_path)
    
    if is_encoder:
        model = get_encoder(
            **model_info)
    else:
        if 'kind' in model_info:
            if model_info['kind'] == 'MultiHead':
                model_list = []
                for key in model_info.keys():
                    model_list.append(get_head(
                        **model_info[key]))
                model = MultiHead(model_list)
            else:
                model = get_head(
                    **model_info)
        else:
            model_list = []
            for key in model_info.keys():
                model_list.append(get_head(
                    **model_info[key]))
            model = MultiHead(model_list)

    if model_state_path is not None:
        model.load_state_dict(torch.load(model_state_path))
    return model




class CompoundModel(nn.Module):
    def __init__(self, encoder, head):
        super(CompoundModel, self).__init__()
        self.encoder = encoder
        self.head = head
        self.file_id = self.encoder.file_id + '__' +self.head.file_id
    #TODO: check that output of encoder matches input of head, accounting for other_vars

        self.other_dim = self.head.input_size - self.encoder.latent_size

    def concat_other_vars(self, x, other_vars=None):
        if (self.other_dim == 0) and (other_vars is not None):
            print('Warning: other_vars provided but not used')
            return x
        elif (self.other_dim > 0) and (other_vars is not None):
            assert other_vars.shape[1] == self.other_dim
            other_vars = other_vars.clone().detach().requires_grad_(True)
            return torch.cat((x, other_vars), 1)
        elif (self.other_dim > 0) and (other_vars is None):
            return torch.cat((x, torch.zeros(x.shape[0], self.other_dim)), 1)
        else:
            return x

    def forward(self, x, other_vars=None):
        z = self.encoder.transform(x)
        return self.head(self.concat_other_vars(z, other_vars))
    
    def predict(self, x, other_vars=None):
        z = self.encoder.transform(x)
        return self.head.predict(self.concat_other_vars(z, other_vars))

    def score(self,y_output,y):
        if hasattr(self.head,'score'):
            return self.head.score(y_output,y)
        else:
            raise NotImplementedError('score method not implemented for this model')

    def get_info(self):
        model_info = {}
        model_info['encoder'] = self.encoder.get_info()
        model_info['head'] = self.head.get_info()
        return model_info
    
    def save_info(self, save_path, save_name=None):
        if save_name is None:
            if os.path.isfile(save_path):
                save_name = os.path.basename(save_path)
                save_path = os.path.dirname(save_path)
            else:
                save_name = self.file_id + '_info.json'
        save_json(self.get_info(), os.path.join(save_path, save_name))

    def save_state_to_path(self, save_path, save_name=None):
        if save_name is None:
            if os.path.isfile(save_path):
                save_name = os.path.basename(save_path)
                save_path = os.path.dirname(save_path)
            else:
                save_name = self.file_id + '_state.pt'
        torch.save(self.state_dict(), os.path.join(save_path, save_name))
        pass

def create_compound_model_from_info(
        model_info=None,
        encoder_info=None,
        head_info=None,
        model_state_dict = None,
        encoder_state_dict=None,
        head_state_dict=None):
    
    if (model_info is not None):
        if 'encoder' in model_info:
            encoder_info = model_info['encoder']
        if 'head' in model_info:
            head_info = model_info['head']

    encoder = get_encoder(
        **encoder_info
    )

    head = get_head(
        **head_info
    )
    if hasattr(head,'y_idx'):
        if max(head.y_idx) > len(head.y_idx)+1:
            print('Warning the head y_idx probably needs to be fixed')

    if model_state_dict is not None:
        # if head.kind != 'MultiHead':
            # head = MultiHead([head])
        model = CompoundModel(encoder,head)
        model.load_state_dict(model_state_dict)
    else:
        if encoder_state_dict is not None:
            encoder.load_state_dict(encoder_state_dict)
        if head_state_dict is not None:
            head.load_state_dict(head_state_dict)

        # if head.kind != 'MultiHead':
            # head = MultiHead([head])

        model = CompoundModel(encoder,head)

    return model

#########################################################
#########################################################
# Pytorch Sklearn Model


class PytorchModel(BaseEstimator):

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def fit(self, X, y, other_vars=None, **kwargs):
        raise NotImplementedError('fit method not implemented')
        pass

    def predict(self, X, other_vars=None,batch_size=64,use_predict=True):
        if other_vars is None:
            other_vars = torch.zeros(X.shape[0], 1)
        else:
            other_vars = torch.tensor(other_vars, dtype=torch.float32)
        X = torch.tensor(X, dtype=torch.float32)
        y_outputs = None
        self.model.eval()
        with torch.inference_mode():
    
            for i in range(0, X.shape[0], batch_size):
                if (use_predict) and hasattr(self.model,'predict'):
                    y_out = self.model.predict(X[i:i+batch_size], other_vars[i:i+batch_size])
                else:
                    y_out = self.model.forward(X[i:i+batch_size], other_vars[i:i+batch_size])
                
                if isinstance(y_out, dict):
                    if y_outputs is None:
                        y_outputs = {}
                        for k in y_out.keys():
                            y_outputs[k] = y_out[k].detach().numpy()
                    else:
                        for k in y_out.keys():
                            y_outputs[k] = np.concatenate(
                                (y_outputs[k], y_out[k].detach().numpy()), axis=0)
                else:
                    if y_outputs is None:
                        y_outputs = y_out.detach().numpy()
                    else:
                        y_outputs = np.concatenate(
                            (y_outputs, y_out.detach().numpy()), axis=0)

        return y_outputs

    def score(self, X, y, other_vars=None, batch_size=64):
        # check if the score method is an attribute of model
        if hasattr(self.model, 'score'):
            y_outputs = self.predict(X,
                                     other_vars=other_vars,
                                     batch_size=batch_size,
                                     use_predict=False)
            if isinstance(y_outputs, dict):
                y_outputs = {k: torch.tensor(v) for k, v in y_outputs.items()}
                return self.model.score(y_outputs, torch.tensor(y))
            else:
                return self.model.score(torch.tensor(y_outputs),torch.tensor(y))
        else:
            raise NotImplementedError('score method not implemented')
        


def create_pytorch_model_from_info(
        model_info=None,
        model_state=None,
        full_model=None):
    if full_model is not None:
        return PytorchModel(full_model)
    
    model = create_compound_model_from_info(
        model_info=model_info,
        model_state_dict=model_state
    )
    return PytorchModel(model)


#########################################################
#########################################################
    # OLDER HEADS
#########################################################
#########################################################



#########################################################
############ Classification Models ############


### Cox Proportional Hazards Neural Network Model
# TODO: not sure if this is the correct implementation
# class CoxNN(nn.Module):
#     def __init__(self, input_size, hidden_size, 
#                  num_hidden_layers=1, dropout_rate=0.2,
#                  activation='leakyrelu', use_batch_norm=False):
#         super(CoxNN, self).__init__()
#         self.goal = 'regress'
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.dropout_rate = dropout_rate
#         self.activation = activation
#         self.network = Dense_Layers(input_size, hidden_size,
#                                     output_size=1, 
#                                     num_hidden_layers=num_hidden_layers,
#                                     dropout_rate=dropout_rate,
#                                     activation=activation,
#                                     use_batch_norm=use_batch_norm)
        
#         self.loss_func = CoxPHLoss()
    
#     def forward(self, x):
#         return self.network(x)
    
#     def loss(self, y_output, durations, events, ignore_nan=False):
#         if ignore_nan:
#             mask = ~torch.isnan(durations)
#             if mask.sum().item() == 0:
#                 return torch.tensor(0, dtype=torch.float32)
#             return self.loss_func(y_output[mask], durations[mask], events[mask])
#         else:
#             return self.loss_func(y_output, durations, events)
    
#     def predict_risk(self, x):
#         # which is the correct output layer?
#         # return F.sigmoid(self.network(x))
#         return torch.exp(self.network(x))
    
#     def reset_params(self):
#         _reset_params(self)

#     def get_hyperparameters(self):
#         return {'input_size': self.input_size,
#                 'hidden_size': self.hidden_size,
#                 'num_hidden_layers': self.num_hidden_layers,
#                 'dropout_rate': self.dropout_rate,
#                 'activation': self.activation,
#                 'use_batch_norm': self.use_batch_norm}
    
#########################################################
### TGEM Models ###
# The batch size isn't used for anything?!
        
class mulitiattention(torch.nn.Module):
    def __init__(self, n_head, input_size, query_gene, mode=0):
        super(mulitiattention, self).__init__()
        self.n_head = n_head
        self.input_size = input_size
        self.mode = mode
        self.query_gene = query_gene
        self.save_memory = True

        self.WQ = nn.Parameter(torch.Tensor(self.n_head, input_size, 1), requires_grad=True)
        self.WK = nn.Parameter(torch.Tensor(self.n_head, input_size, 1), requires_grad=True)
        self.WV = nn.Parameter(torch.Tensor(self.n_head, input_size, 1), requires_grad=True)
        torch.nn.init.xavier_normal_(self.WQ, gain=1)
        torch.nn.init.xavier_normal_(self.WK, gain=1)

        torch.nn.init.xavier_normal_(self.WV)
        self.W_0 = nn.Parameter(torch.Tensor(self.n_head * [0.001]), requires_grad=True)
        print('init')
        # gpu_tracker.track()

    def QK_diff(self, Q_seq, K_seq):
        QK_dif = -1 * torch.pow((Q_seq - K_seq), 2)
        return torch.nn.Softmax(dim=2)(QK_dif)

    def mask_softmax_self(self, x):
        d = x.shape[1]
        x = x * ((1 - torch.eye(d, d))) #.to(device))
        return x

    def attention(self, x, Q_seq, WK, WV):
        if self.mode == 0:
            K_seq = x * WK
            K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.input_size)
            K_seq = K_seq.permute(0, 2, 1)
            V_seq = x * WV
            QK_product = Q_seq * K_seq
            z = torch.nn.Softmax(dim=2)(QK_product)

            z = self.mask_softmax_self(z)
            out_seq = torch.matmul(z, V_seq)

        ############this part is not working well
        if self.mode == 1:
            zz_list = []
            for q in range(self.input_size // self.query_gene):
                # gpu_tracker.track()
                K_seq = x * WK
                V_seq = x * WV
                Q_seq_x = x[:, (q * self.query_gene):((q + 1) * self.query_gene), :]
                Q_seq = Q_seq_x.expand(Q_seq_x.shape[0], Q_seq_x.shape[1], self.input_size)
                K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.query_gene)
                K_seq = K_seq.permute(0, 2, 1)

                QK_diff = self.QK_diff(Q_seq, K_seq)
                z = torch.nn.Softmax(dim=2)(QK_diff)
                z = torch.matmul(z, V_seq)
                zz_list.append(z)
            out_seq = torch.cat(zz_list, dim=1)
            ####################################
        return out_seq

    def forward(self, x):

        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        out_h = []
        for h in range(self.n_head):
            Q_seq = x * self.WQ[h, :, :]
            Q_seq = Q_seq.expand(Q_seq.shape[0], Q_seq.shape[1], self.input_size)
            if self.save_memory:
                attention_out = cp(self.attention, x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])
            else:
                attention_out = self.attention(x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])

            out_h.append(attention_out)
        out_seq = torch.cat(out_h, dim=2)
        out_seq = torch.matmul(out_seq, self.W_0)
        return out_seq


class layernorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(layernorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class res_connect(nn.Module):

    ##########    A residual connection followed by a layer norm.

    def __init__(self, size, dropout):
        super(res_connect, self).__init__()
        self.norm = layernorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, out):
        ###Apply residual connection to any sublayer with the same size
        return x + self.norm(self.dropout(out))




class TGEM_Encoder(torch.nn.Module):
    def __init__(self, input_size,  n_head=5, dropout_rate=0.3, activation='linear', 
                 query_gene=64, d_ff=1024, mode=0, n_layers=3):
        super(TGEM_Encoder, self).__init__()
        self.goal = 'encode'
        self.kind = 'TGEM_Encoder'
        self.n_head = n_head
        self.input_size = input_size
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.n_layers = n_layers
        ##########   ##########
        # mode 1 is not working right now.
        self.mode = mode
        self.query_gene = query_gene #in original version, this was not a object parameter 
        # query gene is NOT used when mode=0. It is used when mode=1. 
        # But according to the code, mode 1 isn't even working
        if mode == 1:
            raise NotImplementedError('mode 1 is not working right now.')
        ##########   ##########

        if n_layers > 3:
            raise NotImplementedError('n_layers must be 3 or less for TGEM_Encoder')

        if self.activation == 'relu':
            self.activation_func = torch.nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.activation_func = torch.nn.LeakyReLU(0.1)
        elif self.activation == 'gelu':
            self.activation_func = torch.nn.GELU()
        elif self.activation == 'linear':
            self.activation_func = torch.nn.Identity()
        elif self.activation == 'tanh':
            self.activation_func = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.activation_func = torch.nn.Sigmoid()
        else:
            raise ValueError('{} is not a valid activation function'.format(self.act_fun))


        self.activation_func 
        self.mulitiattention1 = mulitiattention( self.n_head, self.input_size, query_gene,
                                                mode)
        self.mulitiattention2 = mulitiattention( self.n_head, self.input_size, query_gene,
                                                mode)
        self.mulitiattention3 = mulitiattention( self.n_head, self.input_size, query_gene,
                                                mode)
        # self.ffn1 = nn.Linear(self.input_size, self.d_ff)
        # self.ffn2 = nn.Linear(self.d_ff, self.input_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sublayer = res_connect(input_size, dropout_rate)
    

    def init_layers(self):
        pass

    def reset_params(self):
        _reset_params(self)

    # def feedforward(self, x):
    #     out = F.relu(self.ffn1(x))
    #     out = self.ffn2(self.dropout(out))
    #     return out

    def forward(self, x):

        out_attn = self.mulitiattention1(x)
        out_attn_1 = self.sublayer(x, out_attn)
        
        if self.n_layers < 2:
            return self.activation_func(out_attn_1)
        
        out_attn_2 = self.mulitiattention2(out_attn_1)
        out_attn_2 = self.sublayer(out_attn_1, out_attn_2)
        
        if self.n_layers < 3:
            return self.activation_func(out_attn_2)
        
        out_attn_3 = self.mulitiattention3(out_attn_2)
        out_attn_3 = self.sublayer(out_attn_2, out_attn_3)
        if self.n_layers < 4:
            return self.activation_func(out_attn_3)
        
        return self.activation_func(out_attn_3)
    
    def transform(self, x):
        return self.forward(x)
    
    def generate(self, x):
        # return a zero tensor the same size as x
        return torch.zeros_like(x)
    
    def loss(self, x, x_recon):
        return torch.tensor(0, dtype=torch.float32)
    
    def forward_to_loss(self, x):
        return torch.tensor(0, dtype=torch.float32)

    def transform_with_loss(self, x):
        return self.forward(x), torch.tensor(0, dtype=torch.float32)

    def get_hyperparameters(self):
        return {
                'n_head': self.n_head,
                'input_size': self.input_size,
                'query_gene': self.query_gene,
                'd_ff': self.d_ff,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation}        



    

#########################################################
#########################################################
## Custom Loss Functions
#########################################################

def transform_labels_to_binary(y, num_classes):    
    '''
    transform the original class labels into a set of C-1
    binary labels, where each binary label indicates whether the 
    sample belongs to a certain ordinal group or not. 
    This transformation allows us to encode the ordinal relationships between the classes
    '''
    transformed_labels = torch.zeros(y.shape[0], num_classes-1)
    
    for idx, label in enumerate(y):
        transformed_labels[idx, 0:int(label)] = 1
        
    return transformed_labels

def transform_probs_to_labels(y_probs,thresh=0.5):
    # inverse of the "transform_labels_to_binary", applied to y_probs produced from the ordinal model
    y_binary = (y_probs>thresh).to(int)
    y_labels = torch.sum(y_binary, dim=1)
    return y_labels


def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return cox_ph_loss(log_h, durations, events)

# %%

