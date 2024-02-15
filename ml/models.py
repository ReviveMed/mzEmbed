# %%
import torch
from torch import nn
import torch.nn.functional as F
from loss import CoxPHLoss

#TODO add option for batch normalization
#TODO add option for different activation functions

### Basic Multi-layer Perceptron
class Dense_Layers(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
        num_hidden_layers=1, dropout_rate=0.2, activation='leakyrelu',
        use_batch_norm=False, act_on_output_layer=False):
        super(Dense_Layers, self).__init__()

        if activation == 'leakyrelu':
            activation_func = nn.LeakyReLU()
        elif activation == 'relu':
            activation_func = nn.ReLU()
        elif activation == 'tanh':
            activation_func = nn.Tanh()
        elif activation == 'sigmoid':
            activation_func = nn.Sigmoid()
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


## Basic Multi-layer Perceptron
# class Dense_Layers(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, 
#         num_hidden_layers=1, dropout_rate=0.2, activation='leakyrelu',
#         use_batch_norm=False):
#         super(Dense_Layers, self).__init__()

#         if activation == 'leakyrelu':
#             activation_func = nn.LeakyReLU()
#         elif activation == 'relu':
#             activation_func = nn.ReLU()
#         elif activation == 'tanh':
#             activation_func = nn.Tanh()
#         elif activation == 'sigmoid':
#             activation_func = nn.Sigmoid()
#         else:
#             raise ValueError('activation must be one of "leakyrelu", "relu", "tanh", or "sigmoid"')

#         if num_hidden_layers < 1:
#             raise ValueError('num_hidden_layers must be at least 1')

#         # Define the input layer
#         self.input_layer =  nn.Sequential(nn.Linear(input_size, hidden_size))
#         if use_batch_norm:
#             self.input_layer.add_module('batch_norm', nn.BatchNorm1d(hidden_size))
#         self.input_layer.add_module('activation', activation_func)
#         self.input_layer.add_module('dropout', nn.Dropout(dropout_rate))

#         # define the hidden layers
#         self.hidden_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size))
#         if use_batch_norm:
#             self.hidden_layer.add_module('batch_norm', nn.BatchNorm1d(hidden_size))
#         self.hidden_layer.add_module('activation', activation_func)
#         self.hidden_layer.add_module('dropout', nn.Dropout(dropout_rate))

#         # define the output layer
#         self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size))

#         # self.network = nn.Sequential()
#         # self.network.add_module('input_layer', self.input_layer)
#         self.network = nn.Sequential(self.input_layer)
#         for i in range(num_hidden_layers-1):
#             # self.network.add_module(f'hidden_layer_{i}', self.hidden_layer)
#             self.network.add_module(f'hidden_layer', self.hidden_layer)
#         self.network.add_module('output_layer', self.output_layer)

#     def forward(self, x):
#         return self.network(x)




############ Autoencoder Models ############

### Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size,
                 num_hidden_layers=1, dropout_rate=0,
                 activation='leakyrelu', use_batch_norm=False, act_on_latent_layer=False):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.act_on_latent_layer = act_on_latent_layer
        self.encoder = Dense_Layers(input_size, hidden_size, 2*latent_size, 
                                    num_hidden_layers, dropout_rate, activation,
                                    use_batch_norm,act_on_output_layer=act_on_latent_layer)
        
        self.decoder = Dense_Layers(latent_size, hidden_size, input_size,
                                    num_hidden_layers, dropout_rate, activation,
                                    use_batch_norm)
        
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
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return torch.div(torch.add(recon_loss, kl_loss), x.size(0))
        # return torch.add(recon_loss, kl_loss)
    
    def forward_to_loss(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return self.loss(x, x_recon, mu, log_var)

    def generate(self, z):
        return self.decoder(z)
    
    def get_hyperparameters(self):
        return {'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'latent_size': self.latent_size,
                'num_hidden_layers': self.num_hidden_layers,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation,
                'use_batch_norm': self.use_batch_norm,
                'act_on_latent_layer': self.act_on_latent_layer}
    
    
### Autoencoder
class AE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size,
                 num_hidden_layers=1, dropout_rate=0,
                 activation='leakyrelu', use_batch_norm=False, act_on_latent_layer=False):
        super(AE, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.act_on_latent_layer = act_on_latent_layer
        self.encoder = Dense_Layers(input_size, hidden_size, latent_size, 
                                    num_hidden_layers, dropout_rate, activation,
                                    use_batch_norm, act_on_output_layer=act_on_latent_layer)
        
        self.decoder = Dense_Layers(latent_size, hidden_size, input_size,
                                    num_hidden_layers, dropout_rate, activation,
                                    use_batch_norm)
        
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

    def generate(self, z):
        return self.decoder(z)

    def get_hyperparameters(self):
        return {'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'latent_size': self.latent_size,
                'num_hidden_layers': self.num_hidden_layers,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation,
                'use_batch_norm': self.use_batch_norm,
                'act_on_latent_layer': self.act_on_latent_layer}

############ Classification Models ############
    
### Binary Classifier
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 num_hidden_layers=1, dropout_rate=0.2,
                 activation='leakyrelu', use_batch_norm=False):
        super(BinaryClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.network = Dense_Layers(input_size, hidden_size,
                                    output_size=1, 
                                    num_hidden_layers=num_hidden_layers,
                                    dropout_rate=dropout_rate,
                                    activation=activation,
                                    use_batch_norm=use_batch_norm)

        self.loss_func = nn.BCEWithLogitsLoss()

    def define_loss(self, class_weight=None, reduction='mean'):
        self.class_weight = class_weight
        self.reduction = reduction
        # self.loss_func = nn.BCEWithLogitsLoss(weight=class_weight,
                                                # reduction=reduction)
        # pos weight = # of negative samples / # of positive samples
        # class_weight = [1/(# of negative samples), 1/(# of positive samples)]
        # so pos_weight = (1/neg_weight)/(1/pos_weight) = pos_weight/neg_weight
        pos_weight = class_weight[1]/class_weight[0]
        self.loss_func = nn.BCEWithLogitsLoss(reduction=reduction,
                                                pos_weight=pos_weight)

    def forward(self, x):
        return self.network(x)
    
    def logits_to_proba(self, y_logits):
        return F.sigmoid(y_logits)

    def predict_proba(self, x):
        return F.sigmoid(self.network(x))
    
    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) > threshold).float()

    def loss(self, y_logits, y_true, ignore_nan=False):
        # if class_weight is None:
            # class_weight = torch.tensor([1, 1], dtype=torch.float32)
        if ignore_nan:
            mask = ~torch.isnan(y_true)
            return self.loss_func(y_logits[mask], y_true[mask]) 
        else:
            return self.loss_func(y_logits, y_true)
            # return F.binary_cross_entropy_with_logits(y_logits, y_true, 
            #                                 reduction=reduction,
            #                                 weight= class_weight)

    def get_hyperparameters(self):
        return {'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_hidden_layers': self.num_hidden_layers,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation,
                'use_batch_norm': self.use_batch_norm}   


### Multi-class Classifier
class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, 
                 num_hidden_layers=1, dropout_rate=0.2,
                 activation='leakyrelu', use_batch_norm=False):
        super(MultiClassClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.network = Dense_Layers(input_size, hidden_size,
                                    output_size=num_classes, 
                                    num_hidden_layers=num_hidden_layers,
                                    dropout_rate=dropout_rate,
                                    activation=activation,
                                    use_batch_norm=use_batch_norm)
        
        self.loss_func = nn.CrossEntropyLoss()
        # CrossEntropyLoss automatically applies softmax to the output layer
    
    def define_loss(self, class_weight=None, label_smoothing=0, reduction='mean'):
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.loss_func = nn.CrossEntropyLoss(weight=class_weight,
                                                label_smoothing=label_smoothing,
                                                reduction=reduction)

    def forward(self, x):
        return self.network(x)
    
    def logits_to_proba(self, y_logits):
        return F.softmax(y_logits, dim=1)

    def predict_proba(self, x):
        return F.softmax(self.network(x), dim=1)
    
    def predict(self, x):
        return torch.argmax(self.predict_proba(x), dim=1)

    def loss(self, y_logits, y_true, ignore_nan=False):
        if ignore_nan:
            mask = ~torch.isnan(y_true)
            return self.loss_func(y_logits[mask], y_true[mask])
        else:
            return self.loss_func(y_logits, y_true)
            # return F.cross_entropy(y_logits, y_true,
            #                         weight=self.class_weight,
            #                         reduction=self.reduction)

    def forward_to_loss(self, x, y_true):
        return self.loss(self.forward(x), y_true)

    def get_hyperparameters(self):
        return {'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_classes': self.num_classes,
                'num_hidden_layers': self.num_hidden_layers,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation,
                'use_batch_norm': self.use_batch_norm}


##### Regression Models #####
        
### Linear Regression
class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 num_hidden_layers=1, dropout_rate=0.2,
                 activation='leakyrelu', use_batch_norm=False):
        super(RegressionNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.network = Dense_Layers(input_size, hidden_size,
                                    output_size=1, 
                                    num_hidden_layers=num_hidden_layers,
                                    dropout_rate=dropout_rate,
                                    activation=activation,
                                    use_batch_norm=use_batch_norm)
        
        self.loss_func = nn.MSELoss()
    
    def forward(self, x):
        return self.network(x)
    
    def loss(self, y_pred, y_true, ignore_nan=False):
        if ignore_nan:
            mask = ~torch.isnan(y_true)
            return self.loss_func(y_pred[mask], y_true[mask])
        else:
            return self.loss_func(y_pred, y_true)
            # return F.mse_loss(y_pred, y_true, reduction='mean')
        
    def get_hyperparameters(self):
        return {'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_hidden_layers': self.num_hidden_layers,
                'dropout_rate': self.dropout_rate}

### Cox Proportional Hazards Neural Network Model
# TODO: not sure if this is the correct implementation
class CoxNN(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 num_hidden_layers=1, dropout_rate=0.2,
                 activation='leakyrelu', use_batch_norm=False):
        super(CoxNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.network = Dense_Layers(input_size, hidden_size,
                                    output_size=1, 
                                    num_hidden_layers=num_hidden_layers,
                                    dropout_rate=dropout_rate,
                                    activation=activation,
                                    use_batch_norm=use_batch_norm)
        
        self.loss_func = CoxPHLoss()
    
    def forward(self, x):
        return self.network(x)
    
    def loss(self, y_output, durations, events, ignore_nan=False):
        if ignore_nan:
            mask = ~torch.isnan(durations)
            return self.loss_func(y_output[mask], durations[mask], events[mask])
        else:
            return self.loss_func(y_output, durations, events)
    
    def predict_risk(self, x):
        # which is the correct output layer?
        # return F.sigmoid(self.network(x))
        return torch.exp(self.network(x))
    
    def get_hyperparameters(self):
        return {'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_hidden_layers': self.num_hidden_layers,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation,
                'use_batch_norm': self.use_batch_norm}