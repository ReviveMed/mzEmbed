# %%
import torch
from torch import nn
import torch.nn.functional as F
from loss import CoxPHLoss
from torch.utils.checkpoint import checkpoint as cp

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


class TGEM(torch.nn.Module):
    def __init__(self, input_size,  n_head, num_classes, dropout_rate, act_fun, 
                 query_gene=64, d_ff=1024, mode=0):
        super(TGEM, self).__init__()
        self.n_head = n_head
        self.input_size = input_size
        self.num_classes = num_classes
        self.d_ff = d_ff
        self.act_fun = act_fun
        # mode 1 is not working right now.
        self.mode = mode
        
        if self.act_fun == 'relu':
            self.activation_func = torch.nn.ReLU()
        elif self.act_fun == 'leakyrelu':
            self.activation_func = torch.nn.LeakyReLU(0.1)
        elif self.act_fun == 'gelu':
            self.activation_func = torch.nn.GELU()
        elif self.act_fun == 'linear':
            self.activation_func = torch.nn.Identity()
        else:
            raise ValueError('{} is not a valid activation function'.format(self.act_fun))


        self.activation_func 
        self.query_gene = query_gene #in original version, this was not a object parameter 
        self.mulitiattention1 = mulitiattention( self.n_head, self.input_size, query_gene,
                                                mode)
        self.mulitiattention2 = mulitiattention( self.n_head, self.input_size, query_gene,
                                                mode)
        self.mulitiattention3 = mulitiattention( self.n_head, self.input_size, query_gene,
                                                mode)
        self.fc = nn.Linear(self.input_size, self.num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=1)
        self.ffn1 = nn.Linear(self.input_size, self.d_ff)
        self.ffn2 = nn.Linear(self.d_ff, self.input_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sublayer = res_connect(input_size, dropout_rate)

        self.loss_func = nn.CrossEntropyLoss()
        # CrossEntropyLoss automatically applies softmax to the output layer
    
    def define_loss(self, class_weight=None, label_smoothing=0, reduction='mean'):
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        # self.loss_func = nn.NLLLoss(weight=class_weight,
        #                             label_smoothing=label_smoothing,
        #                             reduction=reduction)
        self.loss_func = nn.CrossEntropyLoss(weight=class_weight,
                                                label_smoothing=label_smoothing,
                                                reduction=reduction)

    def feedforward(self, x):
        out = F.relu(self.ffn1(x))
        out = self.ffn2(self.dropout(out))
        return out

    def forward(self, x):

        out_attn = self.mulitiattention1(x)
        out_attn_1 = self.sublayer(x, out_attn)
        out_attn_2 = self.mulitiattention2(out_attn_1)
        out_attn_2 = self.sublayer(out_attn_1, out_attn_2)
        out_attn_3 = self.mulitiattention3(out_attn_2)
        out_attn_3 = self.sublayer(out_attn_2, out_attn_3)
        out_attn_3 = self.activation_func(out_attn_3)
        y_output = self.fc(out_attn_3)
        # y_output = F.log_softmax(y_output, dim=1) # not as numerically stable as using CrossEntropyLoss

        return y_output

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

    def logits_to_proba(self, y_logits):
        # return np.exp(y_logits) # when the logits are the log_softmax
        return F.softmax(y_logits, dim=1)
    
    def get_hyperparameters(self):
        return {
                'n_head': self.n_head,
                'input_size': self.input_size,
                'query_gene': self.query_gene,
                'num_classes': self.num_classes,
                'd_ff': self.d_ff,
                'act_fun': self.act_fun}        



class TGEMRegression(TGEM):
    def __init__(self, *args, **kwargs):
        super(TGEMRegression, self).__init__(*args, **kwargs)

        # Change the final layer to output a single value for regression
        self.fc = nn.Linear(self.input_size, 1) # I could consider adding some additional layers
        # self.fc1 = nn.Linear(self.input_size, 4)
        # self.fc2 = nn.Linear(4, 1)

        self.loss_func = nn.MSELoss()
        # CrossEntropyLoss automatically applies softmax to the output layer
    
    def define_loss(self, reduction='mean'):

        self.reduction = reduction
        self.loss_func = nn.MSELoss(reduction=reduction)


    def forward(self, x):
        out_attn = self.mulitiattention1(x)
        out_attn_1 = self.sublayer(x, out_attn)
        out_attn_2 = self.mulitiattention2(out_attn_1)
        out_attn_2 = self.sublayer(out_attn_1, out_attn_2)
        out_attn_3 = self.mulitiattention3(out_attn_2)
        out_attn_3 = self.sublayer(out_attn_2, out_attn_3)
        if self.act_fun == 'relu':
            out_attn_3 = F.relu(out_attn_3)
        if self.act_fun == 'leakyrelu':
            m = torch.nn.LeakyReLU(0.1)
            out_attn_3 = m(out_attn_3)
        if self.act_fun == 'gelu':
            m = torch.nn.GELU()
            out_attn_3 = m(out_attn_3)
        
        # out1 = self.fc1(out_attn_3)
        # out1 = F.relu(out1)
        # y_pred = self.fc2(out1)
        y_pred = self.fc(out_attn_3)

        return y_pred