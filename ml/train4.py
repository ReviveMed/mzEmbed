

import torch
import json
import os
import torch.nn.functional as F
from models import get_reg_penalty, grad_reverse
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import time
import random
import math
import torch.utils.data
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchmetrics import Accuracy, AUROC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from collections import defaultdict
import neptune
from neptune.utils import stringify_unsupported


##################################################################################
##################################################################################
######### for training the compound model

class CompoundDataset(Dataset):
    def __init__(self, X, y_head=None, y_adv=None, other=None):
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        
        if (y_head is None) or (y_head.size == 0):
            self.y_head = torch.tensor(np.zeros((len(X), 1)), dtype=torch.float32)
        else:
            for col in y_head.columns:
                if y_head[col].dtype == 'object':
                    print('converting', col, 'to category')
                    y_head.loc[:, col] = y_head[col].astype('category').cat.codes
            self.y_head = torch.tensor(y_head.astype(float).to_numpy(), dtype=torch.float32)
        
        if (y_adv is None) or (y_adv.size == 0):
            self.y_adv = torch.tensor(np.zeros((len(X), 1)), dtype=torch.float32)
        else:
            for col in y_adv.columns:
                if y_adv[col].dtype == 'object':
                    print('converting', col, 'to category')
                    y_adv.loc[:, col] = y_adv[col].astype('category').cat.codes
            self.y_adv = torch.tensor(y_adv.astype(float).to_numpy(), dtype=torch.float32)

        if (other is None) or (other.size == 0):
            self.other = torch.tensor(np.zeros((len(X), 1)), dtype=torch.float32)
        else:
            self.other = other


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_head[idx,:], self.y_adv[idx,:], self.other[idx,:]





def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, random_state=None):
    # Code from Alvtron on Github
    #  a simple function for conducting a stratified split with random shuffling, 
    # similar to that of StratifiedShuffleSplit from scikit-learn
    # https://gist.github.com/Alvtron/9b9c2f870df6a54fda24dbd1affdc254
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels

def create_dataloaders_old(torch_dataset, batch_size, holdout_frac=0, shuffle=True, set_name='train'):
    
    if holdout_frac == 0:
        return {set_name: DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)}

    train_size = int((1-holdout_frac) * len(torch_dataset))
    holdout_size = len(torch_dataset) - train_size

    train_dataset, holdout_dataset = torch.utils.data.random_split(torch_dataset, [train_size, holdout_size])

    return {set_name: DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
            f'{set_name}_holdout': DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)}


def create_dataloaders(torch_dataset, batch_size, holdout_frac=0, shuffle=True, set_name='train', stratify=None):
    if holdout_frac == 0:
        return {set_name: DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)}

    # Get the targets of your dataset if it's available
    if stratify is not None:
        targets = [item[stratify] for item in torch_dataset]
        # convert nans to -1 for the purpose of stratification
        for item in targets:
            item[torch.isnan(item)] = -1
    else:
        targets = None

    # Split the indices of your dataset
    indices = list(range(len(torch_dataset)))
    train_indices, holdout_indices = train_test_split(indices, test_size=holdout_frac, stratify=targets)

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    holdout_sampler = SubsetRandomSampler(holdout_indices)

    return {set_name: DataLoader(torch_dataset, batch_size=batch_size, sampler=train_sampler),
            f'{set_name}_holdout': DataLoader(torch_dataset, batch_size=batch_size, sampler=holdout_sampler)}


def get_optimizer(optimizer_name, model, learning_rate=0.001, weight_decay=0,betas=(0.9, 0.999)):
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,betas=betas)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,betas=betas)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError('Optimizer not yet implemented')


##################################################################################
##################################################################################

def train_compound_model(dataloaders,encoder,head,adversary, run, **kwargs):

    # run should be a neptune run object
    num_epochs = kwargs.get('num_epochs', 10)
    encoder_weight = kwargs.get('encoder_weight', 1)
    head_weight = kwargs.get('head_weight', 1)
    adversary_weight = kwargs.get('adversary_weight', 1)
    early_stopping_patience = kwargs.get('early_stopping_patience', -1)
    noise_factor = kwargs.get('noise_factor', 0)
    l1_reg_weight = kwargs.get('l1_reg_weight', 0)
    l2_reg_weight = kwargs.get('l2_reg_weight', 0)
    penalize_decoder_weights = kwargs.get('penalize_decoder_weights', False) # for backwards compatibility, set to True
    phase_list = kwargs.get('phase_list', None)
    scheduler_kind = kwargs.get('scheduler_kind', None)
    scheduler_kwargs = kwargs.get('scheduler_kwargs', {})
    verbose = kwargs.get('verbose', True)
    #adversarial_mini_epochs = kwargs.get('adversarial_mini_epochs', 1)
    adversarial_start_epoch = kwargs.get('adversarial_start_epoch', -1)
    prefix = kwargs.get('prefix', 'train')
    train_name = kwargs.get(f'train_name', 'train')
    freeze_encoder = kwargs.get('freeze_encoder', False)
    
    optimizer_name = kwargs.get('optimizer_name', 'adamw')
    head_optimizer_name = kwargs.get('head_optimizer_name', optimizer_name)
    adversarial_optimizer_name = kwargs.get('adversarial_optimizer_name', 'sgd')

    learning_rate = kwargs.get('learning_rate', kwargs.get('lr', 0.001))
    head_learning_rate = kwargs.get('head_learning_rate', learning_rate)
    adversarial_learning_rate = kwargs.get('adversarial_learning_rate', 10*learning_rate)

    weight_decay = kwargs.get('weight_decay', 0)
    head_weight_decay = kwargs.get('head_weight_decay', weight_decay)
    adversary_weight_decay = kwargs.get('adversary_weight_decay', 0.1*weight_decay)

    
    clip_grads_with_norm = kwargs.get('clip_grads_with_norm', True)
    clip_grads_with_value = kwargs.get('clip_grads_with_value', False)
    clip_value = kwargs.get('clip_value', 1)

    if (not clip_grads_with_norm) and (not clip_grads_with_value):
        print('Warning: clip_grads_with_norm=False and clip_grads_with_value=False: this may lead to exploding gradients')
    assert clip_value > 0

    if phase_list is None:
        phase_list = list(dataloaders.keys())

    print('Training on', train_name)
    if train_name not in phase_list:
        print(f'Warning: train_name ({train_name}) not in phase_list')
        for phase in phase_list:
            if 'train' in phase:
                train_name = phase
                break
    print('Training on', train_name)
    start_time = time.time()

    if scheduler_kind is not None:
        # Right now, Scheduler only impacts the encoder optimizer
        #TODO Do we want to combine the encoder and head optimizers so it impacts both?
        assert scheduler_kind in ['ReduceLROnPlateau', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR']
        raise NotImplementedError('Scheduler not yet implemented')
    # if scheduler_kind is not None:
        # 'scheduler_kind': 'ReduceLROnPlateau',
        #     'scheduler_kwargs': {
        #         'factor': 0.1,
        #         'patience': 5,
        #         'min_lr': 1e-6
        #     }

    dataset_size_dct = {phase: len(dataloaders[phase].dataset) for phase in phase_list}
    batch_size_dct = {phase: dataloaders[phase].batch_size for phase in phase_list}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    learning_parameters = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'noise_factor': noise_factor,
        'l1_reg_weight': l1_reg_weight,
        'l2_reg_weight': l2_reg_weight,
        'encoder_weight': encoder_weight,
        'head_weight': head_weight,
        'adversary_weight': adversary_weight,
        'phase_list': phase_list,
        'phase_sizes': dataset_size_dct,
        'batch_sizes': batch_size_dct,
        # 'adversarial_mini_epochs': adversarial_mini_epochs,
        'adversarial_start_epoch': adversarial_start_epoch,
        'freeze_encoder': freeze_encoder,
    }
    #TODO add a run prefix to all the keys
    run[f'{prefix}/learning_parameters'] = stringify_unsupported(learning_parameters)

    if early_stopping_patience < 1:
        early_stopping_patience = num_epochs



    # define the losses averages across the epochs
    encoder_type = encoder.kind
    encoder.to(device)
    head.to(device)
    adversary.to(device)
    
    # Freeze the weights of the encoder except for the last layer
    if freeze_encoder:
        if hasattr(encoder, 'encoder'):
            print('freeze the encoder parameters')
            for param in encoder.encoder.parameters():
                param.requires_grad = False

        if encoder.kind == 'metabFoundation':  
            print('unfreeze the last layer of the encoder')    
            for param in encoder.embed_to_encoder.transformer_encoder.layers[-1].parameters():
                param.requires_grad = True

        elif encoder.kind == 'VAE':
            raise NotImplementedError('Freezing the encoder for VAE is not yet implemented')
        else:
            raise NotImplementedError('Freezing the encoder for this encoder type is not yet implemented')
    
    # encoder_optimizer = get_optimizer(optimizer_name, filter(lambda p: p.requires_grad, encoder.parameters()), learning_rate, weight_decay)
    encoder_optimizer = get_optimizer(optimizer_name, encoder, learning_rate, weight_decay)
    
    if head_weight > 0:
        head_optimizer = get_optimizer(head_optimizer_name, head, head_learning_rate, head_weight_decay)
    else: 
        head_optimizer = None
    
    if adversary_weight > 0:
        adversary_optimizer = get_optimizer(adversarial_optimizer_name, adversary, adversarial_learning_rate, adversary_weight_decay)
    else:
        adversary_optimizer = None


    scheduler = None
    if scheduler_kind is not None:
        if scheduler_kind == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, **scheduler_kwargs)
        elif scheduler_kind == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, **scheduler_kwargs)
        elif scheduler_kind == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, **scheduler_kwargs)
        elif scheduler_kind == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(encoder_optimizer, **scheduler_kwargs)
        elif scheduler_kind == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, **scheduler_kwargs)
        else:
            raise NotImplementedError('Scheduler not yet implemented')

    # define the loss history and best params with associated losses
    loss_history = {
        'encoder': {f'{train_name}': [], f'{train_name}_holdout': []},
        'head': {f'{train_name}': [], f'{train_name}_holdout': []},
        'adversary': {f'{train_name}': [], f'{train_name}_holdout': []},
        'joint': {f'{train_name}': [], f'{train_name}_holdout': []},
        'epoch' : {f'{train_name}': [], f'{train_name}_holdout': []},
    }
    best_loss = {'encoder': 1e10, 'head': 1e10, 'adversary': 1e10, 'joint': 1e10, 'epoch': 0}
    best_wts = {'encoder': encoder.state_dict(), 'head': head.state_dict(), 'adversary': adversary.state_dict()}
    patience_counter = 0


    # start the training loop
    for epoch in range(num_epochs):
        if encoder_type == 'TGEM_Encoder':
            print('Epoch', epoch)
        if patience_counter > early_stopping_patience:
            print('Early stopping at epoch', epoch)
            encoder.load_state_dict(best_wts['encoder'])
            head.load_state_dict(best_wts['head'])
            adversary.load_state_dict(best_wts['adversary'])
            break


        for phase in [f'{train_name}', f'{train_name}_holdout']:
            if phase not in dataloaders:
                continue
            if phase == f'{train_name}':
                encoder.train()
                head.train()
                adversary.train()
            else:
                encoder.eval()
                head.eval()
                adversary.eval()

            running_losses = {'encoder': 0, 'head': 0, 'adversary': 0, 'joint': 0}
            
            epoch_head_outputs = torch.tensor([])
            epoch_adversary_outputs = torch.tensor([])
            epoch_head_targets = torch.tensor([])
            epoch_adversary_targets = torch.tensor([])
            num_batches = len(dataloaders[phase])

            for batch_idx, data in enumerate(dataloaders[phase]):
                if encoder_type == 'TGEM_Encoder':
                    if batch_idx % 50 == 0:
                        print('Batch', batch_idx, '/', num_batches)
                    # if batch_idx> 3:
                        # continue

                X, y_head, y_adversary, clin_vars = data
                X = X.to(device)
                y_head = y_head.to(device)
                y_adversary = y_adversary.to(device)
                clin_vars = clin_vars.to(device)

                # noise injection for training to make model more robust
                if (noise_factor>0) and (phase == f'{train_name}'):
                    X = X + noise_factor * torch.randn_like(X)


                # Train the Adversarial Head First
                # This is a change from the original implementation, implemented end of day April 16th
                #TODO: this is a little messy, should probably be refactored
                # Currently transform the data twice with the encoder, this probably only needs to be done once
                # but if we do change it, we need to be careful to zero the gradients properly
                # if (adversary_weight > 0) and (epoch > adversarial_start_epoch):
                #     with torch.set_grad_enabled(phase == f'{train_name}'):
                #         z = encoder.transform(X).detach()
                #         z.requires_grad = True
                #         adversary_optimizer.zero_grad()
                #         y_adversary_output = adversary(z)
                #         # adversary_loss = adversary.loss(y_adversary_output, y_adversary)
                #         multi_loss = adversary.loss(y_adversary_output, y_adversary)
                #         if isinstance(multi_loss, dict):
                #             for key, loss_val in multi_loss.items():
                #                 run[f'{prefix}/{phase}/adversary_loss/{key}'].append(loss_val)
                #             adversary_loss = sum([h.weight * multi_loss[f'{h.kind}_{h.name}'] for h in adversary.heads])
                #         else:
                #             adversary_loss = multi_loss

                #         run[f'{prefix}/{phase}/multi_adversary_loss'].append(adversary_loss)
                    
                #         if phase == f'{train_name}':
                #             adversary_loss.backward()
                #             adversary_optimizer.step()



                encoder_optimizer.zero_grad()
                if head_weight > 0:
                    head_optimizer.zero_grad()
                if (adversary_weight > 0) and (epoch > adversarial_start_epoch):
                    adversary_optimizer.zero_grad()
                with torch.set_grad_enabled(phase == f'{train_name}'):
                    if encoder_weight > 0:
                        z, encoder_loss = encoder.transform_with_loss(X)
                    else:
                        z = encoder.transform(X)
                        encoder_loss = torch.tensor(0)
                    

                    if head_weight > 0:
                        y_head_output = head(torch.cat((z, clin_vars), 1))
                        # multi_loss = head.multi_loss(y_head_output, y_head)
                        multi_loss = head.loss(y_head_output, y_head)
                        if isinstance(multi_loss, dict):
                            for key, loss_val in multi_loss.items():
                                if torch.isnan(loss_val) or torch.isinf(loss_val):
                                    continue
                                # run[f'{prefix}/{phase}/batch/head_loss/{key}'].append(loss_val)
                                run[f'{prefix}/{phase}/batch/head_loss/{key}'].append(stringify_unsupported(loss_val))

                            head_loss = sum([h.weight * multi_loss[f'{h.kind}_{h.name}'] for h in head.heads])
                            # head_weight_sum = sum([h.weight for h in head.heads])
                            # if head_weight_sum>0.1:
                                # head_loss /= head_weight_sum
                        else:
                            head_loss = multi_loss
                        
                    else:
                        y_head_output = torch.tensor([])
                        head_loss = torch.tensor(0)
                

                    if (adversary_weight > 0) and (epoch > adversarial_start_epoch):
                        # z2 = z.detach()
                        # z2.requires_grad = True
                        z_reverse = grad_reverse(z)
                        y_adversary_output = adversary(z_reverse)
                        # adversary_loss = adversary.loss(y_adversary_output, y_adversary)
                        multi_loss = adversary.loss(y_adversary_output, y_adversary)
                        if isinstance(multi_loss, dict):
                            for key, loss_val in multi_loss.items():
                                if torch.isnan(loss_val) or torch.isinf(loss_val):
                                    continue
                                # run[f'{prefix}/{phase}/batch/adversary_loss/{key}'].append(loss_val)
                                run[f'{prefix}/{phase}/batch/adversary_loss/{key}'].append(stringify_unsupported(loss_val))
                                
                            adversary_loss = sum([h.weight * multi_loss[f'{h.kind}_{h.name}'] for h in adversary.heads])
                            # adv_weight_sum = sum([h.weight for h in adversary.heads])
                            # if adv_weight_sum>0.1:
                                # adversary_loss /= adv_weight_sum
                        else:
                            adversary_loss = multi_loss
                    else:
                        y_adversary_output = torch.tensor([])
                        adversary_loss = torch.tensor(0)

                    # check if the losses are nan
                    if torch.isnan(encoder_loss) or torch.isinf(encoder_loss):
                        print('Encoder loss is nan/inf!')
                        print('your learning rate is probably too high')
                        end_time = time.time()
                        elapsed_minutes = (end_time - start_time) / 60
                        print(f'Training took {elapsed_minutes:.2f} minutes')
                        return None, None, None
                    else:
                        run[f'{prefix}/{phase}/batch/encoder_loss'].append(encoder_loss)

                    if torch.isnan(head_loss) or torch.isinf(head_loss):
                        print('Head loss is nan/inf!')
                        print('your learning rate is probably too high')
                        end_time = time.time()
                        elapsed_minutes = (end_time - start_time) / 60
                        print(f'Training took {elapsed_minutes:.2f} minutes')
                        return None, None, None
                    else:
                        run[f'{prefix}/{phase}/batch/multi_head_loss'].append(head_loss)

                    if torch.isnan(adversary_loss) or torch.isinf(adversary_loss):
                        print('Adversary loss is nan/inf!')
                        print('your learning rate is probably too high')
                        end_time = time.time()
                        elapsed_minutes = (end_time - start_time) / 60
                        print(f'Training took {elapsed_minutes:.2f} minutes')
                        return None, None, None
                    else:
                        run[f'{prefix}/{phase}/batch/multi_adversary_loss'].append(adversary_loss)
                        # stringifying the loss for neptune
                        # run[f'{prefix}/{phase}/batch/multi_adversary_loss'].append(stringify_unsupported(adversary_loss))


                    if isinstance(y_head_output, dict):
                        if isinstance(epoch_head_outputs, dict):
                            for k in y_head_output.keys():
                                epoch_head_outputs[k] = torch.cat((epoch_head_outputs[k], y_head_output[k].to("cpu").detach()), 0)
                        else:
                            epoch_head_outputs = {k: y_head_output[k].to("cpu").detach() for k in y_head_output.keys()}
                    else:
                        epoch_head_outputs = torch.cat((epoch_head_outputs, y_head_output.to("cpu").detach()), 0)
                    epoch_head_targets = torch.cat((epoch_head_targets, y_head.to("cpu").detach()), 0)

                    if epoch > adversarial_start_epoch:
                        if isinstance(y_adversary_output, dict):
                            if isinstance(epoch_adversary_outputs, dict):
                                for k in y_adversary_output.keys():
                                    epoch_adversary_outputs[k] = torch.cat((epoch_adversary_outputs[k], y_adversary_output[k].to("cpu").detach()), 0)
                            else:
                                epoch_adversary_outputs = {k: y_adversary_output[k].to("cpu").detach() for k in y_adversary_output.keys()}
                        else:
                            epoch_adversary_outputs = torch.cat((epoch_adversary_outputs, y_adversary_output.to("cpu").detach()), 0)

                        epoch_adversary_targets = torch.cat((epoch_adversary_targets, y_adversary.to("cpu").detach()), 0)
                    
                    joint_loss = (encoder_weight*encoder_loss + \
                        head_weight*head_loss + \
                        adversary_weight*adversary_loss) #/ (encoder_weight + head_weight + adversary_weight)

                    if not torch.isnan(joint_loss) or torch.isinf(joint_loss):
                        # run[f'{prefix}/{phase}/batch/joint_loss'].append(joint_loss)
                        # strinify support for neptune
                        run[f'{prefix}/{phase}/batch/joint_loss'].append(stringify_unsupported(joint_loss))
                    
                    #TODO: there is an argument for not recording the losses when there is no gradient
                    # since they are not being used for optimization
                    # but for now we will record them
                    if not torch.isnan(encoder_loss) or torch.isinf(encoder_loss):
                        running_losses['encoder'] += encoder_loss.item()
                    if not torch.isnan(head_loss) or torch.isinf(head_loss):
                        running_losses['head'] += head_loss.item()
                    if not torch.isnan(adversary_loss) or torch.isinf(adversary_loss):
                        running_losses['adversary'] += adversary_loss.item()
                    if not torch.isnan(joint_loss) or torch.isinf(joint_loss):
                        running_losses['joint'] += joint_loss.item()

                    if (not joint_loss.requires_grad) and (phase == f'{train_name}'):
                        print('Joint loss has no gradient! skip backward pass')
                        # this occurs when all of the targets are nan for the heads and adversary, and no encoder loss
                        continue

                    # if (joint_loss.item() == 0):
                    #     print('Joint loss is 0! skip backward pass')
                    #     continue

                    if (phase == f'{train_name}'):
                        
                        joint_loss_w_penalty = joint_loss
                        # right now we are also penalizing the weights in the decoder, do we want to do that?
                        if penalize_decoder_weights: # for backwards compatibility
                            joint_loss_w_penalty += get_reg_penalty(encoder, l1_reg_weight, l2_reg_weight)
                        else:
                            if 'encoder' in encoder._modules:
                                joint_loss_w_penalty += get_reg_penalty(encoder.encoder, l1_reg_weight, l2_reg_weight)
                                if (encoder_weight > 0) and ('decoder' in encoder._modules):
                                    joint_loss_w_penalty += get_reg_penalty(encoder.decoder, l1_reg_weight, l2_reg_weight)
                            else:
                                joint_loss_w_penalty += get_reg_penalty(encoder, l1_reg_weight, l2_reg_weight)


                        joint_loss_w_penalty += get_reg_penalty(head, l1_reg_weight, l2_reg_weight)

                        # we probably don't care about the adversary weights
                        # joint_loss += get_reg_penalty(adversary, l1_reg_weight, l2_reg_weight)

                        joint_loss_w_penalty.backward(retain_graph=True)
                        if clip_grads_with_norm:
                            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_value)
                            torch.nn.utils.clip_grad_norm_(head.parameters(), clip_value)
                            # torch.nn.utils.clip_grad_value_(encoder.parameters(), 1)
                        if clip_grads_with_value:
                            torch.nn.utils.clip_grad_value_(encoder.parameters(), clip_value)
                            torch.nn.utils.clip_grad_value_(head.parameters(), clip_value)

                        encoder_optimizer.step()
                        if head_weight > 0:
                            head_optimizer.step()

                        if (adversary_weight > 0) and (epoch > adversarial_start_epoch):
                            # zero the adversarial optimizer again
                            # adversary_optimizer.zero_grad()

                            # Backward pass and optimize the adversarial classifiers
                            # adversary_loss.backward()
                            adversary_optimizer.step()


            
            ############ end of training adversary on fixed latent space
            #turn on inference mode
            with torch.inference_mode():
                # divide losses by the number of batches
                running_losses['encoder'] /= len(dataloaders[phase])
                running_losses['head'] /= len(dataloaders[phase])
                running_losses['adversary'] /= len(dataloaders[phase])
                running_losses['joint'] /= len(dataloaders[phase])
                # check for nans and infinities, replace with 0
                for k in running_losses.keys():
                    if not torch.isfinite(torch.tensor(running_losses[k])) or torch.isnan(torch.tensor(running_losses[k])):
                        print(f'Warning: {k} loss is not finite')
                        running_losses[k] = 0 #torch.tensor(0.0)

                # run[f'{prefix}/{phase}/epoch/encoder_loss'].append(running_losses['encoder'])
                # run[f'{prefix}/{phase}/epoch/multi_head_loss'].append(running_losses['head'])
                # run[f'{prefix}/{phase}/epoch/multi_adversary_loss'].append(running_losses['adversary'])
                # run[f'{prefix}/{phase}/epoch/joint_loss'].append(running_losses['joint'])

                # strinify support for neptune
                run[f'{prefix}/{phase}/epoch/encoder_loss'].append(stringify_unsupported(running_losses['encoder']))
                run[f'{prefix}/{phase}/epoch/multi_head_loss'].append(stringify_unsupported(running_losses['head']))
                run[f'{prefix}/{phase}/epoch/multi_adversary_loss'].append(stringify_unsupported(running_losses['adversary']))
                run[f'{prefix}/{phase}/epoch/joint_loss'].append(stringify_unsupported(running_losses['joint']))

                loss_history['encoder'][phase].append(running_losses['encoder'])
                loss_history['head'][phase].append(running_losses['head'])
                loss_history['adversary'][phase].append(running_losses['adversary'])
                loss_history['joint'][phase].append(running_losses['joint'])
                loss_history['epoch'][phase].append(epoch)

                eval_scores = {}
                #TODO more efficient to aggregate the scores in the batch loop, see documentation for torchmetrics
                # but unclear how this will work with other metrics
                
                if head_weight > 0:
                    eval_scores.update(head.score(epoch_head_outputs, epoch_head_targets))
                if (adversary_weight > 0) and (epoch > adversarial_start_epoch):
                    eval_scores.update(adversary.score(epoch_adversary_outputs, epoch_adversary_targets))
                
                for eval_name, eval_val in eval_scores.items():
                    
                    if isinstance(eval_val, dict):
                        for k, v in eval_val.items():
                            # run[f'{prefix}/{phase}/epoch/{eval_name}/{k}'].append(v)
                            run[f'{prefix}/{phase}/epoch/{eval_name}__{k}'].append(stringify_unsupported(v))
                    else:
                        # run[f'{prefix}/{phase}/epoch/{eval_name}'].append(eval_val)
                        run[f'{prefix}/{phase}/epoch/{eval_name}'].append(stringify_unsupported(eval_val))


                if (verbose) and (epoch % 10 == 0):
                    print(f'Epoch [{epoch+1}/{num_epochs}], {phase} Loss: {loss_history["joint"][phase][-1]:.4f}')
                    if encoder_weight > 0:
                        print(f'{phase} Encoder Loss: {loss_history["encoder"][phase][-1]:.4f}')

            if phase == f'{train_name}_holdout':
                if scheduler is not None:
                    scheduler.step(loss_history['joint'][phase][-1])

                if loss_history['joint'][phase][-1] < best_loss['joint']:
                    best_loss['joint'] = loss_history['joint'][phase][-1]
                    best_loss['encoder'] = loss_history['encoder'][phase][-1]
                    best_loss['head'] = loss_history['head'][phase][-1]
                    best_loss['adversary'] = loss_history['adversary'][phase][-1]
                    best_loss['epoch'] = epoch
                    best_wts['encoder'] = encoder.state_dict()
                    best_wts['head'] = head.state_dict()
                    best_wts['adversary'] = adversary.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

        run[f'{prefix}/best_epoch'] = best_loss['epoch']
        # run[f'{prefix}/best_joint_loss'] = best_loss['joint']

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f'Training took {elapsed_minutes:.2f} minutes')
    return encoder, head, adversary

# %%
##################################################################################
##################################################################################


def evaluate_compound_model(dataloaders, encoder, head, adversary, run, **kwargs):
    
    prefix = kwargs.get('prefix', 'eval')
    sklearn_models = kwargs.get('sklearn_models', None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder.eval()
    head.eval()
    adversary.eval()

    if sklearn_models is None:
        sklearn_models= {}

    # evaluate the model on datasets
    end_state_losses = {}
    end_state_eval = {}
    phase_list = list(dataloaders.keys())

    train_name = kwargs.get(f'train_name', 'train')

    if train_name not in phase_list:
        for phase in phase_list:
            if 'train' in phase:
                train_name = phase
                break

    if f'{train_name}' in phase_list:
        # put train at the beginning
        phase_list.remove(f'{train_name}')
        phase_list = [f'{train_name}'] + phase_list


    latent_outputs_by_phase = defaultdict(dict)
    adversary_targets_by_phase = defaultdict(dict)
    all_outputs = defaultdict(dict)
    all_targets = defaultdict(dict)
    
    for phase in phase_list:
        encoder = encoder.to(device)
        head = head.to(device)
        adversary = adversary.to(device)
        recon_loss = torch.tensor(0.0).to(device)
        head_loss = torch.tensor(0.0).to(device)
        adversary_loss = torch.tensor(0.0).to(device)
        

        latent_outputs = torch.tensor([])
        head_ouputs = torch.tensor([])
        adversary_outputs = torch.tensor([])
        
        head_targets = torch.tensor([])
        adversary_targets = torch.tensor([])
        
        with torch.inference_mode():
            for data in dataloaders[phase]:
                X, y_head, y_adversary, clin_vars = data
                X = X.to(device)
                y_head = y_head.to(device)
                y_adversary = y_adversary.to(device)
                clin_vars = clin_vars.to(device)
                
                if encoder.kind == 'metabFoundation':
                    X_hidden = torch.rand_like(X) < encoder.default_hidden_fraction
                    X_hidden.to(device)
                    x_seq, x_pos_ids, x_mask, x_pad = encoder.metab_to_seq.transform(X,x_hidden=X_hidden)
                    z, z_enc, z_pos_ids, z_mask, z_pad = encoder.transform(X,x_hidden=X_hidden,as_seq=True)
                    z_recon = encoder.generate(z_enc, x_pos_ids=z_pos_ids, as_seq=True)
                    # compute only the masked value reconstruction
                    recon_loss += F.mse_loss(x_seq[x_mask], z_recon[x_mask])* X.size(0)
                else:
                    z = encoder.transform(X)
                    X_recon = encoder.generate(z)
                    recon_loss += F.mse_loss(X_recon, X)* X.size(0)

                if sklearn_models:
                    latent_outputs = torch.cat((latent_outputs, z), 0)
                
                y_head_output = head(torch.cat((z, clin_vars), 1))
                y_adversary_output = adversary(z)
                head_loss += head.joint_loss(y_head_output, y_head) * y_head.size(0)
                adversary_loss += adversary.joint_loss(y_adversary_output, y_adversary) * y_adversary.size(0)
                
                # y_head_output = y_head_output.to("cpu").detach()
                # y_adversary_output = y_adversary_output.to("cpu").detach()
                if isinstance(y_head_output, dict):
                    if isinstance(head_ouputs, dict):
                        for k in y_head_output.keys():
                            head_ouputs[k] = torch.cat((head_ouputs[k], y_head_output[k].to("cpu").detach()), 0)
                    else:
                        head_ouputs = {k: y_head_output[k].to("cpu").detach() for k in y_head_output.keys()}
                else:
                    head_ouputs = torch.cat((head_ouputs, y_head_output.to("cpu").detach()), 0)

                if isinstance(y_adversary_output, dict):
                    if isinstance(adversary_outputs, dict):
                        for k in y_adversary_output.keys():
                            adversary_outputs[k] = torch.cat((adversary_outputs[k], y_adversary_output[k].to("cpu").detach()), 0)
                    else:
                        adversary_outputs = {k: y_adversary_output[k].to("cpu").detach() for k in y_adversary_output.keys()}
                else:
                    adversary_outputs = torch.cat((adversary_outputs, y_adversary_output.to("cpu").detach()), 0)

                y_head = y_head.to("cpu").detach()
                y_adversary = y_adversary.to("cpu").detach()
                head_targets = torch.cat((head_targets, y_head), 0)
                adversary_targets = torch.cat((adversary_targets, y_adversary), 0)

            # all_outputs[phase] = {
            #     'head': head_ouputs,
            #     'adversary': adversary_outputs
            # }
            # all_targets[phase] = {
            #     'head': head_targets,
            #     'adversary': adversary_targets
            # }

            # head_loss.to("cpu")
            # adversary_loss.to("cpu")
            # recon_loss.to("cpu")
            latent_outputs_by_phase[phase] = latent_outputs
            adversary_targets_by_phase[phase] = adversary_targets
            recon_loss = recon_loss / len(dataloaders[phase].dataset)
            head_loss = head_loss / len(dataloaders[phase].dataset)
            adversary_loss = adversary_loss / len(dataloaders[phase].dataset)
            end_state_losses[phase] = {
                'reconstruction': recon_loss.item(),
                'head': head_loss.item(),
                'adversary': adversary_loss.item(),
                }
            
            head.to("cpu")
            adversary.to("cpu")
            end_state_eval[phase] = {}
            end_state_eval[phase] = head.score(head_ouputs, head_targets)
            end_state_eval[phase].update(adversary.score(adversary_outputs, adversary_targets))
            
            # we now append the evaluation metrics, so we have a history of them
            for eval_name, eval_val in end_state_eval[phase].items():
                if isinstance(eval_val, dict):
                    for k, v in eval_val.items():
                        # run[f'{prefix}/{phase}/{eval_name}/{k}'] = stringify_unsupported(v)
                        run[f'{prefix}/{phase}/{eval_name}__{k}'].append(stringify_unsupported(v))
                else:
                    # run[f'{prefix}/{phase}/{eval_name}'] = stringify_unsupported(eval_val)
                    run[f'{prefix}/{phase}/{eval_name}'].append(stringify_unsupported(eval_val))

            if not torch.isfinite(recon_loss):
                print(f'Warning: recon loss is not finite')
                recon_loss = torch.tensor(0.0)
            if not torch.isfinite(head_loss):
                print(f'Warning: head loss is not finite')
                head_loss = torch.tensor(0.0)
            if not torch.isfinite(adversary_loss):
                print(f'Warning: adversary loss is not finite')
                adversary_loss = torch.tensor(0.0)

            # We now append the losses, so we can have a history over multiple iterations of evaluation
            run[f'{prefix}/{phase}/reconstruction_loss'].append(recon_loss)
            run[f'{prefix}/{phase}/head_loss'].append(head_loss)
            run[f'{prefix}/{phase}/adversary_loss'].append(adversary_loss)
            
            # this line below is redundent
            # for eval_name, eval_val in end_state_eval[phase].items():
            #     run[f'{prefix}/{phase}/{eval_name}'].append(stringify_unsupported(eval_val))

            # run[f'{prefix}/{phase}/reconstruction_loss'] = recon_loss
            # run[f'{prefix}/{phase}/head_loss'] = head_loss
            # run[f'{prefix}/{phase}/adversary_loss'] = adversary_loss
            # for eval_name, eval_val in end_state_eval[phase].items():
            #     run[f'{prefix}/{phase}/{eval_name}'] = stringify_unsupported(eval_val)


    #TODO This part is messy, need to clean up, currently skipping
    if (sklearn_models) and (False):
        try:  
            for adv0 in adversary.heads:

                adv0_name = adv0.name

                for phase in phase_list:
                    latent_outputs = latent_outputs_by_phase[phase]
                    adversary_targets = adversary_targets_by_phase[phase]
                    adv0_targets = adversary_targets[:,adv0.y_idx]
                    
                    for model_name, model in sklearn_models.items():
                        nan_mask = ~torch.isnan(adv0_targets)

                        if phase == f'{train_name}':
                            model.fit(
                                latent_outputs[nan_mask].detach().numpy(), 
                                # adversary_targets[nan_mask].detach().long().numpy())
                                adv0_targets[nan_mask].detach().numpy())
                        else:
                            # task = adversary.task
                            if adv0.kind =='Binary':
                                task = 'binary'
                                num_classes = 2
                                metric = AUROC(task=task,average='macro')
                                probs = torch.tensor(model.predict_proba(latent_outputs[nan_mask].detach().numpy()))[:,1]
                            elif adv0.kind == 'MultiClass':
                                task = 'multiclass'
                                num_classes = adv0.num_classes
                                metric = AUROC(task=task,average='macro',num_classes=num_classes)
                                probs = torch.tensor(model.predict_proba(latent_outputs[nan_mask].detach().numpy()))
                            
                            metric(probs,adv0_targets[nan_mask].long())
                            # run[f'{prefix}/{phase}/{adv0.kind}_{adv0_name}/{model_name} AUROC (macro)'] = stringify_unsupported(metric.compute().item())
                            run[f'{prefix}/{phase}/{adv0.kind}_{adv0_name}/{model_name}_AUROC (macro)'].append(stringify_unsupported(metric.compute().item()))

            
        except Exception as e:
            print('Error in sklearn model evaluation:', e)
            run[f'{prefix}/{phase}/sklearn_error'] = str(e)


    return




# %%
##################################################################################
##################################################################################

# Classifier Evaluation Functions
def evaluate_auc(outputs, targets, model):
    if len(outputs) == 0:
        return 0
    
    if model.goal != 'classify':
        return np.nan
    
    probs = model.logits_to_proba(outputs.detach()) #.numpy()
    nan_mask = ~torch.isnan(targets)
    # probs = probs[nan_mask]
    # targets = targets[nan_mask].long()
    if model.num_classes == 2:
        task = 'binary'
    else:
        task = 'multiclass'

    metric = AUROC(task=task,average='weighted',num_classes=model.num_classes)
    metric(probs[nan_mask],targets[nan_mask].long())
    return metric.compute().item()


def evaluate_adversary_auc(head_ouputs, head_targets, adversary_outputs, adversary_targets,head, adversary):
    return evaluate_auc(adversary_outputs, adversary_targets, adversary)

def evaluate_head_auc(head_ouputs, head_targets, adversary_outputs, adversary_targets,head, adversary):
    return evaluate_auc(head_ouputs, head_targets, head)

# would be better to integrate the adversary into the model class
# def get_ensemble_eval_funcs(model):
#     eval_funcs = {
#         'head_auc': lambda outputs, targets: evaluate_auc(outputs, targets, model),
#     }
#     if model.adversary is not None:
#         eval_funcs['adversary_auc'] = lambda outputs, targets: evaluate_auc(outputs, targets, model.adversary)
#     return eval_funcs


def get_end_state_eval_funcs():
    end_state_eval_funcs = {
        'head_auc': evaluate_head_auc,
        'adversary_auc': evaluate_adversary_auc
    }
    return end_state_eval_funcs

##################################################################################
##################################################################################