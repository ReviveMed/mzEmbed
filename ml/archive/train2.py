

import torch
import json
import os
import torch.nn.functional as F
from models import AE, MultiClassClassifier, BinaryClassifier, VAE, get_reg_penalty
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchmetrics import Accuracy, AUROC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import neptune
from neptune.utils import stringify_unsupported


##################################################################################
##################################################################################
######### for training the compound model

class CompoundDataset(Dataset):
    def __init__(self, X, y_head, y_adv, other=None):
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y_head = torch.tensor(y_head.to_numpy(), dtype=torch.float32)
        self.y_adv = torch.tensor(y_adv.to_numpy(), dtype=torch.float32)
        if other is None:
            self.other = torch.tensor(np.zeros((len(X), 1)), dtype=torch.float32)
        else:
            self.other = other


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_head[idx], self.y_adv[idx], self.other[idx]
    
    def get_class_weights_head(self):
        y_no_nan = self.y_head[~torch.isnan(self.y_head)]
        y_int = y_no_nan.int()
        return 1/torch.bincount(y_int)
    
    def get_class_weights_adv(self):
        y_no_nan = self.y_adv[~torch.isnan(self.y_adv)]
        y_int = y_no_nan.int()
        return 1/torch.bincount(y_int)
    
    def get_num_classes_head(self):
        y_no_nan = self.y_head[~torch.isnan(self.y_head)]
        return len(y_no_nan.unique())
    
    def get_num_classes_adv(self):
        y_no_nan = self.y_adv[~torch.isnan(self.y_adv)]
        return len(y_no_nan.unique())



def create_dataloaders(torch_dataset, batch_size, holdout_frac=0, shuffle=True, set_name='train'):
    
    if holdout_frac == 0:
        return {set_name: DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)}

    train_size = int((1-holdout_frac) * len(torch_dataset))
    holdout_size = len(torch_dataset) - train_size

    train_dataset, holdout_dataset = torch.utils.data.random_split(torch_dataset, [train_size, holdout_size])

    return {set_name: DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
            f'{set_name}_holdout': DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)}


##################################################################################
##################################################################################

def train_compound_model(dataloaders,encoder,head,adversary, run, **kwargs):

    # run should be a neptune run object
    learning_rate = kwargs.get('learning_rate', kwargs.get('lr', 0.001))
    weight_decay = kwargs.get('weight_decay', 0)
    num_epochs = kwargs.get('num_epochs', 10)
    encoder_weight = kwargs.get('encoder_weight', 1)
    head_weight = kwargs.get('head_weight', 1)
    adversary_weight = kwargs.get('adversary_weight', 1)
    early_stopping_patience = kwargs.get('early_stopping_patience', -1)
    noise_factor = kwargs.get('noise_factor', 0)
    l1_reg_weight = kwargs.get('l1_reg_weight', 0)
    l2_reg_weight = kwargs.get('l2_reg_weight', 0)
    phase_list = kwargs.get('phase_list', None)
    scheduler_kind = kwargs.get('scheduler_kind', None)
    scheduler_kwargs = kwargs.get('scheduler_kwargs', {})
    verbose = kwargs.get('verbose', True)
    eval_funcs = kwargs.get('eval_funcs', {})
    adversarial_mini_epochs = kwargs.get('adversarial_mini_epochs', 20)
    prefix = kwargs.get('prefix', 'train')

    

    if phase_list is None:
        phase_list = list(dataloaders.keys())
    
    if scheduler_kind is not None:
        raise NotImplementedError('Scheduler not yet implemented')
    # if scheduler_kind is not None:
    #     'scheduler_kind': 'ReduceLROnPlateau',
    #         'scheduler_kwargs': {
    #             'factor': 0.1,
    #             'patience': 5,
    #             'min_lr': 1e-6
    #         }

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
        'adversarial_mini_epochs': adversarial_mini_epochs,
    }
    #TODO add a run prefix to all the keys
    run[f'{prefix}/learning_parameters'] = stringify_unsupported(learning_parameters)

    if early_stopping_patience < 0:
        early_stopping_patience = num_epochs


    # define the losses averages across the epochs
    encoder_type = encoder.kind
    encoder.to(device)
    head.to(device)
    adversary.to(device)

    # define the optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if head_weight > 0:
        head_optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        head_optimizer = None
    
    if adversary_weight > 0:
        adversary_optimizer = torch.optim.Adam(adversary.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        adversary_optimizer = None

    # define the loss history and best params with associated losses
    loss_history = {
        'encoder': {'train': [], 'train_holdout': []},
        'head': {'train': [], 'train_holdout': []},
        'adversary': {'train': [], 'train_holdout': []},
        'joint': {'train': [], 'train_holdout': []},
        'epoch' : {'train': [], 'train_holdout': []},
    }
    best_loss = {'encoder': 1e10, 'head': 1e10, 'adversary': 1e10, 'joint': 1e10, 'epoch': 0}
    best_wts = {'encoder': encoder.state_dict(), 'head': head.state_dict(), 'adversary': adversary.state_dict()}
    eval_history = {'train': {}, 'train_holdout': {}}
    patience_counter = 0

    # start the training loop
    for epoch in range(num_epochs):
        if encoder_type == 'TGEM_Encoder':
            print('Epoch', epoch)
        run_status = True
        if patience_counter > early_stopping_patience:
            print('Early stopping at epoch', epoch)
            encoder.load_state_dict(best_wts['encoder'])
            head.load_state_dict(best_wts['head'])
            adversary.load_state_dict(best_wts['adversary'])
            break


        for phase in ['train', 'train_holdout']:
            if phase not in dataloaders:
                continue
            if phase == 'train':
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
                    if batch_idx % 5 == 0:
                        print('Batch', batch_idx, '/', num_batches)
                    # if batch_idx> 3:
                        # continue

                X, y_head, y_adversary, clin_vars = data
                X = X.to(device)
                y_head = y_head.to(device)
                y_adversary = y_adversary.to(device)
                clin_vars = clin_vars.to(device)

                # noise injection for training to make model more robust
                if (noise_factor>0) and (phase == 'train'):
                    X = X + noise_factor * torch.randn_like(X)

                encoder_optimizer.zero_grad()
                if head_weight > 0:
                    head_optimizer.zero_grad()
                if adversary_weight > 0:
                    adversary_optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if encoder_weight > 0:
                        z, encoder_loss = encoder.transform_with_loss(X)
                    else:
                        z = encoder.transform(X)
                        encoder_loss = torch.tensor(0)
                    

                    if head_weight > 0:
                        y_head_output = head(torch.cat((z, clin_vars), 1))
                        head_loss = head.loss(y_head_output, y_head)
                    else:
                        y_head_output = torch.tensor([])
                        head_loss = torch.tensor(0)
                

                    if adversary_weight > 0:
                        z2 = z.detach()
                        z2.requires_grad = True

                        y_adversary_output = adversary(z2)
                        adversary_loss = adversary.loss(y_adversary_output, y_adversary)
                    else:
                        y_adversary_output = torch.tensor([])
                        adversary_loss = torch.tensor(0)

                    # check if the losses are nan
                    if torch.isnan(encoder_loss):
                        if run_status:
                            print('Encoder loss is nan!')
                        encoder_loss = torch.tensor(0)

                    if torch.isnan(head_loss):
                        if run_status:
                            print('Head loss is nan!')
                        # print('Head loss is nan!')
                        head_loss = torch.tensor(0)

                    run[f'{prefix}/{phase}/batch/encoder_loss'].append(encoder_loss)
                    run[f'{prefix}/{phase}/batch/head_loss'].append(head_loss)
                    run[f'{prefix}/{phase}/batch/adversary_loss'].append(adversary_loss)

                    # new addition, if join loss is 0, then only nans are present
                    # we don't want to backpropagate
                    if (encoder_loss.item() == 0) and (head_loss.item() == 0):
                        if run_status:
                            print('skipping backprop')
                            print('y head output:', y_head_output)
                            print('y_head:', y_head.view(-1))
                            run_status = False
                        continue

                    epoch_head_outputs = torch.cat((epoch_head_outputs, y_head_output), 0)
                    epoch_adversary_outputs = torch.cat((epoch_adversary_outputs, y_adversary_output), 0)
                    epoch_head_targets = torch.cat((epoch_head_targets, y_head), 0)
                    epoch_adversary_targets = torch.cat((epoch_adversary_targets, y_adversary), 0)
                    
                    joint_loss = (encoder_weight*encoder_loss + \
                        head_weight*head_loss - \
                        adversary_weight*adversary_loss)

                    run[f'{prefix}/{phase}/batch/joint_loss'].append(joint_loss)
                    
                    if phase == 'train':
                        
                        joint_loss_w_penalty = joint_loss
                        # right now we are also penalizing the weights in the decoder, do we want to do that?
                        joint_loss_w_penalty += get_reg_penalty(encoder, l1_reg_weight, l2_reg_weight)
                        # joint_loss += get_reg_penalty(encoder.encoder, l1_reg_weight, l2_reg_weight)
                        joint_loss_w_penalty += get_reg_penalty(head, l1_reg_weight, l2_reg_weight)

                        # we probably don't care about the adversary weights
                        # joint_loss += get_reg_penalty(adversary, l1_reg_weight, l2_reg_weight)

                        joint_loss_w_penalty.backward(retain_graph=True)
                        
                        encoder_optimizer.step()
                        if head_weight > 0:
                            head_optimizer.step()

                        if adversary_weight > 0:
                            # zero the adversarial optimizer again
                            adversary_optimizer.zero_grad()

                            # Backward pass and optimize the adversarial classifiers
                            adversary_loss.backward()
                            adversary_optimizer.step()
                    

                    running_losses['encoder'] += encoder_loss.item()
                    running_losses['head'] += head_loss.item()
                    running_losses['adversary'] += adversary_loss.item()
                    running_losses['joint'] += joint_loss.item()


            ########  Train the adversary on the fixed latent space for a few epochs
            if (adversary_weight > 0) and (phase == 'train'):
                encoder.eval()
                head.eval()
                for mini_epoch in range(adversarial_mini_epochs):       
                    for batch_idx, data in enumerate(dataloaders[phase]):
                        X, y_head, y_adversary, clin_vars = data
                        X = X.to(device)
                        y_adversary = y_adversary.to(device)

                        # noise injection for training to make model more robust
                        if (noise_factor>0):
                            X = X + noise_factor * torch.randn_like(X)

                        with torch.set_grad_enabled(phase == 'train'):
                            z = encoder.transform(X).detach()
                            z.requires_grad = True
                            adversary_optimizer.zero_grad()
                            y_adversary_output = adversary(z)
                            adversary_loss = adversary.loss(y_adversary_output, y_adversary)
                            run[f'{prefix}/{phase}/adversary_loss'].append(adversary_loss)
                        
                            if phase == 'train':
                                adversary_loss.backward()
                                adversary_optimizer.step()
            
            ############ end of training adversary on fixed latent space

            running_losses['encoder'] /= len(dataloaders[phase])
            running_losses['head'] /= len(dataloaders[phase])
            running_losses['adversary'] /= len(dataloaders[phase])
            running_losses['joint'] /= len(dataloaders[phase])

            run[f'{prefix}/{phase}/epoch/encoder_loss'].append(running_losses['encoder'])
            run[f'{prefix}/{phase}/epoch/head_loss'].append(running_losses['head'])
            run[f'{prefix}/{phase}/epoch/adversary_loss'].append(running_losses['adversary'])
            run[f'{prefix}/{phase}/epoch/joint_loss'].append(running_losses['joint'])

            loss_history['encoder'][phase].append(running_losses['encoder'])
            loss_history['head'][phase].append(running_losses['head'])
            loss_history['adversary'][phase].append(running_losses['adversary'])
            loss_history['joint'][phase].append(running_losses['joint'])
            loss_history['epoch'][phase].append(epoch)

            for eval_name, eval_func in eval_funcs.items():
                if ('head' in eval_name) and (epoch_head_outputs.size(0) == 0):
                    continue
                if ('adversary' in eval_name) and (epoch_adversary_outputs.size(0) == 0):
                    continue

                if eval_name not in eval_history[phase]:
                    eval_history[phase][eval_name] = []
                    
                eval_history[phase][eval_name].append(eval_func(head_ouputs = epoch_head_outputs,
                                                    head_targets = epoch_head_targets,
                                                    adversary_outputs = epoch_adversary_outputs,
                                                    adversary_targets = epoch_adversary_targets,
                                                    head = head,
                                                    adversary = adversary))
                
                for eval_name, eval_list in eval_history[phase].items():
                    run[f'{prefix}/{phase}/epoch/{eval_name}'].append(eval_list[-1])

            if (verbose) and (epoch % 10 == 0):
                print(f'Epoch [{epoch+1}/{num_epochs}], {phase} Loss: {loss_history["joint"][phase][-1]:.4f}')
                if encoder_weight > 0:
                    print(f'{phase} Encoder Loss: {loss_history["encoder"][phase][-1]:.4f}')
                for eval_name, eval_list in eval_history[phase].items():
                    print(f'{phase} {eval_name} {eval_list[-1]}')

            if phase == 'train_holdout':
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

    return encoder, head, adversary

# %%
##################################################################################
##################################################################################


def evaluate_compound_model(dataloaders, encoder, head, adversary, run, **kwargs):
    
    eval_funcs = kwargs.get('eval_funcs', get_end_state_eval_funcs())
    prefix = kwargs.get('prefix', 'eval')
    sklearn_models = kwargs.get('sklearn_models', None)
    
    encoder.eval()
    head.eval()
    adversary.eval()

    if sklearn_models is None:
        sklearn_models= {}

    # evaluate the model on datasets
    end_state_losses = {}
    end_state_eval = {}
    phase_list = list(dataloaders.keys())
    if 'train' in phase_list:
        # put train at the beginning
        phase_list.remove('train')
        phase_list = ['train'] + phase_list

    for phase in phase_list:
        recon_loss = torch.tensor(0.0)
        head_loss = torch.tensor(0.0)
        adversary_loss = torch.tensor(0.0)
        

        latent_outputs = torch.tensor([])
        head_ouputs = torch.tensor([])
        adversary_outputs = torch.tensor([])
        
        head_targets = torch.tensor([])
        adversary_targets = torch.tensor([])
        
        with torch.inference_mode():
            for data in dataloaders[phase]:
                X, y_head, y_adversary, clin_vars = data
                z = encoder.transform(X)
                if sklearn_models:
                    latent_outputs = torch.cat((latent_outputs, z), 0)
                X_recon = encoder.generate(z)
                recon_loss += F.mse_loss(X_recon, X)* X.size(0)
                y_head_output = head(torch.cat((z, clin_vars), 1))
                y_adversary_output = adversary(z)
                head_loss += head.loss(y_head_output, y_head) * y_head.size(0)
                adversary_loss += adversary.loss(y_adversary_output, y_adversary) * y_adversary.size(0)
                head_ouputs = torch.cat((head_ouputs, y_head_output), 0)
                adversary_outputs = torch.cat((adversary_outputs, y_adversary_output), 0)
                head_targets = torch.cat((head_targets, y_head), 0)
                adversary_targets = torch.cat((adversary_targets, y_adversary), 0)

            recon_loss = recon_loss / len(dataloaders[phase].dataset)
            head_loss = head_loss / len(dataloaders[phase].dataset)
            adversary_loss = adversary_loss / len(dataloaders[phase].dataset)
            end_state_losses[phase] = {
                'reconstruction': recon_loss.item(),
                'head': head_loss.item(),
                'adversary': adversary_loss.item(),
                }
            
            end_state_eval[phase] = {}
            for eval_name, eval_func in eval_funcs.items():
                end_state_eval[phase][eval_name] = eval_func(head_ouputs=head_ouputs,
                                                head_targets=head_targets,
                                                adversary_outputs=adversary_outputs,
                                                adversary_targets=adversary_targets,
                                                head=head,
                                                adversary=adversary)

            run[f'{prefix}/{phase}/reconstruction_loss'] = recon_loss
            run[f'{prefix}/{phase}/head_loss'] = head_loss
            run[f'{prefix}/{phase}/adversary_loss'] = adversary_loss
            for eval_name, eval_val in end_state_eval[phase].items():
                run[f'{prefix}/{phase}/{eval_name}'] = stringify_unsupported(eval_val)

            if sklearn_models:
                try:
                    for model_name, model in sklearn_models.items():
                        nan_mask = ~torch.isnan(adversary_targets)

                        if phase == 'train':
                            model.fit(
                                latent_outputs[nan_mask].detach().numpy(), 
                                # adversary_targets[nan_mask].detach().long().numpy())
                                adversary_targets[nan_mask].detach().numpy())
                            sklearn_models[model_name] = model
                        else:
                            # task = adversary.task
                            if adversary.num_classes == 2:
                                task = 'binary'
                            else:
                                task = 'multiclass'
                            metric = AUROC(task=task,average='weighted',num_classes=adversary.num_classes)
                            probs = torch.tensor(model.predict_proba(latent_outputs[nan_mask].detach().numpy()))
                            metric(probs,adversary_targets[nan_mask].long())
                            run[f'{prefix}/{phase}/{model_name}_auc'] = stringify_unsupported(metric.compute().item())
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