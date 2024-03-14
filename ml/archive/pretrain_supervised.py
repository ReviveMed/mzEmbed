import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
import json
import os
import matplotlib.pyplot as plt
from models import VAE, AE, TGEM, BinaryClassifier, MultiClassClassifier
from prep import ClassifierDataset
import time
import numpy as np
from torchmetrics import Accuracy, AUROC
from misc import get_clean_batch_sz, get_dropbox_dir


def get_model(model_kind, input_size, model_hyperparameters):

    if model_kind == 'VAE':
        model = VAE(input_size = input_size, **model_hyperparameters)
    elif model_kind == 'AE':
        model = AE(input_size = input_size, **model_hyperparameters)
    elif model_kind == 'TGEM':
        model = TGEM(input_size = input_size, **model_hyperparameters)
    elif model_kind == 'BinaryClassifier':
        model = BinaryClassifier(input_size = input_size, **model_hyperparameters)
    elif model_kind == 'MultiClassClassifier':
        model = MultiClassClassifier(input_size = input_size, **model_hyperparameters)
    else:
        raise ValueError('model_kind not recognized')
    return model



def classifier_training_wrapper(data_dir,**kwargs):

    n_subsets = kwargs.get('n_subsets', 5)
    label_col = kwargs.get('label_col', 'cohort')
    label_encoder = kwargs.get('label_encoder', None)
    val_fraction = kwargs.get('val_fraction', 0.1)
    early_stopping = kwargs.get('early_stopping', -1)
    model_name = kwargs.get('model_name', 'DEFAULT')


     # Load the finetuning data
    test_auc_list = []
    for subset_id in range(n_subsets):
        print('subset_id:', subset_id)
        finetune_dataset = ClassifierDataset(data_dir, 
                                    subset='train_{}'.format(subset_id),
                                    label_col=label_col,
                                    label_encoder=label_encoder)
        
        class_weights = 1 / torch.bincount(finetune_dataset.y.long())

        test_dataset = ClassifierDataset(data_dir, 
                                    subset='test_{}'.format(subset_id),
                                    label_col=label_col,
                                    label_encoder=label_encoder)
    
        # create a validation set
        # should this be stratified?
        if (val_fraction > 0) and (early_stopping > 0):
            val_size = int(val_fraction * len(finetune_dataset))
            train_size = len(finetune_dataset) - val_size
            train_dataset, val_dataset = random_split(finetune_dataset, [train_size, val_size])
            # train_dataset = finetune_dataset[train_dataset.indices]
            # val_dataset = finetune_dataset[val_dataset.indices]
        else:
            train_dataset = finetune_dataset
            val_dataset = None

        # create finetuning dataloaders
        batch_size = int(batch_size)
        dataloaders_dict = {
            'train': DataLoader(train_dataset, 
                                batch_size= get_clean_batch_sz(len(train_dataset), batch_size),
                                shuffle=True),
            # 'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0),
            'test': DataLoader(test_dataset, 
                               batch_size=get_clean_batch_sz(len(test_dataset), batch_size),
                               shuffle=False),
            # 'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        }
        if val_dataset is not None:
            dataloaders_dict['val'] = DataLoader(val_dataset, 
                                                     batch_size=get_clean_batch_sz(len(val_dataset), batch_size),
                                                     shuffle=False)

        model_id = model_name + '_{}'.format(subset_id)
        output_data = run_classification_training(dataloaders_dict,
                                                  model_id=model_id,
                                                  **kwargs)
        test_auc_list.append(output_data['end_state_auroc']['test'])

    avg_cv_AUC = np.mean(test_auc_list)
    return avg_cv_AUC




def run_classification_training(dataloaders,**kwargs):

    assert isinstance(dataloaders, dict)

    save_dir = kwargs.get('save_dir', None)
    device = kwargs.get('device', None)
    phase_list = kwargs.get('phase_list', None)
    task = kwargs.get('task', 'classify')
    yesplot = kwargs.get('yesplot', True)
    verbose = kwargs.get('verbose', False)
    how_average = kwargs.get('how_average', 'weighted')

    # Model hyperparameters
    input_size = kwargs.get('input_size', None)
    model_kind = kwargs.get('model_kind', None)
    model_name = kwargs.get('model_name', None)
    model_id = kwargs.get('model_id',model_name)
    user_model_hyperparameters = kwargs.get('model_hyperparameters', None)
    model_eval_funcs = kwargs.get('model_eval_funcs', None)
    load_model_path = kwargs.get('load_model_path', None)

    # learning hyperparameters
    optimizer_kind = kwargs.get('optimizer_kind', None)
    learning_rate = kwargs.get('learning_rate', None)
    momentum = kwargs.get('momentum', None)
    weight_decay = kwargs.get('weight_decay', 0)
    scheduler_kind = kwargs.get('scheduler_kind', None)
    scheduler_hyperparams = kwargs.get('scheduler_hyperparams', None)
    num_epochs = kwargs.get('num_epochs', None)
    noise_factor = kwargs.get('noise_factor', 0)
    early_stopping = kwargs.get('early_stopping', -1)


    if save_dir is not None:
        model_save_path = os.path.join(save_dir, model_id + '_model.pth')
        output_save_path = os.path.join(save_dir, model_id + '_output.json')
    else:
        model_save_path = None
        output_save_path = None


    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
            # device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if phase_list is None:
        phase_list = list(dataloaders.keys())
    dataset_size_dct = {phase: len(dataloaders[phase].dataset) for phase in phase_list}
    batch_size_dct = {phase: dataloaders[phase].batch_size for phase in phase_list}

    if input_size is None:
        input_size = dataloaders['train'].dataset[0][0].shape[0]

    if early_stopping < 0:
        early_stopping = num_epochs        



    class_weight = kwargs.get('class_weights', None)
    num_classes = kwargs.get('num_classes', None)
    if num_classes is None:
        num_classes = len(torch.unique(dataloaders['train'].dataset[:][1]))

    hyperparams = {k: v for k, v in user_model_hyperparameters.items()}    
    if 'num_classes' in user_model_hyperparameters:
        assert hyperparams['num_classes'] == num_classes
    else:
        hyperparams['num_classes'] = num_classes

    # Instantiate the model
    model = get_model(model_kind, input_size=input_size, **hyperparams)

    if (class_weight is not None) and (len(class_weight) == num_classes):
        if isinstance(class_weight, list):
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
    elif class_weight is not None:
        y_train = dataloaders['train'].dataset[:][1]
        class_weight = 1 / torch.bincount(y_train.long())
    else:
        class_weight = None

    if optimizer_kind == 'Adam':
        # from TGEM paper:
        # optimizer =torch.optim.Adam(model.parameters(), 
            # lr=lr_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
            # amsgrad=False)

        optimizer = optim.Adam(model.parameters(), 
                            lr=learning_rate, 
                            weight_decay=weight_decay)
    else: 
        raise NotImplementedError('Only Adam optimizer is currently supported')

    if scheduler_kind:
        raise NotImplementedError('Scheduler not yet implemented')
    
    if scheduler_hyperparams:
        raise NotImplementedError('Scheduler not yet implemented')

    learning_parameters = {
        'optimizer_kind': optimizer_kind,
        'learning_rate': learning_rate,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'scheduler_kind': scheduler_kind,
        'scheduler_hyperparams': scheduler_hyperparams,
        'num_epochs': num_epochs,
        'noise_factor': noise_factor,
        'early_stopping': early_stopping,
        'class_weight': class_weight.numpy().tolist() if class_weight is not None else None,
        'dataset_sizes': dataset_size_dct,
        'batch_sizes': batch_size_dct
    }

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_wts = None
    best_epoch = -1
    loss_history = {'train': [], 'val': []}
    acc_history = {'train': [], 'val': []}
    auroc_history = {'train': [], 'val': []}
    patience_counter = 0


    if load_model_path:
        print('Loading model from:', load_model_path)
        model.load_state_dict(torch.load(load_model_path))


    model.define_loss(class_weight=class_weight)

    running_accuracy = Accuracy(task=task,average=how_average).to(device)
    running_auroc = AUROC(task=task,average=how_average).to(device)
    # val_accuracy = Accuracy(task=task,average=how_average)
    # val_auroc = AUROC(task=task,average=how_average)

    model = model.to(device)
    for epoch in range(num_epochs):
        if (patience_counter >= early_stopping) and (best_model_wts is not None):
            model.load_state_dict(best_model_wts)
            if verbose: print('Early stopping at epoch', epoch)
            break

        for phase in ['train', 'val']:
            if phase not in dataloaders:
                continue
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_auroc.reset()
            running_accuracy.reset()
            
            # epoch_preds = []
            # epoch_targets = []

            for data in dataloaders[phase]:
                X, y = data
                X = X.to(device)
                y = y.to(device)
                y = y.view(-1,1)
                # zero the parameter gradients, so they don't accumulate
                optimizer.zero_grad()

                # noise injection for training to make model more robust
                if (noise_factor>0) and (phase == 'train'):
                    X = X + noise_factor * torch.randn_like(X)


                with torch.set_grad_enabled(phase == 'train'):
                    # the foward pass is used to build the computational graph to be used later during the backward pass
                    outputs = model(X) 
                    loss = model.loss(outputs, y)
                    # # outputs_probs = model.predict_proba(X_encoded) #this runs a full forward pass which isn't necessary
                    outputs_probs = model.logits_to_proba(outputs)
                    # epoch_preds.append(outputs_probs)
                    # epoch_targets.append(y)
                    if phase == 'train':
                        # calculate the backpropagation to find the gradients of the loss with respect to the model parameters
                        loss.backward()
                        # update the model parameters using the gradients using the optimizer
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_accuracy(outputs_probs, y)
                running_auroc(outputs_probs, y)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            loss_history[phase].append(epoch_loss)
            
            # Alternative to using Torchmetrics
            # epoch_preds = torch.cat(epoch_preds, dim=0)
            # epoch_targets = torch.cat(epoch_targets, dim=0)
            # epoch_accuracy = accuracy_score(epoch_targets, epoch_preds)
            # epoch_auroc = roc_auc_score(epoch_targets, epoch_preds)

            epoch_accuracy = running_accuracy.compute().item()
            epoch_auroc = running_auroc.compute().item()
            acc_history[phase].append(epoch_accuracy)
            auroc_history[phase].append(epoch_auroc)
            if verbose:
                print(f'Epoch {epoch}/{num_epochs - 1}, {phase} loss: {epoch_loss:.6f}, \
                    acc: {epoch_accuracy:.4f}, auroc: {epoch_auroc:.4f}')

            if phase == 'val':
                if epoch < early_stopping/8:
                    # don't take the best model from the first few epochs
                    continue
                if epoch_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1


    # Save model state dict
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)

    if best_epoch < 0:
        best_epoch = num_epochs - 1

    # Evaluate model on training, validation, and test datasets
    phase_sizes = {phase: len(dataloaders[phase].dataset) for phase in phase_list}
    end_state_losses = {}
    end_state_acc = {}
    end_state_auroc = {}
    model = model.to('cpu') # for evaluation
    encoder_model = encoder_model.to('cpu') # for evaluation
    for phase in phase_list:
        model.eval()
        running_loss = 0.0
        running_accuracy.reset()
        running_auroc.reset()
        with torch.inference_mode():
            for data in dataloaders[phase]:
                X, y = data
                y = y.view(-1,1)
                X_encoded = encoder_model.transform(X)
                outputs = model(X_encoded)
                loss = model.loss(outputs, y)
                outputs_probs = model.logits_to_proba(outputs)
                running_loss += loss.item() * X.size(0)
                running_accuracy(outputs_probs, y)
                running_auroc(outputs_probs, y)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            end_state_losses[phase] = epoch_loss
            epoch_accuracy = running_accuracy.compute().item()
            end_state_acc[phase] = epoch_accuracy
            epoch_auroc = running_auroc.compute().item()
            end_state_auroc[phase] = epoch_auroc
            if verbose:
                print(f'End state {phase} loss: {epoch_loss:.6f}, acc: {epoch_accuracy:.4f}, auroc: {epoch_auroc:.4f}')

    model_hyperparameters = model.get_hyperparameters()

    output_data = {
            'model_name': model_name,
            'model_id': model_id,
            'model_kind': model_kind,
            'user_model_hyperparameters': user_model_hyperparameters,
            'model_hyperparameters': model_hyperparameters,
            'learning_parameters': learning_parameters,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'phase_sizes': phase_sizes,
            'end_state_losses': end_state_losses,
            'end_state_acc': end_state_acc,
            'end_state_auroc': end_state_auroc,
            'loss_history': loss_history,
            'acc_history': acc_history,
            'auroc_history': auroc_history
        }

    if output_save_path:
        with open(output_save_path, 'w') as f:
            json.dump(output_data, f)

    if yesplot:
        for history, name in zip([loss_history, acc_history, auroc_history], ['loss', 'acc', 'auroc']):
            plt.figure(figsize=(6,4))
            plt.plot(history['train'], label='train', lw=2)
            plt.plot(history['val'], label='val', lw=2)
            plt.title(f'{model_name}')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, model_id + '_' + name+'.png'))
            plt.close()

    return output_data


if __name__ == '__main__':

    dropbox_dir = get_dropbox_dir()
    data_dir = os.path.join(dropbox_dir, 'development_CohortCombination','reconstruction_study_feb16')
    model_kind = 'TGEM'
    model_name = 'tgem'
    save_dir = os.path.join(data_dir,'models_feb16')
    os.makedirs(save_dir, exist_ok=True)

    hyperparameters = {
        'n_head' : 5,
        'query_gene' : 64,
        'd_ff' : 1024,
        'dropout_rate': 0.3,
        'act_fun': "linear"}
    
    classifier_training_wrapper(data_dir,
                                model_kind=model_kind,
                                model_name=model_name,
                                model_hyperparameters=hyperparameters,
                                save_dir=save_dir,
                                early_stopping=-1,
                                n_subsets= 4,
                                num_epochs=10,
                                batch_size=32,
                                learning_rate=1e-3,
                                yesplot=True,
                                verbose=True)