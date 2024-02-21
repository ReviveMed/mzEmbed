
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
import json
import os
import matplotlib.pyplot as plt
from models import VAE, AE, TGEM, BinaryClassifier, MultiClassClassifier
from prep import ClassifierDataset, PreTrainingDataset
import time
import numpy as np
from torchmetrics import Accuracy, AUROC
from misc import get_clean_batch_sz, get_dropbox_dir, round_to_even


def get_model(model_kind, input_size, **kwargs):

    if model_kind == 'VAE':
        model = VAE(input_size = input_size, **kwargs)
    elif model_kind == 'AE':
        model = AE(input_size = input_size, **kwargs)
    elif model_kind == 'TGEM':
        model = TGEM(input_size = input_size, **kwargs)
    elif model_kind == 'BinaryClassifier':
        model = BinaryClassifier(input_size = input_size, **kwargs)
    elif model_kind == 'MultiClassClassifier':
        model = MultiClassClassifier(input_size = input_size, **kwargs)
    else:
        raise ValueError('model_kind not recognized')
    return model


def unsupervised_pretraining_wrapper(data_dir,**kwargs):

    n_subsets = kwargs.get('n_subsets', 4)
    val_fraction = kwargs.get('val_fraction', 0.1)
    early_stopping = kwargs.get('early_stopping', -1)
    model_name = kwargs.get('model_name', 'DEFAULT')
    batch_size = kwargs.get('batch_size', 64)


     # Load the finetuning data
    test_recon_loss_list = []
    for subset_id in range(n_subsets):
        print('subset_id:', subset_id)
        initial_train_dataset = PreTrainingDataset(data_dir, 
                                    subset='train_{}'.format(subset_id))
        

        test_dataset = PreTrainingDataset(data_dir, 
                                    subset='test_{}'.format(subset_id))
    
        # create a validation set
        # should this be stratified?
        if (val_fraction > 0) and (early_stopping > 0):
            val_size = int(val_fraction * len(initial_train_dataset))
            train_size = len(initial_train_dataset) - val_size
            train_dataset, val_dataset = random_split(initial_train_dataset, [train_size, val_size])
            # train_dataset = finetune_dataset[train_dataset.indices]
            # val_dataset = finetune_dataset[val_dataset.indices]
        else:
            train_dataset = initial_train_dataset
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
        output_data = run_unsupervised_pretraining(dataloaders_dict,
                                                  model_id=model_id,
                                                  **kwargs)
        test_recon_loss_list.append(output_data['end_state_recon_loss']['test'])

    avg_recon_loss = np.mean(test_recon_loss_list)
    return avg_recon_loss


def run_unsupervised_pretraining(dataloaders,**kwargs):

    assert isinstance(dataloaders, dict)

    save_dir = kwargs.get('save_dir', None)
    device = kwargs.get('device', None)
    phase_list = kwargs.get('phase_list', None)
    yesplot = kwargs.get('yesplot', True)
    verbose = kwargs.get('verbose', False)

        # Model hyperparameters
    input_size = kwargs.get('input_size', None)
    model_kind = kwargs.get('model_kind', None)
    model_name = kwargs.get('model_name', None)
    model_id = kwargs.get('model_id',model_name)
    user_model_hyperparameters = kwargs.get('model_hyperparameters', None)
    model_eval_funcs = kwargs.get('model_eval_funcs', None)
    load_model_path = kwargs.get('load_model_path', None)

    # learning hyperparameters
    optimizer_kind = kwargs.get('optimizer_kind', 'Adam')
    learning_rate = kwargs.get('learning_rate', 1e-3)
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
        # input_size = dataloaders['train'].dataset[0][0].shape[0]
        input_size = dataloaders['train'].dataset[0].shape[0]

    if early_stopping < 0:
        early_stopping = num_epochs        

    # Instantiate the model
    model = get_model(model_kind=model_kind, input_size=input_size, **user_model_hyperparameters)

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
        'dataset_sizes': dataset_size_dct,
        'batch_sizes': batch_size_dct
    }

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    loss_history = {'train': [], 'val': []}
    patience_counter = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        if patience_counter >= early_stopping:
            model.load_state_dict(best_model_wts)
            if verbose: print('Early stopping')
            break
        for phase in ['train', 'val']:
            if phase not in dataloaders:
                continue
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for inputs in dataloaders[phase]:
                inputs = inputs.to(device)
                optimizer.zero_grad()

                # noise injection for training to make model more robust
                if (noise_factor>0) and (phase == 'train'):
                    inputs = inputs + noise_factor * torch.randn_like(inputs)

                with torch.set_grad_enabled(phase == 'train'):
                    loss = model.forward_to_loss(inputs)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            # loss_history[phase].append(running_loss / len(dataloaders[phase].dataset))
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            loss_history[phase].append(epoch_loss)

            if verbose:
                print(f'Epoch {epoch}/{num_epochs - 1}, {phase} loss: {epoch_loss}')

            # Check for early stopping
            if phase == 'val':
                if epoch < early_stopping/2:
                    continue
                if epoch_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
        if epoch % 10 == 0:
            if verbose:
                print(f'TIME elapsed: {time.time()-start_time:.2f} seconds')
                print('')
                start_time = time.time()

    # Save model state dict
    # move the model back to the cpu for evaluation
    model = model.to('cpu')            
    torch.save(model.state_dict(), model_save_path)

    # Evaluate model on training, validation, and test datasets
    end_state_losses = {}
    end_state_recon_loss = {}
    for phase in phase_list:
        model.eval()
        running_loss = 0.0
        running_recon_loss = 0.0
        with torch.inference_mode():
            for inputs in dataloaders[phase]:
                    loss = model.forward_to_loss(inputs)
                    z = model.transform(inputs)
                    recon = model.generate(z)
                    recon_loss = torch.mean((recon - inputs)**2)
            running_loss += loss.item() * inputs.size(0)
            running_recon_loss += recon_loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_recon_loss = running_recon_loss / len(dataloaders[phase].dataset)
        end_state_losses[phase] = epoch_loss
        end_state_recon_loss[phase] = epoch_recon_loss
        if verbose: print(f'{phase} loss: {epoch_loss}')

    #get the model hyperparameters
    hyperpararameters = model.get_hyperparameters()    


    # Save hyperparameters and losses to JSON file
    output_data = {
        'model_name': model_name,
        'model_kind': model_kind,
        'model_hyperparameters': hyperpararameters,
        'learning_parameters':learning_parameters,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'end_state_losses': end_state_losses,
        'end_state_recon_loss': end_state_recon_loss,
        'loss_history': loss_history,
    }

    if yesplot:
        # create a figure of size (6,4)
        plt.figure(figsize=(6,4))
        plt.plot(loss_history['train'], label='train', lw=2)
        plt.plot(loss_history['val'], label='val', lw=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.yscale('log')
        plt.legend()
        plt.title(model_name)
        # tight layout
        # plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        fig_save_name = model_name+'_loss_history.jpg'
        fig_save_name = fig_save_name.replace('_',' ')
        plt.savefig(os.path.join(save_dir, fig_save_name), dpi=300)
        plt.close()


    with open(output_save_path, 'w') as f:
        json.dump(output_data, f)

    return output_data


#############
import optuna

def objective(trial):
    hidden_size_mult =  trial.suggest_float('hidden_size_mult', 1, 2.5, step=0.1)

    search_space = {
        'model_kind': trial.suggest_categorical('model_kind', ['AE', 'VAE']),
        'model_name': 'unsupervised',
        'model_hyperparameters' :
            {
            'activation': trial.suggest_categorical('activation', ['tanh', 'leakyrelu','sigmoid']),
            'latent_size': round_to_even(trial.suggest_int('latent_size', 1, 160, log=True)),
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 0, 4),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.5, step=0.05),
            'act_on_latent_layer': True,
            },
        'early_stopping': trial.suggest_categorical('early_stopping', [-1,20]),
        'num_epochs': trial.suggest_int('num_epochs', 100, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'val_fraction': 0.15,
        'batch_size': 64,
        'noise_factor': max(trial.suggest_float('noise_factor', -0.05, 0.1, step=0.01),0),
        'verbose': False,
        'yesplot': False,
    }
    
    latent_size = search_space['model_hyperparameters']['latent_size']
    search_space['model_hyperparameters']['hidden_size'] = round_to_even(int(hidden_size_mult * latent_size))

    if search_space['early_stopping'] == -1:
        search_space['val_fraction'] = 0
    if search_space['early_stopping'] == -1:
        search_space['val_fraction'] = 0
    if search_space['model_kind'] == 'VAE':
        search_space['model_hyperparameters']['dropout_rate'] = 0

    # Run the full training process
    dropbox_dir = get_dropbox_dir()
    data_dir = os.path.join(dropbox_dir, 'development_CohortCombination','reconstruction_study_feb16')

    # create a directory to using the trial id
    datetime_start_str = trial.datetime_start.strftime('%Y%m%d_%H%M%S')
    trial_id = datetime_start_str + '_' + str(trial.number).zfill(4) 
    save_dir = os.path.join(data_dir, 'models_feb16', trial_id)
    search_space['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    # save the search space to a json file
    with open(os.path.join(save_dir, 'search_space.json'), 'w') as f:
        json.dump(search_space, f, indent=4)
        
    avg_recon_loss = unsupervised_pretraining_wrapper(data_dir, **search_space)


    print('average recon loss:', avg_recon_loss)
    return avg_recon_loss





if __name__ == '__main__':
        

    import logging
    import sys

    # Set up logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'unsupervised_feb19'
    storage_name = 'sqlite:///{}.db'.format(study_name)

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=100)


    # dropbox_dir = get_dropbox_dir()
    # data_dir = os.path.join(dropbox_dir, 'development_CohortCombination','reconstruction_study_feb16')
    # model_kind = 'AE'
    # model_name = 'AE'
    # save_dir = os.path.join(data_dir,'models_feb16')
    # os.makedirs(save_dir, exist_ok=True)

    # hyperparameters = {
    #     'latent_size': 8,
    #     'hidden_size': 16,
    #     'num_hidden_layers': 1,
    #     'activation': 'sigmoid',
    #     'dropout_rate': 0.1,
    #     'use_batch_norm': True,
    #     'act_on_latent_layer': True
    # }
    
    # unsupervised_pretraining_wrapper(data_dir,
    #                             model_kind=model_kind,
    #                             model_name=model_name,
    #                             model_hyperparameters=hyperparameters,
    #                             save_dir=save_dir,
    #                             early_stopping=-1,
    #                             num_epochs=100,
    #                             batch_size=64,
    #                             learning_rate=1e-3,
    #                             yesplot=True,
    #                             verbose=True)
    
