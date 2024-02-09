import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
import json
import os
import matplotlib.pyplot as plt
from models import VAE, AE  # Assuming models.py contains a VAE class
import time


def run_train_autoencoder(dataloaders,save_dir,**kwargs):

    # check that the dataloaders are in the correct format
    # should be a dictionary with train, val, and test keys
    assert isinstance(dataloaders, dict)

    # check if the device is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
        # device = torch.device("mps")
    else:
        device = torch.device("cpu")
    #Note: MPS and CUDA only faster when batch size is large, which means need large dataset
    

    # Model hyperparameters
    input_size = kwargs.get('input_size', None)
    hidden_size = kwargs.get('hidden_size', 64)
    latent_size = kwargs.get('latent_size', 32)
    num_hidden_layers = kwargs.get('num_hidden_layers', 1)
    dropout_rate = kwargs.get('dropout_rate', 0)
    activation = kwargs.get('activation', 'leakyrelu')
    use_batch_norm = kwargs.get('use_batch_norm', False)

    # learning Hyperparameters
    num_epochs = kwargs.get('num_epochs', 500)
    learning_rate = kwargs.get('learning_rate', .0001)
    early_stopping_patience = kwargs.get('early_stopping_patience', 50)

    yesplot = kwargs.get('yesplot', True)
    load_existing_model =  kwargs.get('load_existing_model', True)
    model_name = kwargs.get('model_name', 'AE')
    model_kind = kwargs.get('model_kind', 'AE')
    model_save_path = os.path.join(save_dir, model_name+'_model.pth')
    output_save_path = os.path.join(save_dir, model_name+'_output.json')

    
    if input_size is None:
        input_size = dataloaders['train'].dataset[0].shape[0]
    
    dataset_size = len(dataloaders['train'].dataset)
    batch_size = dataloaders['train'].batch_size
    
    if (batch_size < 32):
        # if dataset is small, then use the cpu
        # I guess the evalaation doesn't need to be done on the gpu, but moving back and forth is slow
        device = torch.device("cpu")

    learning_parameters = {
        # 'batch_size': 32,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience
        }

    os.makedirs(save_dir, exist_ok=True)
    # Initialize model, criterion, and optimizer
    if model_kind.lower() in ['vae']:
        model = VAE(input_size, hidden_size, latent_size, num_hidden_layers, dropout_rate,activation,use_batch_norm)
        print('use Variational Autoencoder (VAE)')
    else:
        model = AE(input_size, hidden_size,latent_size, num_hidden_layers, dropout_rate,activation,use_batch_norm)
        print('use Basic Autoencoder (AE)')
    
    if (load_existing_model) and (os.path.exists(model_save_path)):
        print('found existing model, loading it')
        if os.path.exists(output_save_path):
            output_data = json.load(open(output_save_path))
            return output_data
        else:
            model.load_state_dict(torch.load(model_save_path))
            num_epochs = 0
        
        

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    loss_history = {'train': [], 'val': []}
    patience_counter = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        if patience_counter >= early_stopping_patience:
            model.load_state_dict(best_model_wts)
            print('Early stopping')
            break
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for inputs in dataloaders[phase]:
                inputs = inputs.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss = model.forward_to_loss(inputs)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            # loss_history[phase].append(running_loss / len(dataloaders[phase].dataset))
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            loss_history[phase].append(epoch_loss)

            print(f'Epoch {epoch}/{num_epochs - 1}, {phase} loss: {epoch_loss}')

            # Check for early stopping
            if phase == 'val':
                if epoch < early_stopping_patience/2:
                    continue
                if epoch_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
        if epoch % 10 == 0:
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
    for phase in ['train', 'val', 'test']:
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
        print(f'{phase} loss: {epoch_loss}')

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
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, model_name+'_loss_history.png'))
        plt.close()


    with open(output_save_path, 'w') as f:
        json.dump(output_data, f)

    return output_data

###########################
# %% Run the training
###########################

from prep import PreTrainingDataset

if __name__ == '__main__':
    # Example usage
    # dataloaders = {
    #     'train': train_loader,
    #     'val': val_loader,
    #     'test': test_loader
    # }
    # model = run_train_autoencoder(dataloaders,save_dir,**kwargs)
    

    base_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton/development_CohortCombination'
    if not os.path.exists(base_dir):
        base_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination'

    date_name = 'hilic_pos_2024_feb_05_read_norm_poolmap'
    feat_subset_name = 'num_cohorts_thresh_0.5'
    study_subset_name= 'subset all_studies with align score 0.25 from Merge_Jan25_align_80_40_fillna_avg'
    # task_name = 'std_1_Benefit'
    task_name = 'std_1_Multi'
    input_dir = f'{base_dir}/{date_name}/{study_subset_name}/{feat_subset_name}/{task_name}'
    save_dir = os.path.join(input_dir, 'pretrain_models_feb07')
    os.makedirs(save_dir, exist_ok=True)
    
    
    # check of the train_dataset.pth exists
    if os.path.exists(os.path.join(save_dir, 'train_dataset.pth')):
        train_dataset = torch.load(os.path.join(save_dir, 'train_dataset.pth'))
        val_dataset = torch.load(os.path.join(save_dir, 'val_dataset.pth'))
        test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pth'))

    else:
        dataset = PreTrainingDataset(input_dir)
        # create dataloaders for the train, val, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        # save the datasets to a file
        torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pth'))
        torch.save(val_dataset, os.path.join(save_dir, 'val_dataset.pth'))
        torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pth'))


    print('size of training:', len(train_dataset))
    print('size of validation:', len(val_dataset))
    print('size of testing:', len(test_dataset))

    num_workers = 0
    batch_size = 256
    dataloaders =  {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    # use_batch_norm = False

    for model_kind in ['VAE','AE']:
        for hidden_size_mult in [1,2]:
            for latent_size in [64,128,256]:
                # for dropout_rate in [0,0.25]:
                for use_batch_norm in [True,False]:
                    for activation in ['sigmoid']:
                        for num_hidden_layers in [0,1]:
                            
                            if num_hidden_layers == 0:
                                hidden_size = latent_size
                            else:
                                hidden_size = hidden_size_mult*latent_size

                            if model_kind.lower() == 'vae':
                                dropout_rate = 0
                            else:
                                dropout_rate = 0.25

                            model_name = '{}_layers{}_hidden{}_latent{}_'.format(model_kind,num_hidden_layers, hidden_size, latent_size)
                            # model_name = f'x{model_kind}_H{hidden_size}_{num_hidden_layers}_L{latent_size}_D{dropout_rate}_A{activation}'

                            if use_batch_norm:
                                model_name += '_BN'

                            print(f'running {model_name}')
                            if num_hidden_layers == 0:
                                hidden_size0 = 0
                            else:
                                hidden_size0 = hidden_size

                            _ = run_train_autoencoder(dataloaders,save_dir,
                                                        model_name=model_name,
                                                        model_kind=model_kind,
                                                        hidden_size=hidden_size0,
                                                        latent_size=latent_size,
                                                        dropout_rate=dropout_rate,
                                                        activation=activation,
                                                        num_hidden_layers=num_hidden_layers,
                                                        use_batch_norm=use_batch_norm,
                                                        )


