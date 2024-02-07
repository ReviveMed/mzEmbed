import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from models import BinaryClassifier, MultiClassClassifier
from models import VAE, AE  # Assuming models.py contains a VAE class

from torchmetrics import Accuracy, AUROC


def run_train_classifier(dataloaders,save_dir,**kwargs):

    assert isinstance(dataloaders, dict)
    
    # Model hyperparameters
    input_size = kwargs.get('input_size', None)
    head__hidden_size = kwargs.get('hidden_size', 64)
    latent_size = kwargs.get('latent_size', 32)
    head__activation = kwargs.get('activation', 'leakyrelu')
    num_classes = kwargs.get('num_classes', None)
    head__num_hidden_layers = kwargs.get('num_hidden_layers', 1)
    head__dropout_rate = kwargs.get('dropout_rate', 0)
    class_weight = kwargs.get('class_weights', None)
    phase_list = kwargs.get('phase_list', None)


    encoder__hidden_size = kwargs.get('encoder_hidden_size', 64)
    encoder__num_hidden_layers = kwargs.get('encoder_num_hidden_layers', 1)
    encoder__dropout_rate = kwargs.get('encoder_dropout_rate', 0)

    # learning Hyperparameters
    num_epochs = kwargs.get('num_epochs', 200)
    learning_rate = kwargs.get('learning_rate', .0001)
    early_stopping_patience = kwargs.get('early_stopping_patience', 20)

    yesplot = kwargs.get('yesplot', True)
    how_average = kwargs.get('how_average', 'weighted')
    load_existing_model =  kwargs.get('load_existing_model', True)
    

    model_name = kwargs.get('model_name', 'Classifier')
    model_kind = kwargs.get('model_kind', 'Classifier')

    encoder_name = kwargs.get('encoder_name', 'BasicEncoder')
    encoder_kind = kwargs.get('encoder_kind', 'AE')
    encoder_status = kwargs.get('encoder_status', 'finetune')

    model_save_path = os.path.join(save_dir, model_name+'_'+encoder_status+'_'+encoder_name+'_head-model.pth')
    output_save_path = os.path.join(save_dir, model_name+'_'+encoder_status+'_'+encoder_name+'_output.json')
    pretrained_encoder_load_path = kwargs.get('pretrained_encoder_load_path', None)
    finetuned_encoder_save_path = os.path.join(save_dir, model_name+'_'+encoder_status+'_'+encoder_name+'_encoder-model.pth')
    
    if phase_list is None:
        phase_list = list(dataloaders.keys())

    if input_size is None:
        input_size = dataloaders['train'].dataset[0][0].shape[0]

    if num_classes is None:
        y_train = dataloaders['train'].dataset[:][1]
        num_classes = len(torch.unique(y_train))

    if (class_weight is not None) and (len(class_weight) == num_classes):
        class_weight = torch.tensor(class_weight, dtype=torch.float32)
    elif class_weight is not None:
        y_train = dataloaders['train'].dataset[:][1]
        class_weight = 1 / torch.bincount(y_train)
    else:
        class_weight = None


    learning_parameters = {
        # 'batch_size': 32,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience
        }

    os.makedirs(save_dir, exist_ok=True)
    # Initialize model, criterion, and optimizer
    if num_classes==2:
        model = BinaryClassifier(latent_size, head__hidden_size, head__num_hidden_layers, head__dropout_rate,
                                 activation=head__activation)
        print('use Binary Classifier')
        task = 'binary'
    elif num_classes > 2:
        model = MultiClassClassifier(latent_size, head__hidden_size, num_classes, head__num_hidden_layers, head__dropout_rate,
                                        activation=head__activation)
        print('use Multi Classifier')
        task = 'multi'
    
    model.define_loss(class_weight=class_weight)


    if encoder_kind.lower() in ['vae']:
        encoder_model = VAE(input_size, encoder__hidden_size, latent_size, encoder__num_hidden_layers, encoder__dropout_rate)
        print('use Variational Autoencoder (VAE)')
    elif encoder_kind.lower() in ['ae']:
        encoder_model = AE(input_size, encoder__hidden_size, latent_size, encoder__num_hidden_layers, encoder__dropout_rate)
        print('use Basic Autoencoder (AE)')
    else:
        raise ValueError(f'Encoder kind not recognized: {encoder_kind}')

    if pretrained_encoder_load_path is not None:
        if not os.path.exists(pretrained_encoder_load_path):
            raise ValueError(f'Pre-trained model not found at {pretrained_encoder_load_path}')
        
        pre_trained_output_path = pretrained_encoder_load_path.replace('_model.pth', '_output.json')
        if not os.path.exists(pre_trained_output_path):
            raise ValueError(f'Pre-trained model output not found at {pre_trained_output_path}')

        print('using pre-trained model for encoder')    
        with open(pre_trained_output_path, 'r') as f:
            pre_trained_output = json.load(f)
        
        pretrained_encoder_name = pre_trained_output['model_name']
        pretrained_encoder_kind = pre_trained_output['model_kind']
        assert encoder_kind.lower() == pretrained_encoder_kind.lower(), f'Encoder kind mismatch: {encoder_kind} vs {pretrained_encoder_kind}'
        assert encoder_name.lower() == pretrained_encoder_name.lower(), f'Encoder name mismatch: {encoder_name} vs {pretrained_encoder_name}'
        assert latent_size == pre_trained_output['model_hyperparameters']['latent_size'], f'Latent size mismatch: {latent_size} vs {pre_trained_output["model_hyperparameters"]["latent_size"]}'
        assert input_size == pre_trained_output['model_hyperparameters']['input_size'], f'Input size mismatch: {input_size} vs {pre_trained_output["model_hyperparameters"]["input_size"]}'
        assert encoder__hidden_size == pre_trained_output['model_hyperparameters']['hidden_size'], f'Hidden size mismatch: {encoder__hidden_size} vs {pre_trained_output["model_hyperparameters"]["hidden_size"]}'
        assert encoder__num_hidden_layers == pre_trained_output['model_hyperparameters']['num_hidden_layers'], f'Number of hidden layers mismatch: {encoder__num_hidden_layers} vs {pre_trained_output["model_hyperparameters"]["num_hidden_layers"]}'

        if (encoder_status.lower() == 'finetune') or (encoder_status.lower() == 'fixed'):
            encoder_model.load_state_dict(torch.load(pretrained_encoder_load_path))
            print('Pre-trained model loaded from:', pretrained_encoder_load_path)
        elif encoder_status.lower() == 'random':
            print('Randomly initializing the encoder')



    # set the encoder to allow finetuning if not fixed
    if (encoder_status.lower() == 'finetune') or (encoder_status.lower() == 'random'):
        for param in encoder_model.parameters():
            param.requires_grad = True
    elif encoder_status.lower() == 'fixed':
        for param in encoder_model.parameters():
            param.requires_grad = False


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if (load_existing_model) and (os.path.exists(model_save_path)):
        model.load_state_dict(torch.load(model_save_path))
        encoder_model.load_state_dict(torch.load(finetuned_encoder_save_path))
        # load the json output data
        with open(output_save_path, 'r') as f:
            output_data = json.load(f)

        print('found existing model, loading it')
        # TODO: add option to further train the existing model
        return output_data
    

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    loss_history = {'train': [], 'val': []}
    acc_history = {'train': [], 'val': []}
    auroc_history = {'train': [], 'val': []}
    patience_counter = 0

    running_accuracy = Accuracy(task=task,average=how_average)
    running_auroc = AUROC(task=task,average=how_average)
    # val_accuracy = Accuracy(task=task,average=how_average)
    # val_auroc = AUROC(task=task,average=how_average)

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
            running_auroc.reset()
            running_accuracy.reset()

            for data in dataloaders[phase]:
                X, y = data
                y = y.view(-1,1)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    X_encoded = encoder_model.transform(X)
                    outputs = model(X_encoded)
                    loss = model.loss(outputs, y)
                    # # outputs_probs = model.predict_proba(outputs)
                    outputs_probs = model.logits_to_proba(outputs)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_accuracy(outputs_probs, y)
                running_auroc(outputs_probs, y)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            loss_history[phase].append(epoch_loss)
            epoch_accuracy = running_accuracy.compute().item()
            epoch_auroc = running_auroc.compute().item()
            acc_history[phase].append(epoch_accuracy)
            auroc_history[phase].append(epoch_auroc)
            print(f'Epoch {epoch}/{num_epochs - 1}, {phase} loss: {epoch_loss:.6f}, \
                  acc: {epoch_accuracy:.4f}, auroc: {epoch_auroc:.4f}')

            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1


    # Save model state dict
    torch.save(model.state_dict(), model_save_path)
    torch.save(encoder_model.state_dict(), finetuned_encoder_save_path)



    # Evaluate model on training, validation, and test datasets
    phase_sizes = {phase: len(dataloaders[phase].dataset) for phase in phase_list}
    end_state_losses = {}
    end_state_acc = {}
    end_state_auroc = {}
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
            print(f'End state {phase} loss: {epoch_loss:.6f}, acc: {epoch_accuracy:.4f}, auroc: {epoch_auroc:.4f}')

    encoder_hyperparameters = encoder_model.get_hyperparameters()
    model_hyperparameters = model.get_hyperparameters()

    output_data = {
            'model_name': model_name,
            'model_kind': model_kind,
            'encoder_name': encoder_name,
            'encoder_kind': encoder_kind,
            'encoder_status': encoder_status,
            'encoder_pretrained_path': pretrained_encoder_load_path,
            'model_hyperparameters': model_hyperparameters,
            'encoder_hyperparameters': encoder_hyperparameters,
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

    with open(output_save_path, 'w') as f:
        json.dump(output_data, f)

    if yesplot:
        for history, name in zip([loss_history, acc_history, auroc_history], ['loss', 'acc', 'auroc']):
            plt.figure(figsize=(6,4))
            plt.plot(history['train'], label='train', lw=2)
            plt.plot(history['val'], label='val', lw=2)
            plt.title(f'{model_name} with {encoder_name}')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, model_name+'_'+encoder_name+'_' +name+'.png'))
            plt.close()

    return output_data

###########################
# %% Run the training
###########################

from prep import ClassifierDataset

if __name__ == '__main__':

    # Example usage
    # dataloaders = {
    #     'train': train_loader,
    #     'val': val_loader,
    #     'test': test_loader
    # }
    # run_train_classifier(dataloaders,save_dir,**kwargs)
    # pass

    base_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton/development_CohortCombination'
    if not os.path.exists(base_dir):
        base_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination'

    date_name = 'hilic_pos_2024_feb_02_read_norm'
    feat_subset_name = 'num_cohorts_thresh_0.5'
    study_subset_name= 'subset all_studies with align score 0.3 from Merge_Jan25_align_80_40_fillna_avg'
    task_name = 'std_1_Benefit'
    input_dir = f'{base_dir}/{date_name}/{study_subset_name}/{feat_subset_name}/{task_name}'
    save_dir = os.path.join(input_dir, 'fine_tuned')
    os.makedirs(save_dir, exist_ok=True)

    label_mapper = {'CB': 1.0, 'NCB': 0.0, 'ICB': np.nan}
    pretrained_encoder_load_path = os.path.join(input_dir, 'vae_model', 'VAE_model.pth')
    # pretrained_encoder_load_path = os.path.join(input_dir, 'vae_model', 'AE_model.pth')
    # pretrained_encoder_load_path = None

    if os.path.exists(os.path.join(save_dir, 'test_alt_dataset.pth')):
        train_dataset = torch.load(os.path.join(save_dir, 'train_dataset.pth'))
        val_dataset = torch.load(os.path.join(save_dir, 'val_dataset.pth'))
        test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pth'))
        test_alt_dataset = torch.load(os.path.join(save_dir, 'test_alt_dataset.pth'))
    else:
        # train_dataset = ClassifierDataset(input_dir,subset='train')
        # label_encoder = train_dataset.get_label_encoder()
        # val_dataset = ClassifierDataset(input_dir,subset='val',label_encoder=label_encoder)
        # test_dataset = ClassifierDataset(input_dir,subset='test',label_encoder=label_encoder)

        train_dataset = ClassifierDataset(input_dir,subset='train',label_encoder=label_mapper)
        val_dataset = ClassifierDataset(input_dir,subset='val',label_encoder=label_mapper)
        test_dataset = ClassifierDataset(input_dir,subset='test',label_encoder=label_mapper)
        test_alt_dataset = ClassifierDataset(input_dir,subset='test',label_encoder=label_mapper)

        # torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pth'))
        # torch.save(val_dataset, os.path.join(save_dir, 'val_dataset.pth'))
        # torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pth'))
        # torch.save(test_alt_dataset, os.path.join(save_dir, 'test_alt_dataset.pth'))

    print('size of training:', len(train_dataset))
    print('size of validation:', len(val_dataset))
    print('size of test:', len(test_dataset))
    print('size of test_alt:', len(test_alt_dataset))

    batch_size = 32
    dataloaders= {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        'test_alt': DataLoader(test_alt_dataset, batch_size=batch_size, shuffle=False)
    }

    model = run_train_classifier(dataloaders,save_dir,
                                 pretrained_encoder_load_path=pretrained_encoder_load_path,
                                model_name='BenefitClassifier4',
                                model_kind='BinaryClassifier',
                                phase_list=['train', 'val', 'test', 'test_alt'],
                                encoder_name='VAE',
                                encoder_kind='VAE',
                                dropout_rate=0.5,
                                activation='sigmoid')
    
                                 
