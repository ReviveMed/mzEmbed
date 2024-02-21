# %%
import torch
from misc import normalize_loss
import json
import os
import torch.nn.functional as F
from models import AE, MultiClassClassifier, BinaryClassifier
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

def train_semisupervised(dataloaders,encoder,head,adversary,**kwargs):

    learning_rate = kwargs.get('learning_rate', 0.001)
    num_epochs = kwargs.get('num_epochs', 10)
    encoder_weight = kwargs.get('encoder_weight', 1)
    head_weight = kwargs.get('head_weight', 1)
    adversary_weight = kwargs.get('adversary_weight', 1)
    early_stopping_patience = kwargs.get('early_stopping_patience', -1)
    noise_factor = kwargs.get('noise_factor', 0)
    phase_list = kwargs.get('phase_list', None)
    # head_info = kwargs.get('head_info', None)
    # adversary_info = kwargs.get('adversary_info', None)
    verbose = kwargs.get('verbose', True)
    save_dir = kwargs.get('save_dir', None)
    end_state_eval_funcs = kwargs.get('end_state_eval_funcs', {})

    if save_dir is not None:
        save_trained_model = True
        encoder_save_path = os.path.join(save_dir, 'encoder.pth')
        head_save_path = os.path.join(save_dir, 'head.pth')
        adversary_save_path = os.path.join(save_dir, 'adversary.pth')
        output_save_path = os.path.join(save_dir, 'output.json')
    else:
        save_trained_model = False
    

    if phase_list is None:
        phase_list = list(dataloaders.keys())
    

    dataset_size_dct = {phase: len(dataloaders[phase].dataset) for phase in phase_list}
    batch_size_dct = {phase: dataloaders[phase].batch_size for phase in phase_list}
    
    # if head_info is None:
    #     num_head_classes = head.num_classes
    #     head_class_weight = head.class_weight
    #     head_info = {'class_weight': head_class_weight, 'num_classes': num_head_classes}
    # if adversary_info is None:
    #     num_adversary_classes = adversary.num_classes
    #     adversary_class_weight = adversary.class_weight
    #     adversary_info = {'class_weight': adversary_class_weight, 'num_classes': num_adversary_classes}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    learning_parameters = {
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'noise_factor': noise_factor,
        'encoder_weight': encoder_weight,
        'head_weight': head_weight,
        'adversary_weight': adversary_weight,
        'phase_list': phase_list,
        'phase_sizes': dataset_size_dct,
        'batch_sizes': batch_size_dct,
    }

    if early_stopping_patience < 0:
        early_stopping_patience = num_epochs

    # define the loss functions for the head and adversary
        # this should be done ahead of time
    # head.define_loss(**head_info)
    # adversary.define_loss(**adversary_info)

    # define the losses averages across the epochs
    encoder_loss_avg = 1
    head_loss_avg = 1
    adversary_loss_avg = 1

    encoder.to(device)
    head.to(device)
    adversary.to(device)

    # define the optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    head_optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    adversary_optimizer = torch.optim.Adam(adversary.parameters(), lr=learning_rate)

    # define the loss history and best params with associated losses
    loss_history = {'train': [], 'val': []}
    encoder_loss_history = {'train': [], 'val': []}
    head_loss_history = {'train': [], 'val': []}
    adversary_loss_history = {'train': [], 'val': []}
    best_loss = {'encoder': 1e10, 'head': 1e10, 'adversary': 1e10, 'joint': 1e10, 'epoch': 0}
    best_wts = {'encoder': encoder.state_dict(), 'head': head.state_dict(), 'adversary': adversary.state_dict()}
    eval_history = {'train': {}, 'val': {}}
    patience_counter = 0

    # start the training loop
    for epoch in range(num_epochs):
        if patience_counter > early_stopping_patience:
            print('Early stopping at epoch', epoch)
            encoder.load_state_dict(best_wts['encoder'])
            head.load_state_dict(best_wts['head'])
            adversary.load_state_dict(best_wts['adversary'])
            break


        for phase in ['train', 'val']:
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
            # running_outputs = {'y_head': [], 'y_adversary': []}
            epoch_head_outputs = torch.tensor([])
            epoch_adversary_outputs = torch.tensor([])
            epoch_head_targets = torch.tensor([])
            epoch_adversary_targets = torch.tensor([])
            num_batches = len(dataloaders[phase])

            for batch_idx, data in enumerate(dataloaders[phase]):
                X, y_head, y_adversary = data
                X = X.to(device)
                y_head = y_head.to(device)
                y_adversary = y_adversary.to(device)

                # noise injection for training to make model more robust
                if (noise_factor>0) and (phase == 'train'):
                    X = X + noise_factor * torch.randn_like(X)

                encoder_optimizer.zero_grad()
                head_optimizer.zero_grad()
                adversary_optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    z, encoder_loss = encoder.transform_with_loss(X)
                    
                    y_head_output = head(z)
                    head_loss = head.loss(y_head_output, y_head)

                    z2 = z.detach()
                    z2.requires_grad = True

                    y_adversary_output = adversary(z2)
                    adversary_loss = adversary.loss(y_adversary_output, y_adversary)

                    # running_outputs['y_head'].append(y_head_output)
                    # running_outputs['y_adversary'].append(y_adversary_output)
                    epoch_head_outputs = torch.cat((epoch_head_outputs, y_head_output), 0)
                    epoch_adversary_outputs = torch.cat((epoch_adversary_outputs, y_adversary_output), 0)
                    epoch_head_targets = torch.cat((epoch_head_targets, y_head), 0)
                    epoch_adversary_targets = torch.cat((epoch_adversary_targets, y_adversary), 0)
                    
                    curr_batch = num_batches*epoch + batch_idx
                    if phase == 'train':

                        encoder_loss, encoder_loss_avg = normalize_loss(
                            encoder_loss, encoder_loss_avg, 0, curr_batch)

                        head_loss, head_loss_avg = normalize_loss(
                            head_loss, head_loss_avg, 0, curr_batch)

                        adversary_loss, adversary_loss_avg = normalize_loss(
                            adversary_loss, adversary_loss_avg, 0, curr_batch)

                        joint_loss = (encoder_weight*encoder_loss + \
                            head_weight*head_loss - \
                            adversary_weight*adversary_loss)
                        joint_loss.backward(retain_graph=True)
                        
                        encoder_optimizer.step()
                        head_optimizer.step()

                        # zero the adversarial optimizer again
                        adversary_optimizer.zero_grad()

                        # Backward pass and optimize the adversarial classifiers
                        adversary_loss.backward()
                        adversary_optimizer.step()

                    elif phase == 'val':

                        encoder_loss, _ = normalize_loss(
                            encoder_loss, encoder_loss_avg, 0, curr_batch)
                        
                        head_loss, _ = normalize_loss(
                            head_loss, head_loss_avg, 0, curr_batch)
                        
                        adversary_loss, _ = normalize_loss(
                            adversary_loss, adversary_loss_avg, 0, curr_batch)
                        
                        joint_loss = (encoder_weight*encoder_loss + \
                            head_weight*head_loss - \
                            adversary_weight*adversary_loss)
                    

                    running_losses['encoder'] += encoder_loss.item()
                    running_losses['head'] += head_loss.item()
                    running_losses['adversary'] += adversary_loss.item()
                    running_losses['joint'] += joint_loss.item()

            loss_history[phase].append(running_losses['joint']/len(dataloaders[phase]))
            encoder_loss_history[phase].append(running_losses['encoder']/len(dataloaders[phase]))
            head_loss_history[phase].append(running_losses['head']/len(dataloaders[phase]))
            adversary_loss_history[phase].append(running_losses['adversary']/len(dataloaders[phase]))

            for eval_name, eval_func in end_state_eval_funcs.items():
                if eval_name not in eval_history[phase]:
                    eval_history[phase][eval_name] = []
                    
                eval_history[phase][eval_name].append(eval_func(head_ouputs = epoch_head_outputs,
                                                    head_targets = epoch_head_targets,
                                                    adversary_outputs = epoch_adversary_outputs,
                                                    adversary_targets = epoch_adversary_targets,
                                                    head = head,
                                                    adversary = adversary))

            if (verbose) and (epoch % 10 == 0):
                print(f'Epoch [{epoch+1}/{num_epochs}], {phase} Loss: {loss_history[phase][-1]:.4f}')
                for eval_name, eval_list in eval_history[phase].items():
                    print(f'{phase} {eval_name} {eval_list[-1]}')

            if phase == 'val':
                if loss_history[phase][-1] < best_loss['joint']:
                    best_loss['joint'] = loss_history[phase][-1]
                    best_loss['encoder'] = encoder_loss_history[phase][-1]
                    best_loss['head'] = head_loss_history[phase][-1]
                    best_loss['adversary'] = adversary_loss_history[phase][-1]
                    best_loss['epoch'] = epoch
                    best_wts['encoder'] = encoder.state_dict()
                    best_wts['head'] = head.state_dict()
                    best_wts['adversary'] = adversary.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1


    if save_trained_model:
        torch.save(encoder.state_dict(), encoder_save_path)
        torch.save(head.state_dict(), head_save_path)
        torch.save(adversary.state_dict(), adversary_save_path)

    if best_loss['epoch'] < 0:
        best_loss['epoch'] = epoch

    # evaluate the model on datasets
    end_state_losses = {}
    end_state_eval = {}
    encoder.to('cpu')
    head.to('cpu')
    adversary.to('cpu')
    for phase in phase_list:
        if phase not in dataloaders:
            continue
        encoder.eval()
        head.eval()
        adversary.eval()
        recon_loss = 0.0
        head_loss = 0.0
        adversary_loss = 0.0
        head_ouputs = torch.tensor([])
        adversary_outputs = torch.tensor([])
        head_targets = torch.tensor([])
        adversary_targets = torch.tensor([])
        with torch.inference_mode():
            for data in dataloaders[phase]:
                X, y_head, y_adversary = data
                X = X
                y_head = y_head
                y_adversary = y_adversary
                z = encoder.transform(X)
                X_recon = encoder.generate(z)
                recon_loss += F.mse_loss(X_recon, X)* X.size(0)
                y_head_output = head(z)
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
            for eval_name, eval_func in end_state_eval_funcs.items():
                end_state_eval[phase][eval_name] = eval_func(head_ouputs=head_ouputs,
                                                head_targets=head_targets,
                                                adversary_outputs=adversary_outputs,
                                                adversary_targets=adversary_targets,
                                                head=head,
                                                adversary=adversary)

            if verbose:
                print(f'End state {phase} Losses: {end_state_losses[phase]}')


    encoder_hyperparams = encoder.get_hyperparameters()
    head_hyperparams = head.get_hyperparameters()
    adversary_hyperparams = adversary.get_hyperparameters()

    output_data = {
        'learning_parameters': learning_parameters,
        'end_state_eval': end_state_eval,
        'encoder': {
            # 'hyperparameters': encoder_hyperparams,
            'loss_history': encoder_loss_history,
            'end_state_losses': end_state_losses,
            'best_loss': best_loss,
            # 'best_wts': best_wts,
            'loss_avg': encoder_loss_avg,
        },
        'head': {
            # 'hyperparameters': head_hyperparams,
            'loss_history': head_loss_history,
            'end_state_losses': end_state_losses,
            'best_loss': best_loss,
            # 'best_wts': best_wts,
            'loss_avg': head_loss_avg,
        },
        'adversary': {
            # 'hyperparameters': adversary_hyperparams,
            'loss_history': adversary_loss_history,
            'end_state_losses': end_state_losses,
            'best_loss': best_loss,
            # 'best_wts': best_wts,
            'loss_avg': adversary_loss_avg,
        },
    }

    # with open(output_save_path, 'w') as f:
    #     json.dump(output_data, f, indent=4)

    return encoder, head, adversary, output_data


# %%
# if __name__ == "__main__":
test_adversarial_dir = '/Users/jonaheaton/rcc4_mzlearn4-5/test_adversarial'
output_dir = os.path.join(test_adversarial_dir, 'adversarial_network_Feb21_1')
os.makedirs(output_dir, exist_ok=True)
batch_size = 64
num_epochs = 100


# create data loaders
class CustomDataset(Dataset):
    def __init__(self, X, y_benefit, y_gend):
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y_benefit = torch.tensor(y_benefit.to_numpy(), dtype=torch.float32)
        self.y_gend = torch.tensor(y_gend.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_benefit[idx], self.y_gend[idx]

    def get_num_classes_y1(self):
        return len(torch.unique(self.y_benefit))
    
    def get_num_classes_y2(self):
        return len(torch.unique(self.y_gend))
    
    def get_class_weights_y1(self):
        y_benefit_no_nan = self.y_benefit[~torch.isnan(self.y_benefit)]
        y_benefit_int = y_benefit_no_nan.int()
        return 1/torch.bincount(y_benefit_int)
    
    def get_class_weights_y2(self):
        y_gend_no_nan = self.y_gend[~torch.isnan(self.y_gend)]
        y_gend_int = y_gend_no_nan.int()
        return 1/torch.bincount(y_gend_int)

# %%
# Load the data
X_train = pd.read_csv(os.path.join(test_adversarial_dir, 'X_train.csv'),index_col=0)
y_benefit_train = pd.read_csv(os.path.join(test_adversarial_dir, 'y_train.csv'),index_col=0)
y_gend_train = pd.read_csv(os.path.join(test_adversarial_dir, 'sex_train.csv'),index_col=0)

# create a validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_benefit_train, y_benefit_val = train_test_split(X_train, y_benefit_train, test_size=0.2, random_state=42,
                                                                stratify=y_benefit_train)

y_gend_val =  y_gend_train.loc[X_val.index]
y_gend_train = y_gend_train.loc[X_train.index]


# create a dataloader
train_dataset = CustomDataset(X_train, y_benefit_train, y_gend_train)

# %%
num_classes_benefit = train_dataset.get_num_classes_y1()
num_classes_gend = train_dataset.get_num_classes_y2()
weights_benefit = train_dataset.get_class_weights_y1()
weights_gend = train_dataset.get_class_weights_y2()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(X_val, y_benefit_val, y_gend_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# %%

X_test = pd.read_csv(os.path.join(test_adversarial_dir, 'X_test.csv'),index_col=0)
y_benefit_test = pd.read_csv(os.path.join(test_adversarial_dir, 'y_test.csv'),index_col=0)
y_gend_test = pd.read_csv(os.path.join(test_adversarial_dir, 'sex_test.csv'),index_col=0)

# create a dataloader
test_dataset = CustomDataset(X_test, y_benefit_test, y_gend_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# %%
# create the models
input_dim = X_train.shape[1]                                      
latent_dim = 16
encoder = AE(input_size=input_dim, latent_size=latent_dim, hidden_size=32, num_hidden_layers=2, 
                dropout_rate=0.2,use_batch_norm=True,act_on_latent_layer=True)
head = BinaryClassifier(latent_dim, hidden_size=4, num_hidden_layers=2)
adversary = BinaryClassifier(latent_dim, hidden_size=4, num_hidden_layers=2)

head_info = {'class_weight': weights_benefit, 'num_classes': num_classes_benefit}
adversary_info = {'class_weight': weights_gend, 'num_classes': num_classes_gend}

head.define_loss(class_weight=weights_benefit)
adversary.define_loss(class_weight=weights_gend)


# %%
dataloader_dct = {'train': train_loader, 'val': val_loader, 'test': test_loader}

# %% create End-state evaluation functions

from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score


def evaluate_head_accuracy(head_ouputs, head_targets, adversary_outputs, adversary_targets, head, adversary):
    head_probs = head.logits_to_proba(head_ouputs.detach()).numpy()
    head_preds = np.round(head_probs)
    head_acc = balanced_accuracy_score(head_targets, head_preds)
    return head_acc

def evaluate_adversary_accuracy(head_ouputs, head_targets, adversary_outputs, adversary_targets,head, adversary):
    adversary_probs = adversary.logits_to_proba(adversary_outputs.detach()).numpy()
    adversary_preds = np.round(adversary_probs)
    adversary_acc = balanced_accuracy_score(adversary_targets, adversary_preds)
    return adversary_acc

def evaluate_head_auc(head_ouputs, head_targets, adversary_outputs, adversary_targets,head, adversary):
    head_probs = head.logits_to_proba(head_ouputs.detach()).numpy()
    head_auc = roc_auc_score(head_targets, head_probs,
                             average='weighted')
    return head_auc

def evaluate_adversary_auc(head_ouputs, head_targets, adversary_outputs, adversary_targets,head, adversary):
    adversary_probs = adversary.logits_to_proba(adversary_outputs.detach()).numpy()
    adversary_auc = roc_auc_score(adversary_targets, adversary_probs,
                                  average='weighted')
    return adversary_auc

end_state_eval_funcs = {
    'head_accuracy': evaluate_head_accuracy,
    'adversary_accuracy': evaluate_adversary_accuracy,
    'head_auc': evaluate_head_auc,
    'adversary_auc': evaluate_adversary_auc
}




# %%

# TODO: allow adversary to have extra training
# TODO: retrain adversary on fixed latent space at the end of training
# TODO: allow for multiple heads and multiple adversaries   
# TODO define loss-weights in the model object
# TODO ordinal category loss and head
# TODO: Update the TGEM model to be a encoder-style model class "TGEM_Encoder"
# TODO: add a "task" (binary, regression, multi, etc) to the model class
# TODO: add a "goal" (reduce, adversarial, primary) (method: define_goal)

encoder, head, adversary, output_data = train_semisupervised(dataloader_dct,encoder,head,adversary,
                     save_dir=output_dir,
                    num_epochs=num_epochs,
                    learning_rate=0.001,
                    encoder_weight=0.5,
                    head_weight=2,
                    adversary_weight=2,
                    end_state_eval_funcs=end_state_eval_funcs)


print(output_data)