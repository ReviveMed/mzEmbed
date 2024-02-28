import torch
import pandas as pd
import numpy as np
import os

from models import VAE, AE, TGEM, BinaryClassifier, MultiClassClassifier, TGEM_Encoder
from model_training import CompoundDataset, train_compound_model


def get_model(model_kind, input_size, **kwargs):

    if model_kind == 'NA':
        model = None
    elif model_kind == 'VAE':
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


def pretrain_AE_then_finetune_wrapper(X_data,y_data,splits,**kwargs):

    ############################
    # load the kwargs


    save_dir = kwargs.get('save_dir', None)
    encoder_kind = kwargs.get('encoder_kind', 'AE')
    encoder_kwargs = kwargs.get('encoder_kwargs', {})
    other_size = kwargs.get('other_size', 1)

    y_pretrain_cols = kwargs.get('y_pretrain_cols', None)
    y_finetune_cols = kwargs.get('y_finetune_cols', None)
    num_folds = kwargs.get('num_folds', None)

    if num_folds is None:
        num_folds = splits.shape[1]

    ############################
    # Filter, preprocess Data
    X_pretrain = X_data
    y_pretrain = y_data[y_pretrain_cols]

    X_finetune = X_data.loc[splits.index]
    y_finetune = y_data.loc[splits.index][y_finetune_cols]

    input_size = X_pretrain.shape[1]

    ############################
    # Pretrain
    pretrain_batch_size = kwargs.get('pretrain_batch_size', 32)
    pretrain_val_frac = kwargs.get('pretrain_val_frac', 0)
    pretrain_head_kind = kwargs.get('pretrain_head_kind', 'NA')
    pretrain_adv_kind = kwargs.get('pretrain_adv_kind', 'NA')
    pretrain_kwargs = kwargs.get('pretrain_kwargs', {})
    pretrain_head_kwargs = kwargs.get('pretrain_head_kwargs', {})
    pretrain_adv_kwargs = kwargs.get('pretrain_adv_kwargs', {})

    pretrain_dir = os.path.join(save_dir, 'pretrain')
    os.makedirs(pretrain_dir, exist_ok=True)
    pretrain_kwargs['save_dir'] = pretrain_dir
    encoder = get_model(encoder_kind, input_size, **encoder_kwargs)
    
    pretrain_head = get_model(pretrain_head_kind, encoder.get_output_size()+other_size, **pretrain_head_kwargs)
    pretrain_adv = get_model(pretrain_adv_kind, encoder.get_output_size(), **pretrain_adv_kwargs)

    head_col = y_pretrain_cols[0]
    adv_col = y_pretrain_cols[1]
    pretrain_dataset = CompoundDataset(X_pretrain, y_pretrain[head_col], y_pretrain[adv_col])
    if pretrain_val_frac>0:
        train_size = int((1-pretrain_val_frac) * len(pretrain_dataset))
        val_size = len(pretrain_dataset) - train_size

        pretrain_dataset, preval_dataset = torch.utils.data.random_split(pretrain_dataset, [train_size, val_size])
        val_loader = torch.utils.data.DataLoader(preval_dataset, batch_size=pretrain_batch_size, shuffle=False)

    dataloaders = {
        'train': torch.utils.data.DataLoader(pretrain_dataset, batch_size=pretrain_batch_size, shuffle=True),
    }
    if pretrain_val_frac> 0:
        dataloaders['val'] = val_loader


    if pretrain_head is not None:
        num_classes_pretrain_head = pretrain_dataset.get_num_classes_head()
        weights_pretrain_head = pretrain_dataset.get_class_weights_head()
        pretrain_head.define_loss(class_weight=weights_pretrain_head)


    if pretrain_adv is not None:        
        num_classes_pretrain_adv = pretrain_dataset.get_num_classes_adv()
        weights_pretrain_adv = pretrain_dataset.get_class_weights_adv()
        pretrain_adv.define_loss(class_weight=weights_pretrain_adv)

    
    train_compound_model(dataloaders, encoder, pretrain_head, pretrain_adv, **pretrain_kwargs)


    ############################
    # Finetune
    finetune_batch_size = kwargs.get('finetune_batch_size', 32)
    finetune_val_frac = kwargs.get('finetune_val_frac', 0)
    finetune_head_kind = kwargs.get('finetune_head_kind', 'MultiClassClassifier')
    finetune_adv_kind = kwargs.get('finetune_adv_kind', 'NA')
    finetune_head_kwargs = kwargs.get('finetune_head_kwargs', {})
    finetune_adv_kwargs = kwargs.get('finetune_adv_kwargs', {})
    finetune_kwargs = kwargs.get('finetune_kwargs', {})

    finetune_dir = os.path.join(save_dir, 'finetune')
    finetune_kwargs['save_dir'] = finetune_dir
    os.makedirs(finetune_dir, exist_ok=True)

    head_col = y_finetune_cols[0]
    adv_col = y_finetune_cols[1]

    finetune_head = get_model(finetune_head_kind, encoder.get_output_size()+other_size, **finetune_head_kwargs)
    finetune_adv = get_model(finetune_adv_kind, encoder.get_output_size(), **finetune_adv_kwargs)

    if finetune_head is not None:
        num_classes_finetune_head = pretrain_dataset.get_num_classes_head()
        weights_finetune_head = pretrain_dataset.get_class_weights_head()
        finetune_head.define_loss(class_weight=weights_finetune_head)

    if finetune_adv is not None:        
        num_classes_finetune_adv = pretrain_dataset.get_num_classes_adv()
        weights_finetune_adv = pretrain_dataset.get_class_weights_adv()
        finetune_adv.define_loss(class_weight=weights_finetune_adv)

    #### Start the CV loop
    for n_fold in num_folds:
        X_train, y_train = X_finetune.loc[~splits.iloc[n_fold]], y_finetune.loc[~splits.iloc[n_fold]]
        X_test, y_test = X_finetune.loc[splits.iloc[n_fold]], y_finetune.loc[splits.iloc[n_fold]]


        train_dataset = CompoundDataset(X_train, y_train[head_col], y_train[adv_col])
        test_dataset = CompoundDataset(X_test, y_test[head_col], y_test[adv_col])
        
        # Split the training dataset into training and validation sets
        if finetune_val_frac>0:
            train_size = int((1-finetune_val_frac) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=finetune_batch_size, shuffle=False)


        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False)
        }
        
        if finetune_val_frac> 0:
            dataloaders['val'] = val_loader

        
        # Initialize the models
        finetune_head.reset_weights()
        if finetune_adv is not None:
            finetune_adv.reset_weights()

        encoder.load_state_dict(torch.load(os.path.join(pretrain_dir, 'encoder.pth')))
 

        # Run the train and evaluation
        _, _, _, output_data = train_compound_model(dataloaders, encoder, finetune_head, finetune_adv, **finetune_kwargs)




    ############################
    # Finetune from Random
    randtune_dir = os.path.join(save_dir, 'randtune')
    os.makedirs(randtune_dir, exist_ok=True)
    randtune_kwargs = finetune_kwargs
    randtune_kwargs['save_dir'] = randtune_dir
    
    #### Start the CV loop
    for n_fold in num_folds:
        X_train, y_train = X_finetune.loc[~splits.iloc[n_fold]], y_finetune.loc[~splits.iloc[n_fold]]
        X_test, y_test = X_finetune.loc[splits.iloc[n_fold]], y_finetune.loc[splits.iloc[n_fold]]


        train_dataset = CompoundDataset(X_train, y_train[head_col], y_train[adv_col])
        test_dataset = CompoundDataset(X_test, y_test[head_col], y_test[adv_col])
        
        # Split the training dataset into training and validation sets
        if finetune_val_frac>0:
            train_size = int((1-finetune_val_frac) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=finetune_batch_size, shuffle=False)


        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=False)
        }
        
        if finetune_val_frac> 0:
            dataloaders['val'] = val_loader

        
        # Initialize the models
        finetune_head.reset_weights()
        if finetune_adv is not None:
            finetune_adv.reset_weights()

        encoder.load_state_dict(torch.load(os.path.join(pretrain_dir, 'encoder.pth')))
 

        # Run the train and evaluation
        _, _, _, output_data = train_compound_model(dataloaders, encoder, finetune_head, finetune_adv, **randtune_kwargs)



    pass