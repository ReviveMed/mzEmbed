

import torch
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

from prep import ClassifierDataset
from prep_cv import create_cross_validation_directories, run_cross_validation_sklearn_classifier, run_cross_validation_pytorch_classifier
import time
import shutil


base_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton/development_CohortCombination'
if not os.path.exists(base_dir):
    base_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination'

date_name = 'hilic_pos_2024_feb_05_read_norm_poolmap'
feat_subset_name = 'num_cohorts_thresh_0.5'
study_subset_name= 'subset all_studies with align score 0.25 from Merge_Jan25_align_80_40_fillna_avg'
task_name = 'std_1_Multi'
# task_name = 'std_1_Benefit'
input_dir = f'{base_dir}/{date_name}/{study_subset_name}/{feat_subset_name}/{task_name}'

### Benefit Task
# y_stratify_col = 'Benefit'
# label_mapper = {'NCB': 0, 'CB': 1, 'ICB': np.nan}
# # test_files_suffix = ['','_alt']
# # phase_list = ['train','val','test','test_alt']
# test_files_suffix = ['']
# phase_list = ['train','val','test']


### MSKCC Task
# y_stratify_col = 'MSKCC'
# label_mapper = {'FAVORABLE': 1, 'POOR': 0, 'INTERMEDIATE': np.nan}
# # test_files_suffix = []
# phase_list = ['train','val',f'test_{y_stratify_col}']
# test_files_suffix = [f'_{y_stratify_col}']


for y_col in ['Benefit']:

    if y_col == 'NIVO_Benefit':
        y_stratify_col = 'Benefit'
        label_mapper = {'NCB': 0, 'CB': 1, 'ICB': np.nan}
        test_files_suffix = ['', '_alt']
        phase_list = ['train','val','test','test_alt']

    elif y_col == 'Benefit':
        y_stratify_col = 'Benefit'
        label_mapper = {'NCB': 0, 'CB': 1, 'ICB': np.nan}
        # test_files_suffix = ['']
        # phase_list = ['train','val','test']
        test_files_suffix = ['', '_alt']
        phase_list = ['train','val','test','test_alt']
    elif y_col == 'MSKCC':
        y_stratify_col = 'MSKCC'
        label_mapper = {'FAVORABLE': 1, 'POOR': 0, 'INTERMEDIATE': np.nan}
        phase_list = ['train','val',f'test_{y_stratify_col}']
        test_files_suffix = [f'_{y_stratify_col}']
        import warnings
        warnings.filterwarnings('ignore', module='torchmetrics')


    n_repeats = 2
    cv_splits = 5
    save_dir = create_cross_validation_directories(input_dir, cv_splits=cv_splits, n_repeats=n_repeats,
                                                    y_stratify_col=y_stratify_col, cross_val_seed=42, 
                                                    test_files_suffix=test_files_suffix,
                                                    finetune_suffix='_finetune')
    
    if y_stratify_col == 'Benefit':
        X_test_alt_file = os.path.join(input_dir, 'X_test_alt.csv')
        y_test_alt_file = os.path.join(input_dir, 'y_test_alt.csv')
        for iter in range(n_repeats):
            for cvn in range(0,5):
                # copy to subdir
                new_dir = os.path.join(save_dir, f'cross_val_split_{cvn}_{iter}')
                shutil.copy(X_test_alt_file, new_dir)
                shutil.copy(y_test_alt_file, new_dir)

    # continue
    # %% Run the cross-validation across sklearn models

    if True:
        # test sklearn logistic regression
        run_cross_validation_sklearn_classifier(
            save_dir,
            cv_splits = cv_splits, 
            n_repeats=n_repeats, 
            subdir='sklearn_models',
            label_mapper=label_mapper,
            test_files_suffix=test_files_suffix,
            label_col=y_stratify_col,
            model_kind = 'logistic_regression',
            model_name='logistic_regression_default2',
            param_grid={},
        )

        run_cross_validation_sklearn_classifier(
            save_dir,
            cv_splits = cv_splits, 
            n_repeats=n_repeats, 
            subdir='sklearn_models',
            label_mapper=label_mapper,
            test_files_suffix=test_files_suffix,
            label_col=y_stratify_col,
            model_kind = 'random_forest',
            model_name='random_forest_default',
            param_grid={},
        )


        run_cross_validation_sklearn_classifier(
            save_dir,
            cv_splits = cv_splits, 
            n_repeats=n_repeats, 
            subdir='sklearn_models',
            label_mapper=label_mapper,
            test_files_suffix=test_files_suffix,
            label_col=y_stratify_col,
            model_kind = 'svc',
            model_name='svc_default',
            param_grid={},
        )

    # %% Run the cross-validation across pytorch models

    # get a list of all of the pre-trained models
    batch_size = 64 

    head_num_hidden_layers = 0
    head_hidden_sz = 0
    dropout_rate = 0.25
    use_batch_norm= False
    encoder_status = 'random'
    # encoder_status = 'finetune'
    # encoder_kind = 'AE'
    # activation = 'sigmoid'
    activation = 'relu'
    encoder_activation = activation
    encoder_dropout_rate = dropout_rate
    # encoder_learning_rate = learning_rate
    encoder_act_on_latent_layer = True

   #TODO add option to pre-train using the training data to initialize the encoder weights

    params_dict_list = []
    # min_encoder_hidden_size = 9999
    for use_batch_norm in [False,True]:
        for noise_factor in [0.01,0,0.05]: # 0.05
            for early_stopping_patience in [-1,10]:
                for encoder_kind in ['VAE','AE']:
                    for encoder_status in ['finetune','random']:
                        for encoder_hidden_size_mult in [1,2]:
                            for encoder_num_hidden_layers in [0,1]:
                                for latent_size in [64,128]: #256
                                    for num_epochs in [100]: #50,200
                                        for learning_rate in [1e-4]: #1e-5, 1e-3
                                            # for learning_rate in [1e-4]:
                                            
                                            if encoder_kind == 'VAE' and encoder_status == 'random':
                                                continue
                                        
                                            if (encoder_num_hidden_layers == 0) and (encoder_hidden_size_mult > 1):
                                                continue

                                            encoder_hidden_size = latent_size * encoder_hidden_size_mult
                                            head_name  = 'Survey_epoch{}_lr{}_dr{}_noi{}'.format(num_epochs, learning_rate, \
                                                                                                       dropout_rate, noise_factor)
                                            encoder_name = '{}_layers{}_hidden{}_latent{}'.format(encoder_kind,encoder_num_hidden_layers, encoder_hidden_size, latent_size)
                                            if use_batch_norm:
                                                encoder_name += '_BN'
                                            if early_stopping_patience > 0:
                                                head_name += '_stop{}'.format(early_stopping_patience)

                                            params_dict = {
                                                'encoder_hidden_size': encoder_hidden_size,
                                                'encoder_num_hidden_layers': encoder_num_hidden_layers,
                                                'latent_size': latent_size,
                                                'num_epochs': num_epochs,
                                                'encoder_name': encoder_name,
                                                'encoder_kind': encoder_kind,
                                                'head_name': head_name,
                                                'learning_rate': learning_rate,
                                                'encoder_status': encoder_status,
                                                'use_batch_norm': use_batch_norm,
                                                'early_stopping_patience': early_stopping_patience,
                                                'noise_factor': noise_factor
                                            }
                                                
                                            params_dict_list.append(params_dict)

    print('')
    print('Number of models to run:', len(params_dict_list))
    print('')

    for params_dict in params_dict_list:
        encoder_hidden_size = params_dict['encoder_hidden_size']
        encoder_num_hidden_layers = params_dict['encoder_num_hidden_layers']
        latent_size = params_dict['latent_size']
        num_epochs = params_dict['num_epochs']
        encoder_name = params_dict['encoder_name']
        encoder_kind = params_dict['encoder_kind']
        head_name = params_dict['head_name']
        learning_rate = params_dict['learning_rate']
        encoder_status = params_dict['encoder_status']
        use_batch_norm = params_dict['use_batch_norm']
        early_stopping_patience = params_dict['early_stopping_patience']
        noise_factor = params_dict['noise_factor']
        if encoder_num_hidden_layers == 0:
            encoder_hidden_size = latent_size
        start_time = time.time()

        pretrained_encoder_load_path = None
        # pretrained_encoder_load_path = os.path.join(input_dir, 'pretrain_models_feb08', f'{encoder_name}_model.pth')
        # if not os.path.exists(pretrained_encoder_load_path):
        #     print('Pre-trained model does not exist:', pretrained_encoder_load_path)
        #     continue


        print('###############')
        print(f'{head_name} on {encoder_name} started')
        run_cross_validation_pytorch_classifier(save_dir,
                                                cv_splits=cv_splits, 
                                                n_repeats=n_repeats, 
                                                subdir='pytorch_models',
                                                label_mapper=label_mapper,
                                                label_col=y_stratify_col,
                                                batch_size=batch_size,
                                                test_files_suffix=test_files_suffix,
                                                pretrained_encoder_load_path=pretrained_encoder_load_path,
                                                num_epochs=num_epochs, 
                                                learning_rate=learning_rate, 
                                                encoder_learning_rate=learning_rate,
                                                early_stopping_patience=early_stopping_patience,
                                                latent_size = latent_size,
                                                hidden_size = head_hidden_sz,
                                                num_hidden_layers = head_num_hidden_layers,
                                                encoder_hidden_size = encoder_hidden_size,
                                                encoder_dropout_rate = encoder_dropout_rate,
                                                encoder_num_hidden_layers = encoder_num_hidden_layers,
                                                encoder_status=encoder_status,
                                                encoder_activation = encoder_activation,
                                                encoder_use_batch_norm = use_batch_norm,
                                                encoder_act_on_latent_layer=encoder_act_on_latent_layer,
                                                model_name = head_name,
                                                activation = activation,
                                                phase_list=phase_list,
                                                encoder_name=encoder_name,
                                                encoder_kind=encoder_kind,
                                                dropout_rate=dropout_rate,
                                                use_batch_norm=use_batch_norm,
                                                noise_factor = noise_factor,
                                                verbose=False,
                                                load_existing_model = True)

        print('###############')
        print(f'{head_name} on {encoder_name} completed')
        print('###############')
        print('Elapsed time:', (time.time()-start_time)/60, 'minutes')

        # %%

