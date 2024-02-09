

import torch
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

from prep import ClassifierDataset
from prep_cv import create_cross_validation_directories, run_cross_validation_sklearn_classifier, run_cross_validation_pytorch_classifier




base_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton/development_CohortCombination'
if not os.path.exists(base_dir):
    base_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination'

date_name = 'hilic_pos_2024_feb_05_read_norm_poolmap'
feat_subset_name = 'num_cohorts_thresh_0.5'
study_subset_name= 'subset all_studies with align score 0.25 from Merge_Jan25_align_80_40_fillna_avg'
task_name = 'std_1_Multi'
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


for y_stratify_col in ['MSKCC','Benefit']:

    if y_stratify_col == 'Benefit':
        label_mapper = {'NCB': 0, 'CB': 1, 'ICB': np.nan}
        test_files_suffix = ['']
        phase_list = ['train','val','test']
    elif y_stratify_col == 'MSKCC':
        label_mapper = {'FAVORABLE': 1, 'POOR': 0, 'INTERMEDIATE': np.nan}
        phase_list = ['train','val',f'test_{y_stratify_col}']
        test_files_suffix = [f'_{y_stratify_col}']


    n_repeats = 2
    cv_splits = 5
    save_dir = create_cross_validation_directories(input_dir, cv_splits=cv_splits, n_repeats=n_repeats,
                                                    y_stratify_col=y_stratify_col, cross_val_seed=42, 
                                                    test_files_suffix=test_files_suffix,
                                                    finetune_suffix='_finetune')

    # %% Run the cross-validation across sklearn models

    if False:
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
            model_name='logistic_regression_default',
            param_grid={},
        )


    # %% Run the cross-validation across pytorch models

    # get a list of all of the pre-trained models
    encoder_dir = os.path.join(input_dir, 'pretrain_models_feb07')
    encoder_files = [f for f in os.listdir(encoder_dir) if f.endswith('_model.pth')]

    batch_size = 64 
    # batch normalization works better with larger batch size, 64 should be fine

    for head_num_hidden_layers in [1]:
        for head_hidden_sz in [16]:
            use_batch_norm = False
            dropout_rate = 0.25
            # head_hidden_sz = 64
            # head_num_hidden_layers = 2
            head_name  = f'Cl_{head_hidden_sz}_{head_num_hidden_layers}_D{dropout_rate}'
            if use_batch_norm:
                head_name += '_BN'


            for encoder_file in encoder_files:
                encoder_name = encoder_file.split('_model.pth')[0]
                print(encoder_name)
                pretrained_encoder_load_path = os.path.join(encoder_dir, encoder_file)
                
                # load the encoder model hyperparameters
                encoder_output_file = os.path.join(encoder_dir, f'{encoder_name}_output.json')
                with open(encoder_output_file, 'r') as f:
                    encoder_output = json.load(f)
                    encoder_hidden_size = encoder_output['model_hyperparameters']['hidden_size']
                    encoder_dropout_rate = encoder_output['model_hyperparameters']['dropout_rate']
                    encoder_num_hidden_layers = encoder_output['model_hyperparameters']['num_hidden_layers']
                    encoder_kind = encoder_output['model_kind']
                    activation = encoder_output['model_hyperparameters']['activation']
                    if 'use_batch_norm' in encoder_output['model_hyperparameters']:
                        encoder_use_batch_norm = encoder_output['model_hyperparameters']['use_batch_norm']
                    else:
                        encoder_use_batch_norm = False
                    encoder_activation = activation
                    latent_size = encoder_output['model_hyperparameters']['latent_size']

                    if not((encoder_dropout_rate == 0) or (encoder_num_hidden_layers == 1) or (encoder_hidden_size == 16) or (latent_size==32)):
                        print('skipping encoder: ', encoder_name)
                        continue

                    for encoder_status in ['finetune','random']:
                        run_cross_validation_pytorch_classifier(save_dir,
                                                                cv_splits=cv_splits, 
                                                                n_repeats=n_repeats, 
                                                                subdir='pytorch_models',
                                                                label_mapper=label_mapper,
                                                                label_col=y_stratify_col,
                                                                batch_size=batch_size,
                                                                test_files_suffix=test_files_suffix,
                                                                pretrained_encoder_load_path=pretrained_encoder_load_path,
                                                                num_epochs=500, 
                                                                learning_rate=1e-4, 
                                                                encoder_learning_rate=1e-5,
                                                                early_stopping_patience=50,
                                                                latent_size = latent_size,
                                                                hidden_size = head_hidden_sz,
                                                                num_hidden_layers = head_num_hidden_layers,
                                                                encoder_hidden_size = encoder_hidden_size,
                                                                encoder_dropout_rate = encoder_dropout_rate,
                                                                encoder_num_hidden_layers = encoder_num_hidden_layers,
                                                                encoder_status=encoder_status,
                                                                encoder_activation = encoder_activation,
                                                                encoder_use_batch_norm = encoder_use_batch_norm,
                                                                model_name = head_name,
                                                                activation =activation,
                                                                phase_list=phase_list,
                                                                encoder_name=encoder_name,
                                                                encoder_kind=encoder_kind,
                                                                dropout_rate=dropout_rate,
                                                                use_batch_norm=use_batch_norm,
                                                                verbose=False)


    # %%

