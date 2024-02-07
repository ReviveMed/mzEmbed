# Explore the data and the models
# %%
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

from prep import ClassifierDataset
from train import run_train_classifier
from sklearn_models import run_train_sklearn_model

# %%

def create_cross_validation_directories(input_dir, cross_val_splits=5, 
                                        n_repeats=1,
                                        y_stratify_col='Benefit', cross_val_seed=42, 
                                        test_files_suffix=['','_alt'],
                                        finetune_suffix='_finetune'):
    """
    Create cross-validation directories for finetuning a model.

    Parameters:
    - input_dir (str): The directory path where the input files are located.
    - cross_val_splits (int): The number of cross-validation splits to create. Default is 5.
    - y_stratify_col (str): The column name to stratify the data on. Default is 'Benefit'.
    - cross_val_seed (int): The random seed for cross-validation. Default is 42.
    - test_files_suffix (list): The suffixes to be appended to the test file names. Default is ['', '_alt'].
    - finetune_suffix (str): The suffix to be appended to the finetune file names. Default is '_finetune'.

    Returns:
    - save_dir (str): The directory path where the cross-validation directories are saved.
    """

    X_test_files = [f'X_test{suffix}.csv' for suffix in test_files_suffix]
    y_test_files = [f'y_test{suffix}.csv' for suffix in test_files_suffix]

    X_finetune_file = os.path.join(input_dir, f'X{finetune_suffix}.csv')
    y_finetune_file = os.path.join(input_dir, f'y{finetune_suffix}.csv')
    save_dir = os.path.join(input_dir, f'CV{cross_val_splits}_seed{cross_val_seed}_on_{y_stratify_col}')
    if os.path.exists(save_dir):
        return save_dir
    os.makedirs(save_dir, exist_ok=True)

    # create Stratified KFold
    X_finetune = pd.read_csv(X_finetune_file, index_col=0)
    y_finetune = pd.read_csv(y_finetune_file, index_col=0)
    y_stratify = y_finetune[y_stratify_col]

    for iter in range(n_repeats):
        skf = StratifiedKFold(n_splits=cross_val_splits, random_state=cross_val_seed+iter, shuffle=True)
        skf.get_n_splits(X_finetune, y_stratify)

        # create a subdirectory for each of the cross-validation splits
        for i, (train_index, test_index) in enumerate(skf.split(X_finetune, y_stratify)):
            X_train, X_val = X_finetune.iloc[train_index], X_finetune.iloc[test_index]
            y_train, y_val = y_finetune.iloc[train_index], y_finetune.iloc[test_index]
            split_dir = os.path.join(save_dir,  f'cross_val_split_{i}_{iter}')
            os.makedirs(split_dir, exist_ok=True)
            X_train.to_csv(os.path.join(split_dir, 'X_train.csv'))
            X_val.to_csv(os.path.join(split_dir, 'X_val.csv'))
            y_train.to_csv(os.path.join(split_dir, 'y_train.csv'))
            y_val.to_csv(os.path.join(split_dir, 'y_val.csv'))
            
            # also move the Test files to the same directories
            for X_test_file, y_test_file in zip(X_test_files, y_test_files):
                if os.path.exists(os.path.join(input_dir, X_test_file)):
                    X_test = pd.read_csv(os.path.join(input_dir, X_test_file), index_col=0)
                    y_test = pd.read_csv(os.path.join(input_dir, y_test_file), index_col=0)
                    X_test.to_csv(os.path.join(split_dir, X_test_file))
                    y_test.to_csv(os.path.join(split_dir, y_test_file))
                else:
                    print(f'{X_test_file} does not exist in {input_dir}')

    return save_dir



def run_cross_validation_pytorch_classifier(cv_dir,cv_splits,
                                    n_repeats=1,
                                    subdir='pytorch_models',
                                    label_mapper=None,
                                    batch_size=32,
                                    test_files_suffix=['','_alt'],**kwargs):

    if label_mapper is None:
        label_mapper = {'CB': 1.0, 'NCB': 0.0, 'ICB': np.nan}

    gather_output = []

    for iter in range(n_repeats):
        for cv_num in range(cv_splits):
            input_dir = os.path.join(cv_dir, f'cross_val_split_{cv_num}_{iter}')
            save_dir = os.path.join(input_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)

            train_dataset = ClassifierDataset(input_dir,subset='train',label_encoder=label_mapper)
            val_dataset = ClassifierDataset(input_dir,subset='val',label_encoder=label_mapper)
            print('size of training:', len(train_dataset))
            print('size of validation:', len(val_dataset))

            dataloaders= {
                'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            }

            # add all of the test subsets
            for test_subset in test_files_suffix:
                if os.path.exists(os.path.join(input_dir, f'X_test{test_subset}.csv')):
                    test_dataset = ClassifierDataset(input_dir,subset=f'test{test_subset}',label_encoder=label_mapper)
                    dataloaders[f'test{test_subset}'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    print(f'size of testing{test_subset}:', len(test_dataset))
                else:
                    print(f'X_test{test_subset}.csv does not exist in {input_dir}')

            output_data = run_train_classifier(dataloaders,save_dir,**kwargs)
            gather_output.append(output_data)


    ## Summarize the important results
    best_epoch_list = [output['best_epoch'] for output in gather_output]
    testalt_auroc_list = [output['end_state_auroc']['test_alt'] for output in gather_output]
    test_auroc_list = [output['end_state_auroc']['test'] for output in gather_output]
    val_auroc_list = [output['end_state_auroc']['val'] for output in gather_output]
    train_auroc_list = [output['end_state_auroc']['train'] for output in gather_output]
    model_name = gather_output[0]['model_name']
    encoder_name = gather_output[0]['encoder_name']
    pd.DataFrame({'best_epoch': best_epoch_list,
                  'test_alt_auroc': testalt_auroc_list,
                    'test_auroc': test_auroc_list,
                    'val_auroc': val_auroc_list,
                    'train_auroc': train_auroc_list}).to_csv(os.path.join(cv_dir, f'{model_name}_{encoder_name}_summary.csv'))
    
    return


def run_cross_validation_sklearn_classifier(cv_dir,cv_splits,
                                    n_repeats=1,
                                    subdir='sklearn_models',
                                    label_mapper=None,
                                    batch_size=32,
                                    test_files_suffix=['','_alt'],**kwargs):
    
    if label_mapper is None:
        label_mapper = {'CB': 1.0, 'NCB': 0.0, 'ICB': np.nan}

    gather_output = []

    for iter in range(n_repeats):
        for cv_num in range(cv_splits):
            input_dir = os.path.join(cv_dir, f'cross_val_split_{cv_num}_{iter}')
            save_dir = os.path.join(input_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)

            train_dataset = ClassifierDataset(input_dir,subset='train',label_encoder=label_mapper)
            val_dataset = ClassifierDataset(input_dir,subset='val',label_encoder=label_mapper)
            print('size of training:', len(train_dataset))
            print('size of validation:', len(val_dataset))

            data_dict = {
                'train': train_dataset,
                'val': val_dataset,
            }

            # add all of the test subsets
            for test_subset in test_files_suffix:
                if os.path.exists(os.path.join(input_dir, f'X_test{test_subset}.csv')):
                    test_dataset = ClassifierDataset(input_dir,subset=f'test{test_subset}',label_encoder=label_mapper)
                    data_dict[f'test{test_subset}'] = test_dataset
                    print(f'size of testing{test_subset}:', len(test_dataset))
                else:
                    print(f'X_test{test_subset}.csv does not exist in {input_dir}')

            output_data = run_train_sklearn_model(data_dict,save_dir,**kwargs)
            gather_output.append(output_data)

    ## Summarize the important results
    # best_epoch_list = [output['best_epoch'] for output in gather_output]
    testalt_auroc_list = [output['end_state_auroc']['test_alt'] for output in gather_output]
    test_auroc_list = [output['end_state_auroc']['test'] for output in gather_output]
    val_auroc_list = [output['end_state_auroc']['val'] for output in gather_output]
    train_auroc_list = [output['end_state_auroc']['train'] for output in gather_output]
    model_name = gather_output[0]['model_name']
    pd.DataFrame({
                  'test_alt_auroc': testalt_auroc_list,
                    'test_auroc': test_auroc_list,
                    'val_auroc': val_auroc_list,
                    'train_auroc': train_auroc_list}).to_csv(os.path.join(cv_dir, f'{model_name}_summary.csv'))


# %%

if __name__ == '__main__':

    base_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton/development_CohortCombination'
    if not os.path.exists(base_dir):
        base_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination'

    date_name = 'hilic_pos_2024_feb_02_read_norm'
    feat_subset_name = 'num_cohorts_thresh_0.5'
    study_subset_name= 'subset all_studies with align score 0.3 from Merge_Jan25_align_80_40_fillna_avg'
    task_name = 'std_1_Benefit'
    input_dir = f'{base_dir}/{date_name}/{study_subset_name}/{feat_subset_name}/{task_name}'


    save_dir = create_cross_validation_directories(input_dir, cross_val_splits=5,
                                                    y_stratify_col='Benefit', cross_val_seed=42, 
                                                    test_files_suffix=['','_alt'],
                                                    finetune_suffix='_finetune')
# %%

    pretrained_encoder_load_path = os.path.join(input_dir, 'pretrain_models_feb07', 'VAE_H64_1_L64_D0_Aleakyrelu_model.pth')
    encoder_name = 'VAE_H64_1_L64_D0_Aleakyrelu'
    run_cross_validation_pytorch_classifier(save_dir,5, n_repeats=1, 
                                            subdir='temp',
                                            label_mapper=None,
                                            batch_size=32,
                                            test_files_suffix=['','_alt'],
                                            pretrained_encoder_load_path=pretrained_encoder_load_path,
                                            num_epochs=500, 
                                            learning_rate=1e-4, 
                                            early_stopping_patience=50,
                                            latent_size = 64,
                                            encoder_hidden_size = 64,
                                            encoder_dropout_rate = 0.0,
                                            encoder_status='random',
                                            model_name = 'T1',
                                            activation ='sigmoid',
                                            phase_list=['train','val','test','test_alt'],
                                            encoder_name=encoder_name,
                                            encoder_kind='VAE',dropout_rate=0.25)
    


    # test sklearn logistic regression
    # run_cross_validation_sklearn_classifier(
    #     save_dir,5, n_repeats=1, subdir='sklearn_models',
    #     label_mapper=None,
    #     model_kind = 'logistic_regression',
    #     model_name='logistic_regression_default',
    #     param_grid={},
    # )