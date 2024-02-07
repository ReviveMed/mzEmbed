# Functions for preparing data
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data import TensorDataset
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 
######### Simple Dataset for Pretraining an Autoencoder
class PreTrainingDataset(Dataset):
    def __init__(self, input_dir):
        X_df = pd.read_csv(os.path.join(input_dir, 'X_pretrain.csv'), index_col=0)
        self.X = torch.tensor(X_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


######### Simple Dataset for Training a Classifier
class ClassifierDataset(Dataset):
    def __init__(self, input_dir, subset='train',label_encoder=None,y_dropna=True):
        X_df = pd.read_csv(os.path.join(input_dir, f'X_{subset}.csv'), index_col=0)
        y_df = pd.read_csv(os.path.join(input_dir, f'y_{subset}.csv'), index_col=0)
        y_df = y_df.iloc[:,0]
        self.feature_ids = X_df.columns

        # check the data type
        if y_df.dtype == 'object':
        # if y_df.dtypes[0] == 'object':
            
            if label_encoder is None:
                label_encoder = LabelEncoder()
                # y_df = pd.DataFrame(label_encoder.fit_transform(y_df.to_numpy().ravel()),
                                    # index=y_df.index, columns=['y'])
                y_df = pd.Series(label_encoder.fit_transform(y_df.to_numpy().ravel()),
                                    index=y_df.index, name='y')
            
            elif isinstance(label_encoder,LabelEncoder):
                # y_df = pd.DataFrame(label_encoder.transform(y_df.to_numpy().ravel()),
                #                     index=y_df.index, columns=['y'])
                y_df = pd.Series(label_encoder.transform(y_df.to_numpy().ravel()),
                                    index=y_df.index, name='y')

            elif isinstance(label_encoder,dict):
                y_df = y_df.map(label_encoder)
            else:
                raise ValueError('label_encoder must be a LabelEncoder or a dictionary')

        if y_dropna:
            y_df.dropna(inplace=True)
            X_df = X_df.loc[y_df.index]

        self.X = torch.tensor(X_df.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y_df.to_numpy(), dtype=torch.float32)
        self.sample_ids = X_df.index

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_label_encoder(self):
        return self.label_encoder


############# 

class RCC3Dataset(Dataset):
    def __init__(self,peaks_file, metadata_file,use_onehot=False,na_subset=['Benefit']):

        # Load the data
        peaks = pd.read_csv(peaks_file, index_col=0)
        metadata = pd.read_csv(metadata_file, index_col=0)
        metadata.dropna(subset=na_subset, inplace=True)

        self.X = peaks.loc[metadata.index]

        self.benefit = metadata['Benefit'].map({'CB': 1, 'NCB': 0, 'ICB': 0.5})
        self.pfs = metadata['PFS']
        self.os = metadata['OS']
        self.pfs_event = metadata['PFS_Event']
        self.os_event = metadata['OS_Event']
        self.study_week = metadata['study_week']
        self.batch_id = metadata['batch_id']
        self.dose = metadata['Dose (mg/kg)']
        self.phase = metadata['Phase'].map({'RCC3': 3, 'RCC1': 1})
        self.mskcc = metadata['MSKCC'].map({'FAVORABLE': 1, 'INTERMEDIATE': 0.5, 'POOR': 0})

        if use_onehot:
            # One hot encode the metadata
            self.gend_encoder = OneHotEncoder()
            self.region_encoder = OneHotEncoder()
            self.treatment_encoder = OneHotEncoder()
            self.age_group_encoder = OneHotEncoder()
            self.prior_treatment_encoder = OneHotEncoder()
            self.mskcc_encoder = OneHotEncoder()

            self.gend = self.gend_encoder.fit_transform(metadata['Sex'])
            self.region = self.region_encoder.fit_transform(metadata['Region'])
            self.treatment = self.treatment_encoder.fit_transform(metadata['Treatment'])
            self.age_group = self.age_group_encoder.fit_transform(metadata['Age_Group'])
            self.prior_treatment = self.prior_treatment_encoder.fit_transform(metadata['Prior_Treatment'])

        else:
            # Convert sex and region to integer encoding
            self.gend_encoder = LabelEncoder()
            self.region_encoder = LabelEncoder()
            self.treatment_encoder = LabelEncoder()
            self.age_group_encoder = LabelEncoder()
            self.prior_treatment_encoder = LabelEncoder()

            self.gend = self.gend_encoder.fit_transform(metadata['Sex'])
            self.region = self.region_encoder.fit_transform(metadata['Region'])
            self.treatment = self.treatment_encoder.fit_transform(metadata['Treatment'])
            self.age_group = self.age_group_encoder.fit_transform(metadata['Age_Group'])
            self.prior_treatment = self.prior_treatment_encoder.fit_transform(metadata['Prior_Treatment'])


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert the data to PyTorch tensors
        X = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        benefit = torch.tensor(self.benefit.iloc[idx], dtype=torch.float32)
        pfs = torch.tensor(self.pfs.iloc[idx], dtype=torch.float32)
        pfs_event = torch.tensor(self.pfs_event.iloc[idx], dtype=torch.float32)
        gend = torch.tensor(self.gend[idx], dtype=torch.float32)
        age_group = torch.tensor(self.age_group[idx], dtype=torch.float32)
        region = torch.tensor(self.region[idx], dtype=torch.float32)


        return X, benefit, pfs, pfs_event, gend, age_group, region
    
    def select_subset(self, subset_idx):
        self.X = self.X.iloc[subset_idx]
        self.benefit = self.benefit.iloc[subset_idx]
        self.pfs = self.pfs.iloc[subset_idx]
        self.pfs_event = self.pfs_event.iloc[subset_idx]
        self.gend = self.gend[subset_idx]
        self.age_group = self.age_group[subset_idx]
        self.region = self.region[subset_idx]


    def select_train_subset(self, train_frac=0.2, random_state=42):
        # Select a random subset of the data for training
        np.random.seed(random_state)
        train_idx = np.random.choice(len(self.X), int(len(self.X) * train_frac), replace=False)
        self.train_idx = train_idx
        self.select_subset(train_idx)
        self.test_idx = np.delete(np.arange(len(self.X)), train_idx)
        return self.test_idx