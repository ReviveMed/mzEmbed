#%%
'''
Define the MSPeaks class and functions for creating MSPeaks objects.
MSPeaks should be able to work with peak data from mzlearn, mzmine, and other sources.
this object allows us to unify the data from different sources and perform analyses on it.
'''


import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import feather

import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

DEFAULT_VERBOSE = False

#######################################################
# create a  class called MSPeaks
#######################################################

class MSPeaks:
    def __init__(self):
        self.peak_info = None
        self.peak_intensity = None
        self.missing_val_mask = None
        self.targets_info = None
        self.sample_info = None
        self.num_peaks = np.nan
        self.num_samples = np.nan
        self.peak_freq_th = 0
        self.sample_freq_th = 0
        self.study_id = None
        self.overall_RT_min = np.nan
        self.overall_RT_max = np.nan
        self.overall_MZ_min = np.nan
        self.overall_MZ_max = np.nan

    def _update_basic_info(self):
        self.num_peaks = len(self.peak_info.index)
        self.num_samples = len(self.sample_info.index)

      
    def to_dict(self):  
        return {
                'num_peaks':self.num_peaks,
                'num_samples':self.num_samples,
                'peak_freq_th':self.peak_freq_th,
                'sample_freq_th':self.sample_freq_th,
                'study_id':self.study_id,
                'overall_RT_min':self.overall_RT_min,
                'overall_RT_max':self.overall_RT_max,
                'overall_MZ_min':self.overall_MZ_min,
                'overall_MZ_max':self.overall_MZ_max,
                # 'peak_info':self.peak_info.to_dict(),
                # 'peak_intensity':self.peak_intensity.to_dict(),
                # 'missing_val_mask':self.missing_val_mask.to_dict(),
                # 'targets_info':self.targets_info.to_dict(),
                # 'sample_info':self.sample_info.to_dict()
                'peak_info': self.peak_info,
                'peak_intensity': self.peak_intensity,
                'missing_val_mask': self.missing_val_mask,
                'targets_info': self.targets_info,
                'sample_info': self.sample_info
        }



    def from_dict(self,dict_obj):
        # self.peak_info = pd.DataFrame.from_dict(dict_obj['peak_info'])
        # self.peak_intensity = pd.DataFrame.from_dict(dict_obj['peak_intensity'])
        # self.missing_val_mask = pd.DataFrame.from_dict(dict_obj['missing_val_mask'])
        # self.targets_info = pd.DataFrame.from_dict(dict_obj['targets_info'])
        # self.sample_info = pd.DataFrame.from_dict(dict_obj['sample_info'])
        self.peak_info =dict_obj['peak_info']
        self.peak_intensity = dict_obj['peak_intensity']
        self.missing_val_mask = dict_obj['missing_val_mask']
        self.targets_info = dict_obj['targets_info']
        self.sample_info = dict_obj['sample_info']
        self.num_peaks = dict_obj['num_peaks']
        self.num_samples = dict_obj['num_samples']
        self.peak_freq_th = dict_obj['peak_freq_th']
        self.sample_freq_th = dict_obj['sample_freq_th']
        self.study_id = dict_obj['study_id']
        self.overall_RT_min = dict_obj['overall_RT_min']
        self.overall_RT_max = dict_obj['overall_RT_max']
        self.overall_MZ_min = dict_obj['overall_MZ_min']
        self.overall_MZ_max = dict_obj['overall_MZ_max']

    # def save_to_json(self, file_path):
    #     with open(file_path, 'w') as file:
    #         json.dump(self.to_dict(), file)

    # def load_from_json(self, file_path):
    #     with open(file_path, 'r') as file:
    #         self.from_dict(json.load(file))

    def save_to_pickle(self, file_path):
        # save_dict = self.to_dict()
        with open(file_path, 'wb') as file:
            # pickle.dump(self.to_dict(), file)
            pickle.dump(self.to_dict(), file,pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            self.from_dict(pickle.load(file))

    def save(self,file_path):
        if '.pkl' in file_path:
            self.save_to_pickle(file_path)
        else:
            raise NotImplementedError
        
    def load(self,file_path):
        if '.pkl' in file_path:
            self.load_from_pickle(file_path)
        else:
            raise NotImplementedError


    def add_peak_info(self, info):
        self.peak_info = _data_importer(info)
        if ('mzmed' in self.peak_info.columns):
            self.peak_info['mz'] = self.peak_info['mzmed']
        if ('rtmed' in self.peak_info.columns):
            self.peak_info['rt'] = self.peak_info['rtmed']
        assert 'mz' in self.peak_info.columns, "Error: mz column not found in peak_info."
        assert 'rt' in self.peak_info.columns, "Error: rt column not found in peak_info."
        self.num_peaks = len(self.peak_info.index)
        if self.peak_intensity is not None:
            self.check_peaks()

    def add_sample_info(self, info):
        self.sample_info = _data_importer(info)
        self.num_samples = len(self.sample_info.index)
        if self.peak_intensity is not None:
            self.check_samples()

    def add_peak_intensity(self, intensity):
        self.peak_intensity = _data_importer(intensity)
        self.missing_val_mask = pd.isna(self.peak_intensity)
        if self.peak_info is not None:
            self.check_peaks()
        if self.sample_info is not None:
            self.check_samples()

    def add_params_overall(self, params):
        self.overall_RT_min = params.loc['overall_rt_min'].iloc[0]
        self.overall_RT_max = params.loc['overall_rt_max'].iloc[0]
        self.overall_MZ_min = params.loc['overall_mz_min'].iloc[0]
        self.overall_MZ_max = params.loc['overall_mz_max'].iloc[0]

    def add_targets_info(self, info):
        self.targets_info = _data_importer(info)
        # convert the target info, and update the peak_info

    def check_peaks(self,verbose=DEFAULT_VERBOSE):
        # check that the index of peak_info is equal to the index of peak_intensity
        # if not, then find the common index and subset both peak_info and peak_intensity
        # print which peaks were removed
        common_peaks = set(self.peak_info.index) & set(self.peak_intensity.index)
        removed_peaks = set(self.peak_info.index).union(self.peak_intensity.index) - common_peaks
        common_peaks = list(common_peaks)
        removed_peaks = list(removed_peaks)
        self.peak_info = self.peak_info.loc[common_peaks]
        self.peak_intensity = self.peak_intensity.loc[common_peaks]
        self.missing_val_mask = self.missing_val_mask.loc[common_peaks]
        if verbose:
            print(f"while validating Removed {len(removed_peaks)} peaks:\n{removed_peaks}")
            print(f"while validating, {len(self.peak_info.index)} peaks remain.")
        self.num_peaks = len(self.peak_info)

    def check_samples(self,verbose=DEFAULT_VERBOSE):
        # check that the columns of peak_intensity are equal to the index of sample_info
        # if not, then find the common samples and subset both peak_intensity and sample_info
        # print which samples were removed
        common_samples = set(self.peak_intensity.columns) & set(self.sample_info.index)
        removed_samples = set(self.peak_intensity.columns).union(self.sample_info.index) - common_samples
        common_samples = list(common_samples)
        removed_samples = list(removed_samples)
        self.peak_intensity = self.peak_intensity[common_samples]
        self.sample_info = self.sample_info.loc[common_samples]
        self.missing_val_mask = self.missing_val_mask[common_samples]
        if verbose:
            print(f"while validating, Removed {len(removed_samples)} samples:\n{removed_samples}")
            print(f"while validating, {len(self.sample_info.index)} samples remain.")
        self.num_samples = len(self.sample_info.index)


    def get_peaks(self,freq_th=0,sample_subset=None):
        if self.peak_intensity is None:
            return self.peak_info.index
        # return the peaks that are present in at least freq_th fraction of samples
        freq  = _compute_sample_freq_of_peaks(~self.missing_val_mask,sample_list=sample_subset)
        return self.peak_info.loc[freq>freq_th].index

    def get_samples(self,freq_th=0,peak_subset=None):
        # return the samples that have at least freq_th fraction of peaks
        if self.peak_intensity is None:
            return self.sample_info.index
        freq  = _compute_peak_freq_of_samples(~self.missing_val_mask,peak_list=peak_subset)
        return self.sample_info.loc[freq>freq_th].index

    def get_num_peaks(self,freq_th=0):
        return len(self.get_peaks(freq_th))
    
    def get_num_samples(self,freq_th=0):
        return len(self.get_samples(freq_th))
    
    def get_avg_peak_freq(self,sample_subset=None):
        freq  = _compute_sample_freq_of_peaks(self.peak_intensity,sample_list=sample_subset)
        return freq.mean()

    def apply_freq_th_on_peaks(self,freq_th=0.4,inplace=True,sample_subset=None):
        #suggest to apply this BEFORE apply_freq_th_on_samples
        if inplace:
            peak_subset = self.get_peaks(freq_th, sample_subset)
            self.peak_intensity = self.peak_intensity.loc[peak_subset]
            self.missing_val_mask = self.missing_val_mask.loc[peak_subset]
            self.peak_info = self.peak_info.loc[self.peak_intensity.index]
            self.num_peaks = len(self.peak_info.index)
            self.peak_freq_th = freq_th
            self._update_basic_info()
        else:
            return self.peak_intensity.loc[self.get_peaks(freq_th)]

    def apply_freq_th_on_samples(self,freq_th=0.1,inplace=True,peak_subset=None):
        # assert self.peak_freq_th > 0.2, "Error: peak_freq_th must be greater than 0.2"
        if inplace:
            sample_subset = self.get_samples(freq_th, peak_subset)
            self.peak_intensity = self.peak_intensity[sample_subset]
            self.missing_val_mask = self.missing_val_mask[sample_subset]
            self.sample_info = self.sample_info.loc[self.peak_intensity.columns]
            self.num_samples = len(self.sample_info.index)
            self.sample_freq_th = freq_th
            self._update_basic_info()
        else:
            return self.peak_intensity[self.get_samples(freq_th)]


    def rename_selected_peaks(self,current_ids,new_ids=None,verbose=DEFAULT_VERBOSE):
        if new_ids is None:
            assert isinstance(current_ids, dict), "Error: if new_ids is None, current_ids must be a dict."
            new_ids = current_ids.values()
            current_ids = current_ids.keys()

        # useful when we want to rename peaks to match a standard, such as when doing study-alignment
        if 'original_id' not in self.peak_info.columns:
            self.peak_info['original_id'] = self.peak_info.index
        else:
            self.peak_info['previous_id'] = self.peak_info.index
        
        removed_ids  = set(self.peak_info.index) - set(current_ids)
        if verbose: print(f"while renaming peaks, Removed {len(removed_ids)} peaks:\n{removed_ids}")
        self.peak_info = self.peak_info.loc[current_ids,:]
        self.peak_info.index = new_ids
        self.peak_intensity = self.peak_intensity.loc[current_ids,:]
        self.peak_intensity.index = new_ids
        self.missing_val_mask = self.missing_val_mask.loc[current_ids,:]
        self.missing_val_mask.index = new_ids
        self._update_basic_info()
        return

    def rename_selected_samples(self,current_ids,new_ids=None,verbose=DEFAULT_VERBOSE):
        if new_ids is None:
            assert isinstance(current_ids, dict), "Error: if new_ids is None, current_ids must be a dict."
            new_ids = current_ids.values()
            current_ids = current_ids.keys()
        # useful when we want to rename samples to match a standard, such as when doing study-alignment
        removed_ids  = set(self.sample_info.index) - set(current_ids)
        if verbose: print(f"while renaming samples, Removed {len(removed_ids)} samples:\n{removed_ids}")
        self.sample_info = self.sample_info.loc[current_ids,:]
        self.sample_info.index = new_ids
        self.peak_intensity = self.peak_intensity[current_ids]
        self.peak_intensity.columns = new_ids
        self.missing_val_mask = self.missing_val_mask[current_ids]
        self.missing_val_mask.columns = new_ids
        self._update_basic_info()
        return

    def add_study_id(self,study_id,rename_samples=True,verbose=DEFAULT_VERBOSE):
        # add the study name to the sample_info and optionally rename the samples
        self.study_id = study_id
        if 'study_id' in self.sample_info.columns:
            print("Warning: study_id already exists in sample_info. Overwriting.")
        self.sample_info['study_id'] = study_id
        if rename_samples:
            if verbose: print("Renaming samples to include study_id")
            current_ids = self.sample_info.index.tolist()
            new_ids = [study_id+'_'+x for x in current_ids]
            self.rename_selected_samples(current_ids,new_ids,verbose=verbose)
        return

    def plot_sample_freq(self):
        freq = _compute_sample_freq_of_peaks(self.peak_intensity)
        plt.hist(freq)
        plt.xlabel('Fraction of samples with value')
        plt.ylabel('Number of Peaks')
        plt.show()

    def plot_peak_freq(self):
        freq = _compute_peak_freq_of_samples(self.peak_intensity)
        plt.hist(freq)
        plt.xlabel('Fraction of peaks with value')
        plt.ylabel('Number of Samples')
        plt.show()

    def fill_missing_values(self,method='knn'):
        # fill missing values in peak_intensity
        # method can be 'mean', 'median', or 'zero'
        if isinstance(method, str):
            self.peak_intensity = _impute_peaks_missing_val(self.peak_intensity,fill_na_method=method)
            # peak_intensity_T= self.peak_intensity.T
            # if method == 'mean':
            #     self.peak_intensity = peak_intensity_T.fillna(peak_intensity_T.mean()).T
        elif isinstance(method, float):
            self.peak_intensity = self.peak_intensity.fillna(method)
            
        else:
            print(f"Error: method {method} not supported.")
        
    def create_hidden_refs(self,hidden_frac=0.2):
        assert 'Ref' in self.sample_info.columns, "Error: Ref column not found in sample_info."

        raise NotImplementedError

    def normalize_intensity(self,norm_func=None):
        if norm_func is None:
            print('no normalization applied')
            return
        assert callable(norm_func), "Error: norm_func must be a callable function."
        # self.peak_intensity = norm_func(peak_intensity=self.peak_intensity,
        #                                 peak_info=self.peak_info,
        #                                 sample_info=self.sample_info)
        
        self.peak_intensity = norm_func(self.peak_intensity,
                                self.peak_info,
                                self.sample_info)
        # return self.peak_intensity


        return

    def pca_plot_samples(self,save_path=None,plot_name=None):
        pca_df = get_pca(self.peak_intensity)
        pca_df = add_sample_info_to_df(pca_df, self.sample_info)
        pca_score = score_reduc_dim_mix(pca_df,dim_name='PC',label_col='run_order',n_neighbors=10,n_iter=3)
        if plot_name is None:
            num_samples = len(self.sample_info.index)
            num_feats = len(self.peak_info.index)
            plot_name = f'{self.study_id} PCA of Samples ({num_samples}, {num_feats} feats), mix score={pca_score:.2f}'
        plot_reduc_dim(plot_name,pca_df,'PC',
                hue_id='run_order',
                save_path=save_path)
        return

    def pca_plot_peaks(self):
        raise NotImplementedError
    
    def umap_plot_samples(self,save_path=None,plot_name=None):
        umap_df = get_umap(self.peak_intensity)
        umap_df = add_sample_info_to_df(umap_df, self.sample_info)
        umap_score = score_reduc_dim_mix(umap_df,dim_name='UMAP',label_col='run_order',n_neighbors=10,n_iter=3)
        if plot_name is None:
            num_samples = len(self.sample_info.index)
            num_feats = len(self.peak_info.index)
            plot_name = f'{self.study_id} UMAP of Samples ({num_samples}, {num_feats} feats), mix score={umap_score:.2f}'
        plot_reduc_dim(plot_name,umap_df,'UMAP',
                hue_id='run_order',
                save_path=save_path)
        return

    def umap_plot_peaks(self):
        raise NotImplementedError
    
    def get_target_matches(self):
        raise NotImplementedError
    
    def remove_outlier_samples(self,method=None):
        if method is None:
            method = 'low_frequency'
        
        if method == 'low_frequency':
            # remove super low frequency peaks
            self.apply_freq_th_on_peaks(freq_th=0.1,inplace=True)
            # remove samples with too few peaks
            self.apply_freq_th_on_samples(freq_th=0.1,inplace=True)
        else:
            raise NotImplementedError
    
    def remove_outlier_peaks(self):
        raise NotImplementedError
    

#######################################################
######### Creation Functions #########
#######################################################

# def load_mspeaks_from_pickle(filename):
#     with open(filename, 'rb') as f:
#         return pickle.load(f)


def create_mspeaks(peak_info=None, sample_info=None, peak_intensity=None, targets_info=None, params_overall=None):

    # create an empty MSPeaks object
    mspeaks_obj = MSPeaks()

    # add the peak_info, sample_info, peak_intensity, and targets_info
    if peak_info is not None:
        mspeaks_obj.add_peak_info(peak_info)
    if sample_info is not None:
        mspeaks_obj.add_sample_info(sample_info)
    if peak_intensity is not None:
        mspeaks_obj.add_peak_intensity(peak_intensity)
    if targets_info is not None:
        mspeaks_obj.add_targets_info(targets_info)
    if params_overall is not None:
        mspeaks_obj.add_params_overall(params_overall)

    return mspeaks_obj


def create_mspeaks_from_mzlearn_result(result_dir,peak_subdir='final_peaks',peak_intensity='intensity_max'):
    # read in the peak_info
    peak_info_path = os.path.join(result_dir,peak_subdir,'peak_info.csv')
    if not os.path.exists(peak_info_path):
        peak_info_path = os.path.join(result_dir,peak_subdir,'extra_data.csv')
    if os.path.exists(peak_info_path):
        peak_info = pd.read_csv(peak_info_path, index_col=0)
    else:
        peak_info = None

    # read in the sample_info
    try:
        sample_info = pd.read_csv(os.path.join(result_dir,'sample_info','sample_info.csv'), index_col=0)
    except FileNotFoundError:
        sample_info = None

    # read in the peak_intensity
    try:
        peak_intensity = pd.read_csv(os.path.join(result_dir,peak_subdir,f'{peak_intensity}.csv'), index_col=0)
    except FileNotFoundError:
        peak_intensity = None

    # read in the targets_info
    try:
        targets_info = pd.read_csv(os.path.join(result_dir, 'targets.csv'), index_col=0)
    except FileNotFoundError:
        targets_info = None
    
    # read in the overall_params
    try:
        params_overall = pd.read_csv(os.path.join(result_dir, 'params_overall.csv'), index_col=0).iloc[1:,:]
        params_overall = params_overall.astype(float)
    except FileNotFoundError:
        params_overall = None

    # create the mspeaks object
    return create_mspeaks(peak_info, sample_info, peak_intensity, targets_info, params_overall)


#######################################################
########### Helper functions ###########
#######################################################


def _compute_sample_freq_of_peaks(df,sample_list=None):
    if sample_list is None:
        sample_list = df.columns
    # check if datatype is boolean
    if df.dtypes.iloc[0] == bool:
        freq_vals = df[sample_list].mean(axis=1)
    else:        
        freq_vals = 1- df[sample_list].isnull().sum(axis=1)/len(sample_list)
    return freq_vals 

def _compute_peak_freq_of_samples(df,peak_list=None):
    if peak_list is None:
        peak_list = df.index
    # check if datatype is boolean
    if df.dtypes[0] == bool:
        freq_vals = df.loc[peak_list].mean(axis=0)
    else:
        freq_vals = 1- df.loc[peak_list].isnull().sum(axis=0)/len(peak_list)
    return freq_vals

def _data_importer(data):
    '''
    This function takes in one variable "data". 
    If data is a string, assume it is a path, attempts to open the file with the given path 
    and read its contents. If the file is not found, it prints an error 
    message and returns None. If data is not a string, it assumes that it is already 
    in the correct form and simply returns the value.
    it prints an error message and returns None. 
    If the file path exists, it checks the file extension and opens 
    the file appropriately depending on whether it is a .txt, .csv, or .tsv file. 
    If the file format is not supported, it prints an error message and returns None.
    '''
    if isinstance(data, str):
        # assume that the data is a file path
        if not os.path.exists(data):
            print(f"Error: file {data} not found.")
            return None
        else:
            if data.endswith('.txt'):
                with open(data, 'r') as f:
                    return f.read()
            elif data.endswith('.csv'):
                return pd.read_csv(data, index_col=0)
            elif data.endswith('.tsv'):
                return pd.read_csv(data, delimiter='\t', index_col=0)
            else:
                print(f"Error: unsupported file format for {data}.")
                return None
    else:
        return data


def _impute_peaks_missing_val(peaks_df,fill_na_method='knn',**kwargs):
    backup_fillval = peaks_df.median(axis=0).median()
    yes_log = False
    if 'log_' in fill_na_method:
        peaks_df = np.log2(peaks_df)
        yes_log = True
        fill_na_method = fill_na_method.replace('log_','')

    if fill_na_method == 'zero':
        # temporarily impute nan values with 0
        peaks_df = peaks_df.fillna(0)
    elif fill_na_method == 'min':
        peaks_df = ((peaks_df.T).fillna(peaks_df.T.min())).T
    elif fill_na_method == 'min/2':
        peaks_df = ((peaks_df.T).fillna(peaks_df.T.min()/2)).T
    elif fill_na_method == 'mean':
        # temporarily impute nan values with the feature average        
        peaks_df = ((peaks_df.T).fillna(peaks_df.T.mean())).T
    elif fill_na_method == 'median':
        # temporarily impute nan values with the feature median    
        peaks_df = ((peaks_df.T).fillna(peaks_df.T.median())).T
    elif fill_na_method == 'mean_sample':
        # temporarily impute nan values with the sample-wide average, NOT RECOMMENDED      
        peaks_df = peaks_df.fillna(peaks_df.mean(axis=0))
    elif fill_na_method == 'median_sample':
        # temporarily impute nan values with the sample-wide median, NOT RECOMMENDED
        peaks_df = peaks_df.fillna(peaks_df.median(axis=0))
    elif fill_na_method == 'min_sample':
        # temporarily impute nan values with the sample-wide median, NOT RECOMMENDED
        peaks_df = peaks_df.fillna(peaks_df.min(axis=0))
    elif fill_na_method == 'knn':
        if 'n_neighbors' in kwargs:
            n_neighbors = kwargs['n_neighbors']
        else: n_neighbors = 5
        imp = KNNImputer(n_neighbors=n_neighbors,keep_empty_features=True)
        # peaks_df = imp.fit_transform(peaks_df.T).T
        peaks_df = pd.DataFrame( imp.fit_transform(peaks_df.T).T,index=peaks_df.index,columns=peaks_df.columns)
    else:
        raise ValueError(f'fill_na_method ({fill_na_method}) not recognized')
    peaks_df = peaks_df.fillna(backup_fillval)

    if yes_log:
        peaks_df = 2**peaks_df

    return peaks_df


#######################################################
########### Helper functions for plotting ###########
#######################################################
# %%


def prep_for_reduc_dim(peaks_df):
    data = np.log2(peaks_df.T)
    data = StandardScaler().fit_transform(data)
    data = pd.DataFrame(data, index=peaks_df.columns, columns=peaks_df.index)
    data.fillna(0, inplace=True)
    return data


def get_pca(peaks_df):
    data = prep_for_reduc_dim(peaks_df)
    pca = PCA(n_components=2)
    x_pca = np.array(pca.fit_transform(data))

    pca_df = pd.DataFrame(x_pca, index=peaks_df.columns, columns=['PC1', 'PC2'])
    return pca_df


def get_umap(peaks_df,n_neighbors=50):
    data = prep_for_reduc_dim(peaks_df)
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(data)
    umap_df = pd.DataFrame(embedding, index=peaks_df.columns, columns=['UMAP1', 'UMAP2'])
    return umap_df


def add_sample_info_to_df(reduc_dim_df, sample_info):
    reduc_dim_df['Ref'] = sample_info['Ref']
    reduc_dim_df['run_order'] = sample_info['run_order']
    reduc_dim_df['batch_id'] = sample_info['batch_id']
    if 'True_Ref' in sample_info.columns:
        reduc_dim_df['True_Ref'] = sample_info['True_Ref']
        reduc_dim_df['Hidden_Ref'] = sample_info['Hidden_Ref']
    return reduc_dim_df


def score_reduc_dim_mix(reduc_dim_df,
                        dim_name='PC',
                        label_col='run_order',
                        n_iter=3,
                        n_neighbors=10):
    neigh_reg = KNeighborsRegressor(n_neighbors=n_neighbors)

    scores= []
    X = reduc_dim_df[[f'{dim_name}1',f'{dim_name}2']].values
    labels = reduc_dim_df[label_col].values
    n_iter = n_iter
    for ii in range(1,n_iter):
        rand_perm = np.random.permutation(len(X))
        X = X[rand_perm]
        labels = labels[rand_perm]
        kf = KFold(n_splits=10)
        for i, (train_index,test_index) in enumerate(kf.split(X)):
            xtrain, xtest = X[train_index], X[test_index]
            ytrain, ytest = labels[train_index], labels[test_index]
            neigh_reg.fit(xtrain, ytrain)
            scores.append(neigh_reg.score(xtest, ytest))

    score_avg = np.clip(np.mean(scores),0,1)
    score_res = 1-score_avg
    return score_res


def score_reduc_dim_dist(reduc_dim_df,
                         dim_name='PC',
                         label_col='Ref',
                         n_sample=20):
    
    X = reduc_dim_df[[f'{dim_name}1',f'{dim_name}2']].values
    rand_idx = np.random.permutation(len(X))
    n_sample = min(n_sample,len(X))
    all_sample_dists = np.mean(np.concatenate([np.linalg.norm(X-X[rand_idx[i],:],axis=1) for i in range(n_sample)]))

    pool_val_list = reduc_dim_df[label_col].unique()
    # order the pool values and remove 0
    
    n_pools_total = np.sum(reduc_dim_df[label_col]>0)
    pool_val_list = np.sort(pool_val_list)[1:]
    ref_sample_dist_frac_tot = 0
    for pool_val in pool_val_list:
        X_pool = X[reduc_dim_df[label_col]==pool_val,:]
        rand_idx = np.random.permutation(len(X_pool))
        n_subset_pools = X_pool.shape[0]
        n_sample = min(n_sample,len(X_pool))
        ref_sample_dists = np.mean(np.concatenate([np.linalg.norm(X_pool-X_pool[rand_idx[i],:],axis=1) for i in range(n_sample)]))
        ref_sample_dist_frac = ref_sample_dists/all_sample_dists

        ref_sample_dist_frac_tot += ref_sample_dist_frac*n_subset_pools/n_pools_total


    return ref_sample_dist_frac_tot

def plot_reduc_dim(plot_title,reduc_dim_df,dim_name=None,
                   hue_id='run_order',save_path=None,style_col='Ref',size_col='Ref'):
    
    sns.set_context('talk')
    # set the default figure size
    # sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set(rc={'figure.figsize':(9.8,6.7)})
    sns.set_style('whitegrid')
    reduc_dim_df['Ref'] = reduc_dim_df['Ref'].astype(bool)
    if 'True_Ref' in reduc_dim_df.columns:
        reduc_dim_df['True_Ref'] = reduc_dim_df['True_Ref'].astype(bool)
        reduc_dim_df['Hidden_Ref'] = reduc_dim_df['Hidden_Ref'].astype(bool)
    sz = 100/np.log10(reduc_dim_df.shape[0])
    markers = {True:'o',False:'X'}
    # markers = {True:'o',False:'s'}
    sizes = {True:1.5*sz,False:sz}
    
    palette_id = 'turbo'
    if hue_id == 'batch_id':
        palette_id = 'tab10'
        if reduc_dim_df['batch_id'].max() > 10:
            palette_id = 'viridis'

    if dim_name is None:
        if 'PC1' in reduc_dim_df.columns:
            dim_name = 'PC'
        elif 'UMAP1' in reduc_dim_df.columns:
            dim_name = 'UMAP'
        else:
           raise ValueError('Could not determine dimension name. Please specify manually.')

    sns.scatterplot(data=reduc_dim_df,x=f'{dim_name}1',y=f'{dim_name}2',hue=hue_id,
                    palette=palette_id,s=sz,style=style_col,
                    markers=markers,size=size_col,sizes=sizes,alpha=0.8)
    

    plt.title(plot_title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,dpi=300,bbox_inches='tight')
        plt.close()
    return

# # %%
# if __name__ == "__main__":

#     # %%
#     result_dir = '/Users/jonaheaton/Desktop/Data-engine/ST001428/result_hilic_pos_2023-09-28-01-29-57-'
#     st001428 = create_mspeaks_from_mzlearn_result(result_dir)
#     # %%


#     st001428.get_samples(0.1)
#     # %%

#     st001428.plot_peak_freq()
#     # %%
#     st001428.get_peaks(0.4)
#     # %%
#     st001428.plot_sample_freq()
#     # %%
