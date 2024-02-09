import pandas as pd
import numpy as np
from inmoose.pycombat import pycombat_norm
from sklearn.preprocessing import LabelEncoder, StandardScaler


def min_max_scale(df):
    overall_min = df.min().min()
    overall_max = df.max().max()
    df = (df-overall_min)/(overall_max-overall_min)
    return df


def fill_na_by_cohort(combined_intensity,cohort_labels,method='mean_1th'):
    assert len(combined_intensity.columns) == len(cohort_labels)
    # columns are samples, rows are features
    for cohort in set(cohort_labels):
        # cohort_idx = np.where(cohort_labels==cohort)[0]
        cohort_idx = np.array(cohort_labels) == cohort
        cohort_data = combined_intensity.iloc[:,cohort_idx].copy()
        not_nan_fts = cohort_data.notna().sum(axis=1) > 0
        cohort_data = cohort_data.loc[not_nan_fts].copy()

        if method == 'mean_0th':
            # average across the features
            cohort_data.fillna(cohort_data.mean(),inplace=True)
        elif method == 'mean_1th':
            # average across the samples
            cohort_dataT = cohort_data.T
            cohort_dataT.fillna(cohort_dataT.mean(),inplace=True)
            cohort_data = cohort_dataT.T
        elif method == 'min_0th':
            cohort_data.fillna(cohort_data.min(),inplace=True)
        elif method == 'min_1th':
            cohort_dataT = cohort_data.T
            cohort_dataT.fillna(cohort_dataT.min(),inplace=True)
            cohort_data = cohort_dataT.T
        else:
            raise ValueError(f'Invalid method: {method}')

        combined_intensity.loc[cohort_data.index].iloc[:,cohort_idx] = cohort_data
    return combined_intensity



def standardize_across_cohorts(combined_intensity,cohort_labels,method):
    assert len(combined_intensity.columns) == len(cohort_labels)
    combined_intensity.fillna(combined_intensity.mean(),inplace=True)

    if method == 'combat':
        print('fill missing values with the sample mean')
        data_corrected = pycombat_norm(combined_intensity,cohort_labels)
    elif method == 'raw':
        print('fill missing values with the sample mean')
        data_corrected = combined_intensity.copy()
    elif method =='min_max':
        print('fill missing values with the sample mean')
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(cohort_labels==cohort)[0]
            cohort_data = data_corrected.iloc[:,cohort_idx].copy()
            cohort_data = min_max_scale(cohort_data)
            data_corrected.iloc[:,cohort_idx] = cohort_data

    elif method == 'zscore_0':
        print('avg along 0th dim:', combined_intensity.mean(axis=0).shape)
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(np.array(cohort_labels)==cohort)[0]
            cohort_data = data_corrected.iloc[:,cohort_idx].copy()
            cohort_data.fillna(cohort_data.mean(),inplace=True)
            cohort_data = StandardScaler().fit_transform(cohort_data)
            # cohort_data = (cohort_data-cohort_data.mean(axis=0))/cohort_data.std(axis=0)
            data_corrected.iloc[:,cohort_idx] = cohort_data
    elif method == 'zscore_1':
        print('avg along 1th dim:', combined_intensity.mean(axis=1).shape)
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(np.array(cohort_labels)==cohort)[0]
            cohort_data = data_corrected.iloc[:,cohort_idx].copy()
            # (cohort_data.T).fillna(cohort_data.T.mean(),inplace=True)
            cohort_data.fillna(cohort_data.mean(),inplace=True)
            cohort_data = StandardScaler().fit_transform(cohort_data.T).T
            # cohort_data = (cohort_data-cohort_data.mean(axis=1))/cohort_data.std(axis=1)
            # cohort_data.fillna(0,inplace=True)
            data_corrected.iloc[:,cohort_idx] = cohort_data
    
    elif method == 'std_0':
        data_corrected = combined_intensity.copy()
        for cohort in set(cohort_labels):
            cohort_idx = np.where(np.array(cohort_labels)==cohort)[0]
            cohort_data = data_corrected.iloc[:,cohort_idx].copy()
            cohort_data = StandardScaler().fit_transform(cohort_data)
            data_corrected.iloc[:,cohort_idx] = cohort_data     
    
    elif method == 'std_1':     
        data_corrected = combined_intensity.copy()    
        for cohort in set(cohort_labels):
            cohort_idx = np.where(np.array(cohort_labels)==cohort)[0]
            cohort_data = data_corrected.iloc[:,cohort_idx].copy()
            cohort_data = StandardScaler().fit_transform(cohort_data.T).T
            data_corrected.iloc[:,cohort_idx] = cohort_data
    else:
        raise ValueError(f'Invalid method: {method}')

    return data_corrected
