# %%
import os
import pandas as pd
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# %%
data_path = '/Users/jonaheaton/Desktop/Data-engine'
origin_name = 'ST001236_and_ST001237'
origin_result_path = os.path.join(data_path,origin_name,'rcc1_rcc3_output_v1_2023-09-28-21-22-21-')
# origin_rcc_metadata_file = os.path.join(data_path,origin_name,'sample_info.csv')
origin_rcc_metadata_file = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination/data_2023_november/rcc_sample_info3.csv'

rcc_metadata = pd.read_csv(origin_rcc_metadata_file,index_col=0)
select_files = rcc_metadata[rcc_metadata['study_week'] =='baseline'].index.tolist()
finetune_label_col = 'Sex'
# %%
train_select_files = rcc_metadata[
                             (rcc_metadata['phase']==3) &
                             (rcc_metadata[finetune_label_col].isin(['M','F']))
                             ].index.tolist()

test_select_files = rcc_metadata[
                             (rcc_metadata['phase']==1) &
                             (rcc_metadata[finetune_label_col].isin(['M','F']))
                             ].index.tolist()


# fraction of the cohorts a feature must be found to be considered for downstream analysis
num_cohorts_thresh = 0.5

# How to correct for cohort effects?
cohort_correction_method = 'combat'

# %% What is the fine-tuning task?
task_name = f'{cohort_correction_method}_{finetune_label_col}'

training_files  = train_select_files
test_files = test_select_files
validation_files = []
validation_frac = 0.2
validation_rand_seed= 42



selected_studies_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination/hilic_pos_2023_nov_21/subset all_studies'
feat_thresh_name = 'num_cohorts_thresh_{:.1f}'.format(num_cohorts_thresh)
select_feats_dir = os.path.join(selected_studies_dir,feat_thresh_name)
peak_intensity_file = os.path.join(select_feats_dir,'peak_intensity_combat.csv')

data_corrected = pd.read_csv(peak_intensity_file,index_col=0)

full_descrip_df_file = os.path.join(select_feats_dir,'combined_metadata.csv')
full_descrip_df = pd.read_csv(full_descrip_df_file,index_col=0)
# %% Create the task directory
task_dir = os.path.join(select_feats_dir,task_name)
os.makedirs(task_dir,exist_ok=True)


pretrain_label_col = 'label6'

input_data = data_corrected.T
input_pretrain_labels = full_descrip_df[pretrain_label_col].copy()
if finetune_label_col not in full_descrip_df.columns:
    full_descrip_df[finetune_label_col] = pd.NA
    full_descrip_df.loc[rcc_metadata.index,finetune_label_col] = rcc_metadata[finetune_label_col]

input_finetune_labels = full_descrip_df[finetune_label_col].copy()

# %%
pretrain_files = [x for x in input_data.index if x not in training_files+test_files]

y_pretrain = input_pretrain_labels.loc[pretrain_files].copy()
y_finetune = input_finetune_labels.loc[training_files].copy()
y_test = input_finetune_labels.loc[test_files].copy()

X_pretrain = input_data.loc[pretrain_files,:].copy()
X_finetune = input_data.loc[training_files,:].copy()
X_test = input_data.loc[test_files,:].copy()


if len(validation_files) == 0:
    try:
        X_train, X_val, y_train, y_val = train_test_split(X_finetune, y_finetune, 
                                                        test_size=validation_frac, 
                                                        random_state=validation_rand_seed, 
                                                        stratify=y_finetune)
    except ValueError:
        # y_finetune is not a format that can be stratified
        X_train, X_val, y_train, y_val = train_test_split(X_finetune, y_finetune, 
                                                        test_size=validation_frac, 
                                                        random_state=validation_rand_seed, 
                                                        stratify=None)
    validation_files = X_val.index.tolist()
else:
    X_train = X_finetune.copy()
    y_train = y_finetune.copy()
    X_val = input_data.loc[:,validation_files].copy()
    y_val = input_finetune_labels.loc[validation_files].copy()

X_pretrain.to_csv(os.path.join(task_dir,'X_pretrain.csv'))
X_train.to_csv(os.path.join(task_dir,'X_train.csv'))
X_val.to_csv(os.path.join(task_dir,'X_val.csv'))
X_test.to_csv(os.path.join(task_dir,'X_test.csv'))

y_pretrain.to_csv(os.path.join(task_dir,'y_pretrain.csv'))
y_train.to_csv(os.path.join(task_dir,'y_train.csv'))
y_val.to_csv(os.path.join(task_dir,'y_val.csv'))
y_test.to_csv(os.path.join(task_dir,'y_test.csv'))
