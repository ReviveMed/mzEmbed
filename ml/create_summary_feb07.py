# %%
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# %%

base_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton/development_CohortCombination'
if not os.path.exists(base_dir):
    base_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination'

date_name = 'hilic_pos_2024_feb_05_read_norm_poolmap'
feat_subset_name = 'num_cohorts_thresh_0.5'
study_subset_name= 'subset all_studies with align score 0.25 from Merge_Jan25_align_80_40_fillna_avg'
task_name = 'std_1_Multi'
input_dir = f'{base_dir}/{date_name}/{study_subset_name}/{feat_subset_name}/{task_name}'

# %%
pretrain_model_subdir = 'pretrain_models_feb07'

cv_splits = 5
cv_seed = 42
y_stratify_col = 'Benefit'
cv_subdir = f'CV{cv_splits}_seed{cv_seed}_on_{y_stratify_col}'


summary_result_dir = os.path.join(input_dir, cv_subdir)
pytroch_result_dir = os.path.join(input_dir, cv_subdir,'cross_val_split_0_0/pytorch_models')

# %%

out_cvres_list = []
summary_files = [f for f in os.listdir(summary_result_dir) if f.endswith('summary.csv')]
for f in summary_files:
    df = pd.read_csv(f'{summary_result_dir}/{f}', index_col=0)
    model_id = f.split('_summary.csv')[0]
    # output_dict = df.mean().to_dict()
    output_dic = df.iloc[:,0].to_dict()
    output_dic['model_id'] = model_id

    out_cvres_list.append(output_dic)

    
cvres_summary_df = pd.DataFrame(out_cvres_list)    
cvres_summary_df.set_index('model_id', inplace=True)
new_cols = [(f'CV {cv_splits}-fold', col) for col in cvres_summary_df.columns]
cvres_summary_df.columns = pd.MultiIndex.from_tuples(new_cols, names=['0th level', '1st level'])

# %%

model_id_list = cvres_summary_df.index.tolist()
out_dict_list = []
for model_id in model_id_list:
    result_file = f'{pytroch_result_dir}/{model_id}_output.json'
    if not os.path.exists(result_file):
        continue

    result_dict = json.load(open(result_file))
    out_dict = {}
    out_dict['basic info'] = {}
    ignore_keys_list = ['best_val_loss', 'best_epoch']
    for key, value in result_dict.items():
        if 'history' in key:
            continue
        if 'path' in key:
            continue
        if key in ignore_keys_list:
            continue

        # if key == 'model_hyperparameters':

        if isinstance(value,dict):
            out_dict[key] = value
            continue

        out_dict['basic info'][key] = value

        # load any pre-training information
        if key == 'encoder_name':
            pretrain_result_file = f'{input_dir}/{pretrain_model_subdir}/{value}_output.json'
            if os.path.exists(pretrain_result_file):
                pretrain_model_dict = json.load(open(pretrain_result_file))
                for pretrain_key, pretrain_value in pretrain_model_dict.items():
                    if 'history' in pretrain_key:
                        continue
                    if 'path' in pretrain_key:
                        continue
                    if pretrain_key in ignore_keys_list:
                        continue
                    if isinstance(pretrain_value,dict):
                        out_dict[f'PRETRAIN {pretrain_key}'] = pretrain_value
                        continue
                    # out_dict['pretrain_info'] = {}
                    # out_dict['pretrain_info'][pretrain_key] = pretrain_value


    out_dict_flatten = {}
    out_dict_flatten['model_id'] = model_id
    for key, value in out_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out_dict_flatten[(key,sub_key)] = sub_value
        else:
            out_dict_flatten[(key,key)] = value 


    out_dict_list.append(out_dict_flatten)    


param_summary_df = pd.DataFrame(out_dict_list)
param_summary_df.set_index('model_id', inplace=True)
param_summary_df.columns = pd.MultiIndex.from_tuples(param_summary_df.columns, names=['0th level', '1st level'])

# %%


full_summary_df = pd.concat([cvres_summary_df, param_summary_df], axis=1)
# remove any rows whose (CV 5-fold, AVG val_auroc) is NaN
full_summary_df.dropna(subset=[(f'CV {cv_splits}-fold', 'AVG val_auroc')], inplace=True)
# drop any columns that are all NaN
full_summary_df.dropna(axis=1, how='all', inplace=True)

# some other cleanup
# round the CV-fold columns to 3 decimal places
cv_cols = [col for col in full_summary_df.columns if 'CV' in col[0]]
full_summary_df[cv_cols] = full_summary_df[cv_cols].round(4)

# sort the columns
full_summary_df = full_summary_df.reindex(sorted(full_summary_df.columns), axis=1)

# put the "end_state" columns at the end
end_state_cols = [col for col in full_summary_df.columns if 'end_state' in col[0]]
non_end_state_cols = [col for col in full_summary_df.columns if 'end_state' not in col[0]]
full_summary_df = full_summary_df[non_end_state_cols + end_state_cols]

# put the "PRETRAIN" columns at the end
pretrain_cols = [col for col in full_summary_df.columns if 'PRETRAIN' in col[0]]
non_pretrain_cols = [col for col in full_summary_df.columns if 'PRETRAIN' not in col[0]]
full_summary_df = full_summary_df[non_pretrain_cols + pretrain_cols]

full_summary_df.to_excel(f'{input_dir}/model_summary_{y_stratify_col}.xlsx',header=True, index=True)