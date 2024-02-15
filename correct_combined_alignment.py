################
# %% Preamble
################
import os
import pandas as pd
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
from study_alignment.standardize import min_max_scale, standardize_across_cohorts, fill_na_by_cohort

# %%
dropbox_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/'
cohort_combine_dir = os.path.join(dropbox_dir,'development_CohortCombination')
norm_setting_dir = os.path.join(cohort_combine_dir,'norm_settings')
# output_dir = f'{cohort_combine_dir}/hilic_pos_2024_feb_02_read_norm'
output_dir = f'{cohort_combine_dir}/hilic_pos_2024_feb_09_read_norm_poolmap'

alignment_dir_name = ''
align_score_th = 0.3
selected_subset_name = f'all_studies with align score {align_score_th}'

selected_subset_name = selected_subset_name + ' from ' + alignment_dir_name
num_cohorts_thresh = 0.5

selected_studies_dir = os.path.join(output_dir,f'subset {selected_subset_name}')
feat_thresh_name = f'num_cohorts_thresh_{num_cohorts_thresh}'
select_feats_dir = os.path.join(selected_studies_dir,feat_thresh_name)

# %%
origin_name = 'ST001236_and_ST001237'
origin_metadata_file = f'{cohort_combine_dir}/clean_rcc_metadata.csv'
origin_metadata = pd.read_csv(origin_metadata_file,index_col=0)
# rename the index of metadata to include the study name
origin_metadata.index = [f'{origin_name}_{y}' for y in origin_metadata.index]

# %%

# finetune_label_col_name = 'Benefit'
# finetune_label_col_list = [finetune_label_col_name]

# finetune_label_col_name = 'MSKCC'
# finetune_label_col_list = [finetune_label_col_name]

finetune_label_col_name = 'Multi'
finetune_label_col_list = ['Benefit','PFS','PFS_Event','OS','OS_Event','ORR','ExtremeResponder','MSKCC','Treatment',\
                           'Sex','Region','study_week','Dose (mg/kg)','phase','Age_Group','Prior_2','Age']


train_select_files =origin_metadata[(origin_metadata['phase']=='RCC3')
                                     & (origin_metadata['Treatment'].isin(['NIVOLUMAB','EVEROLIMUS']))
                                     & (origin_metadata['study_week']== 'baseline')
                                     ].index.tolist() 
test_select_files = origin_metadata[(origin_metadata['phase']=='RCC1')
                                     & (origin_metadata['Treatment']=='NIVOLUMAB')
                                     & (origin_metadata['study_week']== 'baseline')
                                     ].index.tolist()
test_select_file_alt = []

# cohort_correction_method = 'combat'
cohort_correction_method = 'std_1'
# cohort_correction_method = 'zscore_1'
task_name = f'{cohort_correction_method}_{finetune_label_col_name}'

dataset_labels = f'{dropbox_dir}/Spreadsheets/pretraining datasets Q4/pretraining_dataset_labels.csv'
labels_df = pd.read_csv(dataset_labels,index_col=0)
# pretrain_label_col = 'label6'
pretrain_label_col = 'Sex'

training_files  = train_select_files
test_files = test_select_files
validation_files = []
validation_frac = 0.2
validation_rand_seed= 42



# %%

if os.path.exists(os.path.join(select_feats_dir,'combined_study.csv')):
    combined_study = pd.read_csv(os.path.join(select_feats_dir,'combined_study.csv'),index_col=0)
    metadata_df = pd.read_csv(os.path.join(select_feats_dir,'combined_metadata.csv'),index_col=0)
else:
    raise ValueError('combined_study.csv does not exist in select_feats_dir')




cohort_labels = metadata_df['Study_num'].to_list()

# correct for cohort effects
if os.path.exists(os.path.join(select_feats_dir,f'peak_intensity_{cohort_correction_method}.csv')):
    data_corrected = pd.read_csv(os.path.join(select_feats_dir,f'peak_intensity_{cohort_correction_method}.csv'),index_col=0)
else:
    data_corrected = fill_na_by_cohort(combined_study,cohort_labels)
    data_corrected = standardize_across_cohorts(data_corrected,cohort_labels,method=cohort_correction_method)
    data_corrected.to_csv(os.path.join(select_feats_dir,f'peak_intensity_{cohort_correction_method}.csv'))

# check that there are no null values with the labels
print(metadata_df[pretrain_label_col].isnull().sum())
metadata_df[pretrain_label_col].to_csv(f'{select_feats_dir}/sample_{pretrain_label_col}.csv')

####################
# %% Look at the PCA of the combined study
####################
if False:
    from sklearn.decomposition import PCA
    import seaborn as sns

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_corrected.T)

    pca_df = pd.DataFrame(pca_result,columns=['PC1','PC2'],index=data_corrected.columns)
    pca_df['Study'] = metadata_df['Study']
    pca_df['Study_num'] = metadata_df['Study_num']
    pca_df['label'] = metadata_df[pretrain_label_col]

    pca_df.to_csv(os.path.join(select_feats_dir,f'pca_df_{cohort_correction_method}.csv'))

    # plot the PCA
    sns.scatterplot(x='PC1',y='PC2',hue='Study',data=pca_df)
    plt.savefig(os.path.join(select_feats_dir,f'pca_plot_{cohort_correction_method}.png'),bbox_inches='tight')
    plt.title(cohort_correction_method)
    plt.close()


##################
# %% Look at the UMAP of the combined study
##################
if False:
    import umap
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(data_corrected.T)

    umap_df = pd.DataFrame(umap_result,columns=['UMAP1','UMAP2'],index=data_corrected.columns)
    umap_df['Study'] = metadata_df['Study']
    umap_df['Study_num'] = metadata_df['Study_num']
    umap_df['label'] = metadata_df[pretrain_label_col]

    umap_df.to_csv(os.path.join(select_feats_dir,f'umap_df_{cohort_correction_method}.csv'))

    # plot the UMAP
    sns.scatterplot(x='UMAP1',y='UMAP2',hue='Study',data=umap_df)
    plt.savefig(os.path.join(select_feats_dir,f'umap_plot_{cohort_correction_method}'),bbox_inches='tight')
    plt.title(cohort_correction_method)
    plt.close()



###################
# %% Create the task specific data
###################

task_dir = os.path.join(select_feats_dir,task_name)
os.makedirs(task_dir,exist_ok=True)

y_test_path = os.path.join(task_dir,'y_test.csv')

if True: #not os.path.exists(y_test_path):

    input_data = data_corrected.T
    for finetune_label_col in finetune_label_col_list:
        if finetune_label_col not in metadata_df.columns:
            print(f'warning: {finetune_label_col} not in metadata_df.columns')
            if finetune_label_col in origin_metadata.columns:
                common_index = list(set(metadata_df.index).intersection(set(origin_metadata.index)))
                metadata_df[finetune_label_col] = None
                metadata_df.loc[common_index,finetune_label_col] = origin_metadata.loc[common_index,finetune_label_col]
                print(f'found {finetune_label_col} in origin_metadata, adding to metadata_df on common index (N={len(common_index)}')
            else:
                print(f'warning: {finetune_label_col} also not in origin_metadata.columns')
                raise ValueError(f'finetune_label_col {finetune_label_col} not in metadata_df.columns')

    # if training_files[0] not in metadata_df.index:
    #     print('warning: training_files[0] not in metadata_df.index')
    #     training_files = [f'{origin_name}_{x}' for x in training_files]
    #     test_files = [f'{origin_name}_{x}' for x in test_files]
        # validation_files = [f'{origin_name}_{x}' for x in validation_files]

    input_pretrain_labels = metadata_df[pretrain_label_col].copy()
    # input_finetune_labels = metadata_df[finetune_label_col_list].copy()
    input_finetune_labels = origin_metadata[finetune_label_col_list].copy()

    pretrain_files = [x for x in input_data.index if x not in training_files+test_files]


    y_pretrain = input_pretrain_labels.loc[pretrain_files].copy()
    y_finetune = input_finetune_labels.loc[training_files].copy()
    y_test = input_finetune_labels.loc[test_files].copy()
    y_test_alt = input_finetune_labels.loc[test_select_file_alt].copy()

    X_pretrain = input_data.loc[pretrain_files,:].copy()
    X_finetune = input_data.loc[training_files,:].copy()
    X_test = input_data.loc[test_files,:].copy()
    X_test_alt = input_data.loc[test_select_file_alt,:].copy()

    X_finetune.to_csv(os.path.join(task_dir,'X_finetune.csv'))
    y_finetune.to_csv(os.path.join(task_dir,'y_finetune.csv'))

    X_pretrain.to_csv(os.path.join(task_dir,'X_pretrain.csv'))
    y_pretrain.to_csv(os.path.join(task_dir,'y_pretrain.csv'))

    X_test.to_csv(os.path.join(task_dir,'X_test.csv'))
    y_test.to_csv(os.path.join(task_dir,'y_test.csv'))
    if len(test_select_file_alt) > 0:
        X_test_alt.to_csv(os.path.join(task_dir,'X_test_alt.csv'))
        y_test_alt.to_csv(os.path.join(task_dir,'y_test_alt.csv'))
    
    try:
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
            # training_files_without_validation = X_train.index.tolist()
            # y_val = input_finetune_labels.loc[validation_files].copy()
            # y_train = input_finetune_labels.loc[training_files_without_validation].copy()
        else:
            X_train = X_finetune.copy()
            y_train = y_finetune.copy()
            X_val = input_data.loc[:,validation_files].copy()
            y_val = input_finetune_labels.loc[validation_files].copy()

        X_train.to_csv(os.path.join(task_dir,'X_train.csv'))
        X_val.to_csv(os.path.join(task_dir,'X_val.csv'))

        y_train.to_csv(os.path.join(task_dir,'y_train.csv'))
        y_val.to_csv(os.path.join(task_dir,'y_val.csv'))
    except ValueError:
            print('no validation and training subset created')
            pass

else:
    print('task data already exists with the data')