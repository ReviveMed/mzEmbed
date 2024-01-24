# Prepare the data for a learning task

# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

dropbox_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/'
data_dir = f'{dropbox_dir}/development_CohortCombination/data_2023_november'
input_dir= os.path.join(data_dir,'synth_norm','selected_studies 12','thresh50')




# %%




# %%
# Create custom labels
new_metadata_file = os.path.join(data_dir,'metadata_oct13_new2.csv')
dataset_labels = os.path.join(data_dir,'pretraining_dataset_labels_oct19.csv')
metadata_df = pd.read_csv(new_metadata_file,index_col=0)
labels_df = pd.read_csv(dataset_labels,index_col=0)

label_col = 'label6'
for study_id in labels_df.index:
    label_id  = labels_df.loc[study_id,label_col]
    label_id = label_id.strip(' ')
    metadata_df.loc[metadata_df['Study']==study_id,label_col] = label_id

metadata_df[label_col].value_counts()

# check that there are no null values
print(metadata_df[label_col].isnull().sum())
metadata_df[label_col].to_csv(f'{input_dir}/sample_{label_col}.csv')
pre_train_label_path = f'{input_dir}/sample_{label_col}.csv'


# %%
rcc_metadat = metadata_df[metadata_df['Study'].isin(['ST001236','ST001237'])].copy()
rcc_baseline = rcc_metadat[rcc_metadat['study_week'] == 'baseline'].copy()

print(rcc_baseline[rcc_baseline['phase']==3]['survival class'].value_counts())


temp = rcc_baseline[rcc_baseline['phase']==3]['survival class'].copy()
temp.dropna(inplace=True)
temp = temp[temp != 0.5]
temp.to_csv(f'{input_dir}/survival_class_phase3.csv')

temp = rcc_baseline[rcc_baseline['phase']==1]['survival class'].copy()
temp.dropna(inplace=True)
temp = temp[temp != 0.5]
temp.to_csv(f'{input_dir}/survival_class_phase1.csv')

# %%
print(rcc_baseline['MSKCC'].value_counts())

rcc_baseline['MSKCC_integer'] = np.nan
rcc_baseline.loc[rcc_baseline['MSKCC'] == 'INTERMEDIATE', 'MSKCC_integer'] = 1
rcc_baseline.loc[rcc_baseline['MSKCC'] == 'FAVORABLE', 'MSKCC_integer'] = 2
rcc_baseline.loc[rcc_baseline['MSKCC'] == 'POOR', 'MSKCC_integer'] = 0


temp = rcc_baseline[rcc_baseline['phase']==3]['MSKCC_integer'].copy()
temp.dropna(inplace=True)



# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(temp.index, temp, test_size=0.2, random_state=42, stratify=temp)

rcc_baseline.loc[X_train, 'MSKCC_integer'].to_csv(f'{input_dir}/MSKCC_integer_train.csv')
rcc_baseline.loc[X_test, 'MSKCC_integer'].to_csv(f'{input_dir}/MSKCC_integer_test.csv')


##########################
##########################

# %%
# Survival Class Task
task_dir = os.path.join(input_dir,'combat_survival_class_task')
os.makedirs(task_dir,exist_ok=True)
feature_path = f'{input_dir}/peak_intensity_combat.csv'
label_path = f'{input_dir}/survival_class_phase3.csv'
label_path2 = f'{input_dir}/survival_class_phase1.csv'

features = pd.read_csv(feature_path,index_col=0)
labels = pd.read_csv(label_path,index_col=0)
labels2 = pd.read_csv(label_path2,index_col=0)
pretrain_labels = pd.read_csv(pre_train_label_path,index_col=0)

# %%
X = features[labels.index].T
X_test = features[labels2.index].T
y_test = labels2['survival class']
y = labels['survival class']

# train test split to get the validation set
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=y)

hold_out_files = labels.index.tolist() + labels2.index.tolist()
pretrain_files = [col for col in features.columns if col not in hold_out_files]

X_pretrain = features[pretrain_files].T
y_pretrain = pretrain_labels[label_col]

X_train.to_csv(os.path.join(task_dir,'X_train.csv'))
X_test.to_csv(os.path.join(task_dir,'X_test.csv'))
X_val.to_csv(os.path.join(task_dir,'X_val.csv'))
X_pretrain.to_csv(os.path.join(task_dir,'X_pretrain.csv'))
y_train.to_csv(os.path.join(task_dir,'y_train.csv'))
y_test.to_csv(os.path.join(task_dir,'y_test.csv'))
y_val.to_csv(os.path.join(task_dir,'y_val.csv'))
y_pretrain.to_csv(os.path.join(task_dir,'y_pretrain.csv'))
# %%
##########################
##########################

# MSKCC Risk Group Class Task
task_dir = os.path.join(input_dir,'combat_mskcc_regression_task')
os.makedirs(task_dir,exist_ok=True)
feature_path = f'{input_dir}/peak_intensity_combat.csv'
label_path = f'{input_dir}/MSKCC_integer_train.csv'
label_path2 = f'{input_dir}/MSKCC_integer_test.csv'
features = pd.read_csv(feature_path,index_col=0)
labels = pd.read_csv(label_path,index_col=0)
labels2 = pd.read_csv(label_path2,index_col=0)
pretrain_labels = pd.read_csv(pre_train_label_path,index_col=0)


X = features[labels.index].T
X_test = features[labels2.index].T
y_test = labels2['MSKCC_integer']
y = labels['MSKCC_integer']

# train test split to get the validation set
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=y)

hold_out_files = labels.index.tolist() + labels2.index.tolist()
pretrain_files = [col for col in features.columns if col not in hold_out_files]

X_pretrain = features[pretrain_files].T
y_pretrain = pretrain_labels[label_col]

X_train.to_csv(os.path.join(task_dir,'X_train.csv'))
X_test.to_csv(os.path.join(task_dir,'X_test.csv'))
X_val.to_csv(os.path.join(task_dir,'X_val.csv'))
X_pretrain.to_csv(os.path.join(task_dir,'X_pretrain.csv'))
y_train.to_csv(os.path.join(task_dir,'y_train.csv'))
y_test.to_csv(os.path.join(task_dir,'y_test.csv'))
y_val.to_csv(os.path.join(task_dir,'y_val.csv'))
y_pretrain.to_csv(os.path.join(task_dir,'y_pretrain.csv'))
# %%
# %%
# Surival Regression Task
##########################
##########################

task_dir = os.path.join(input_dir,'combat_survival_regression_task')
os.makedirs(task_dir,exist_ok=True)
feature_path = f'{input_dir}/peak_intensity_combat.csv'
features = pd.read_csv(feature_path,index_col=0)
# %%
os_val_col = 'OS (months)'
os_cens_col = 'OS censor'
os_event_col = 'Death Observed'
rcc_baseline = metadata_df[metadata_df['study_week'] == 'baseline'].copy()
rcc3_baseline = rcc_baseline[rcc_baseline['Study']=='ST001237']
rcc1_baseline = rcc_baseline[rcc_baseline['Study']=='ST001236']

X = features[rcc3_baseline.index].T
X_test = features[rcc1_baseline.index].T

if os_cens_col in rcc_baseline.columns:
    y = rcc3_baseline[[os_val_col,os_cens_col]]
    y_test = rcc1_baseline[[os_val_col,os_cens_col]]
elif os_event_col in rcc_baseline.columns:
    y = rcc3_baseline[[os_val_col,os_event_col]]
    y_test = rcc1_baseline[[os_val_col,os_event_col]]
else:
    raise ValueError('OS censor or OS event column not found')


# %%
# train test split to get the validation set
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  )


# X_pretrain = features[pretrain_files].T
# y_pretrain = pretrain_labels[label_col]

X_train.to_csv(os.path.join(task_dir,'X_train.csv'))
X_test.to_csv(os.path.join(task_dir,'X_test.csv'))
X_val.to_csv(os.path.join(task_dir,'X_val.csv'))
# X_pretrain.to_csv(os.path.join(task_dir,'X_pretrain.csv'))
y_train.to_csv(os.path.join(task_dir,'y_train.csv'))
y_test.to_csv(os.path.join(task_dir,'y_test.csv'))
y_val.to_csv(os.path.join(task_dir,'y_val.csv'))
# y_pretrain.to_csv(os.path.join(task_dir,'y_pretrain.csv'))
# %%

# %%
##########################
##########################

# MSKCC Risk Group Binary Task
task_dir = os.path.join(input_dir,'combat_mskcc_binary_task')
os.makedirs(task_dir,exist_ok=True)
feature_path = f'{input_dir}/peak_intensity_combat.csv'
label_path = f'{input_dir}/MSKCC_integer_train.csv'
label_path2 = f'{input_dir}/MSKCC_integer_test.csv'
features = pd.read_csv(feature_path,index_col=0)
labels = pd.read_csv(label_path,index_col=0)
labels2 = pd.read_csv(label_path2,index_col=0)
pretrain_labels = pd.read_csv(pre_train_label_path,index_col=0)



y_test = labels2.loc[labels2['MSKCC_integer'].isin([0,2]),'MSKCC_integer']
y = labels.loc[labels['MSKCC_integer'].isin([0,2]),'MSKCC_integer']
y[y==2] = 1
y_test[y_test==2] = 1
y.rename('MSKCC_binary',inplace=True)
y_test.rename('MSKCC_binary',inplace=True)
X = features[y.index].T
X_test = features[y_test.index].T

# train test split to get the validation set
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=y)

hold_out_files = y.index.tolist() + y_test.index.tolist()
pretrain_files = [col for col in features.columns if col not in hold_out_files]

X_pretrain = features[pretrain_files].T
y_pretrain = pretrain_labels[label_col]

X_train.to_csv(os.path.join(task_dir,'X_train.csv'))
X_test.to_csv(os.path.join(task_dir,'X_test.csv'))
X_val.to_csv(os.path.join(task_dir,'X_val.csv'))
X_pretrain.to_csv(os.path.join(task_dir,'X_pretrain.csv'))
y_train.to_csv(os.path.join(task_dir,'y_train.csv'))
y_test.to_csv(os.path.join(task_dir,'y_test.csv'))
y_val.to_csv(os.path.join(task_dir,'y_val.csv'))
y_pretrain.to_csv(os.path.join(task_dir,'y_pretrain.csv'))
# %%
