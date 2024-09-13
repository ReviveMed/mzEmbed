import pandas as pd
import numpy as np

import os
import copy
import tqdm
import itertools
import json
from misc import save_json

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
# import XGBoost
from xgboost import XGBClassifier



import optuna
import neptune

from neptune.utils import stringify_unsupported

from utils_neptune import get_latest_dataset
import argparse

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

PROJECT_ID = 'revivemed/Survival-RCC'

storage_name = 'optuna'
USE_WEBAPP_DB = True
USE_NEPTUNE = True
WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'

######### Helpers and defaults #########

logistic_regression_param_grid = {
            # 'penalty': ['l1', 'l2'],
            # 'solver' : ['liblinear'],
            'penalty': ['elasticnet'],
            'solver' : ['saga'],
            'l1_ratio': [0, 0.25, 0.33, 0.5, 0.66, 0.75, 1],
            'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            'class_weight': ['balanced'],
            'max_iter': [10000],
            'tol': [1e-3, 1e-4],
            }

random_forest_param_grid = {
                'n_estimators': [10, 20, 50, 100, 200, 400],
                'max_depth': [2, 3, 5, 10, None],
                'min_samples_split': [0.05, 0.1, 0.2, 2,5,7,10],
                'min_samples_leaf': [0.025, 0.05, 0.1, 2, 4, 6, 8],
                'bootstrap': [True, False],
                # 'max_features': ['auto', 'sqrt', 'log2'], #auto not in current version
                'max_features': ['sqrt', 'log2'],
                'ccp_alpha': [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                'class_weight': ['balanced']
}

decision_tree_param_grid = {
                'max_depth': [2, 3, 5, 10, None],
                'min_samples_split': [0.05, 0.1, 0.2, 2, 5, 7, 10],
                'min_samples_leaf': [0.025, 0.05, 0.1, 2, 4, 6, 8],
                'class_weight': ['balanced']
}

svc_param_grid = {
                'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.66, 1, 1.5, 2, 5, 10],
                'kernel': ['linear', 'rbf', 'poly'], 
                'gamma': ['scale', 'auto'],
                # 'early_stopping': [True],
                'class_weight': ['balanced'],
                'probability': [True]
                # 'validation_fraction': [0.1, 0.2]
            }

xgboost_param_grid = {
                'n_estimators': [10, 20, 50, 100, 200, 400],
                'max_depth': [2, 3, 5, 10],
                'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                'subsample': [0.5, 0.7, 0.9, 1],
                'colsample_bytree': [0.5, 0.7, 0.9, 1],
                'colsample_bylevel': [0.5, 0.7, 0.9, 1],
                'colsample_bynode': [0.5, 0.7, 0.9, 1],
                'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                'scale_pos_weight': [0.5, 1, 2, 5, 10],
                'base_score': [0.5],
                'eval_metric': ['logloss'],
                'use_label_encoder': [False]
            }

def get_base_model(model_kind):
    if model_kind == 'logistic_regression':
        base_model = LogisticRegression()
    elif model_kind == 'random_forest':
        base_model = RandomForestClassifier()
    elif model_kind == 'decision_tree':
        base_model = DecisionTreeClassifier()
    elif model_kind == 'svc':
        base_model = SVC(probability=True)
    elif model_kind == 'xgboost':
        base_model = XGBClassifier()
    elif model_kind == 'logistic_regression_multiclass':
        # base_model = LogisticRegression(multi_class='multinomial')
        base_model = LogisticRegression(multi_class='ovr')
    else:
        raise ValueError('The model name is not recognized.')
    return base_model

def get_default_param_grid(model_kind):
    if model_kind == 'logistic_regression':
        param_grid = logistic_regression_param_grid
    elif model_kind == 'random_forest':
        param_grid = random_forest_param_grid
    elif model_kind == 'decision_tree':
        param_grid = decision_tree_param_grid
    elif model_kind == 'svc':
        param_grid = svc_param_grid
    elif model_kind == 'xgboost':
        param_grid = xgboost_param_grid
    else:
        raise ValueError('The model name is not recognized.')
    return param_grid


def create_data_dict(data_dir, set_name, target_col, data_dict=None):
    if data_dict is None:
        data_dict = {}
    X_path = os.path.join(data_dir, f'X_finetune_{set_name}.csv')
    y_path = os.path.join(data_dir, f'y_finetune_{set_name}.csv')
    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    not_nan_idx = ~y[target_col].isna()
    data_dict[set_name] = {
        'X': X.loc[not_nan_idx],
        'y': y.loc[not_nan_idx,target_col],
        'target_col': target_col,
        'X_file': X_path,
        'y_file': y_path
    }
    return data_dict



def fit_model(model_name, params, data_dict, set_name='train'):
    
    model = get_base_model(model_name)
    model.set_params(**params)
    X = data_dict[set_name]['X']
    y = data_dict[set_name]['y']
    y_pred = model.predict(X)
    unique_labels = np.unique(y)
    num_labels = len(unique_labels)
    if num_labels > 2:
        if model_name == 'logistic_regression':
            model.set_params(multi_class='ovr')
        elif model_name == 'svc':
            model.set_params(decision_function_shape='ovr')
        elif model_name == 'xgboost':
            model.set_params(objective='multi:softmax', num_class=num_labels)
    model.fit(X, y)
    return model

def evaluate_model(model, data_dict, set_name='test'):
    X = data_dict[set_name]['X']
    y = data_dict[set_name]['y']
    y_pred = model.predict(X)
    unique_labels = np.unique(y)
    num_labels = len(unique_labels)
    # print(f"Number of labels: {num_labels}")
    if num_labels > 2:
        y_prob = model.predict_proba(X)
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob, multi_class='ovr')
    else:
        y_prob = model.predict_proba(X)[:,1]
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
    return acc, auc

def run_model(model_name, params, data_dict, train_name='train', 
              eval_name='val', n_repeat=1,run_dict=None):
    
    if run_dict:
        run_dict['model_name'] = model_name
        run_dict['params'] = stringify_unsupported(params)
        run_dict['target_col'] = data_dict[train_name]['target_col']
        run_dict['num_classes'] = len(data_dict[train_name]['y'].unique())
        for set_name in data_dict.keys():
            if 'target_col':
                target_col = data_dict[set_name]['target_col']
                run_dict[f'dataset/{set_name}/target_col'] = target_col
            if 'X_file' in data_dict[set_name].keys():
                run_dict[f'dataset/{set_name}/X'].track_files(data_dict[set_name]['X_file'])
            if 'y_file' in data_dict[set_name].keys():
                run_dict[f'dataset/{set_name}/y'].track_files(data_dict[set_name]['y_file'])

    
    val_auc_list = []
    train_auc_list = []
    for _ in range(n_repeat):
        model = fit_model(model_name, params, data_dict, train_name)
        
        ## Classifier Specific Code
        train_acc, train_auc = evaluate_model(model, data_dict, train_name)
        val_acc, val_auc = evaluate_model(model, data_dict, eval_name)
        if run_dict:
            run_dict['metrics/train/accuracy'].append(train_acc)
            run_dict['metrics/train/auc'].append(train_auc)
            run_dict['metrics/val/accuracy'].append(val_acc)
            run_dict['metrics/val/auc'].append(val_auc)

        val_auc_list.append(val_auc)
        train_auc_list.append(train_auc)

    val_auc = np.mean(val_auc_list)
    train_auc = np.mean(train_auc_list)
    return val_auc, train_auc

def base_objective(trial, model_name, data_dict, 
                   train_name='train', eval_name='val', 
                   run_dict=None, n_repeat=1, seed=0):
    
    param_grid = get_default_param_grid(model_name)
    params = {}
    for key, values in param_grid.items(): 
        # check if all of the values are integers
        if all(isinstance(x, int) for x in values):
            min_val = min(values)
            max_val = max(values)
            #check if log scale
            if (min_val > 0) and (max_val/min_val > 100):
                params[key] = trial.suggest_int(key, min(values), max(values), log=True)
            else:
                params[key] = trial.suggest_int(key, min(values), max(values))
        elif all(isinstance(x, float) for x in values):
            if (min_val > 0) and (max_val/min_val > 100):
                params[key] = trial.suggest_float(key, min(values), max(values), log=True)
            else:
                params[key] = trial.suggest_float(key, min(values), max(values))
        else:
            params[key] = trial.suggest_categorical(key, values)
    
    val_auc, train_auc = run_model(model_name, 
                                   params = params, 
                                    data_dict = data_dict,
                                    train_name = train_name,
                                    eval_name = eval_name,
                                    n_repeat = n_repeat,
                                    run_dict = run_dict)


    trial.set_user_attr('val_auc', val_auc)
    trial.set_user_attr('train_auc', train_auc)

    if val_auc > train_auc:
        val_auc = 0.9*train_auc
    return val_auc



if __name__ == '__main__':

    # get user input
    parser = argparse.ArgumentParser(description='Run traditional classifiers')
    # parser.add_argument('--model_name', type=str, default='logistic_regression', help='The name of the model to run')
    parser.add_argument('--model_name', type=str, default='xgboost', help='The name of the model to run')
    parser.add_argument('--n_trials', type=int, default=10, help='The number of trials to run')
    parser.add_argument('--n_repeat', type=int, default=1, help='The number of times to repeat each trial')
    parser.add_argument('--y_col', type=str, default='MSKCC BINARY', help='The name of the target column')
    parser.add_argument('--direction', type=str, default='maximize', help='The direction of the optimization')
    parser.add_argument('--study_name', type=str, default=None, help='The name of the study')

    args = parser.parse_args()
    model_name = args.model_name
    n_trials = args.n_trials
    n_repeat = args.n_repeat
    y_col = args.y_col
    direction = args.direction
    study_name = args.study_name

    # %% get the home directory
    home_dir = os.path.expanduser("~")
    data_dir = f'{home_dir}/DATA3'
    output_dir = f'{home_dir}/OUTPUT'
    if study_name is None:
        study_name = f'{model_name}_{y_col}'
    if USE_WEBAPP_DB:
        print('using webapp database')
        storage_name = WEBAPP_DB_LOC

    # %%
    ## Get the latest dataset
    print('Getting the latest dataset')
    data_dir = get_latest_dataset(data_dir=data_dir,project=PROJECT_ID)
    if os.path.exists(data_dir+'/hash.txt'):
        with open(data_dir+'/hash.txt','r') as f:
            dataset_hash = f.read()
    else:
        dataset_hash = None

    # %%
    ## Load the data
    print('Loading the data')
    data_dict = {}
    if n_trials>0:
        data_dict = create_data_dict(data_dir, 'train', y_col, data_dict)
        data_dict = create_data_dict(data_dir, 'val', y_col, data_dict)


    def objective(trial):
        if USE_NEPTUNE:
            run = neptune.init_run(project=PROJECT_ID, api_token=NEPTUNE_API_TOKEN)
            run['optuna_trial_number'] = trial.number
            run_dict = run['training_run']
            if dataset_hash:
                run['dataset/hash'] = dataset_hash
            trial.set_user_attr('neptune_id', run["sys/id"].fetch())
        else:
            run_dict = None

        obj_result = base_objective(trial, model_name, data_dict, 
                              train_name='train', 
                              eval_name='val', 
                              n_repeat=n_repeat,
                              run_dict=run_dict)

        if USE_NEPTUNE:
            run.stop()

        return obj_result

    # %%
    ## Create Optuna Study and run the optimization

    study = optuna.create_study(direction=direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)

    if n_trials>0:
        study.optimize(objective, n_trials=n_trials)

    else:
        print('No trials to run, train and evaluate on the top model')
        data_dict = create_data_dict(data_dir, 'trainval', y_col, data_dict)
        data_dict = create_data_dict(data_dir, 'test', y_col, data_dict)

        best_params = study.best_params


        if USE_NEPTUNE:
            run_id = study.best_trial.user_attrs['neptune_id']
            run = neptune.init_run(project=PROJECT_ID, api_token=NEPTUNE_API_TOKEN,
                               with_id=run_id)
            run_dict = run['testing_run']
        else:
            run_dict = None

        val_auc, train_auc = run_model(model_name, 
                                       params = best_params, 
                                        data_dict = data_dict,
                                        train_name = 'trainval',
                                        eval_name = 'test',
                                        n_repeat = n_repeat,
                                        run_dict = run_dict)

        if USE_NEPTUNE:
            run.stop()

        print(f'Test AUC: {val_auc}')

    # %%
