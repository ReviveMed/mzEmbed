import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from sklearn.model_selection import RandomizedSearchCV,  StratifiedKFold, RepeatedStratifiedKFold
from utils.misc import save_json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import make_scorer
# from sklearn.externals import joblib

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


def get_base_model(model_kind):
    if model_kind == 'logistic_regression':
        base_model = LogisticRegression()
    elif model_kind == 'random_forest':
        base_model = RandomForestClassifier()
    elif model_kind == 'decision_tree':
        base_model = DecisionTreeClassifier()
    elif model_kind == 'svc':
        base_model = SVC(probability=True)
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
    else:
        raise ValueError('The model name is not recognized.')
    return param_grid


######### Main functions #########


def run_train_sklearn_model(data_dict, save_dir, **kwargs):
    """
    Trains a sklearn model using the provided data and saves the model and results.

    Args:
        data_dict (dict): A dictionary containing the training data (and optionally validation and test data)
            for the model. The keys are the phase names and the values are the data objects.
            Currently the data objects are assumed to be torch tensors.
        save_dir (str): The directory path where the model and results will be saved.
        **kwargs: Additional keyword arguments for configuring the model training.
            - model_name (str, optional): The name of the model. If not provided, the default name will be used.
            - model_kind (str, optional): The kind of model to train. Default is 'logistic_regression'.
            - param_grid (dict, optional): The parameter grid for hyperparameter tuning. Default is None.
            - base_model (object, optional): The base model object. If not provided, a default model will be used based on the model_kind.
            - phase_list (list, optional): The list of phases to train the model on. If not provided, all phases in the data_dict will be used.
            - num_classes (int, optional): The number of classes in the target variable. If not provided, it will be inferred from the training data.
            - feat_subset (object, optional): The feature subset selection object. Default is None.

    Returns:
        dict: A dictionary containing the model information and evaluation results.

    Raises:
        ValueError: If the model name is not recognized.
        NotImplementedError: If feature subset selection is not yet implemented.
    """

    # check that the input directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_name = kwargs.get('model_name',None)
    model_kind = kwargs.get('model_kind','logistic_regression')
    param_grid = kwargs.get('param_grid',None)
    base_model = kwargs.get('base_model',None)
    phase_list = kwargs.get('phase_list',None)
    num_classes = kwargs.get('num_classes',None)
    feat_subset = kwargs.get('feat_subset',None)
    n_iter = kwargs.get('n_iter',100)
    cv = kwargs.get('cv',5)
    verbose = kwargs.get('verbose',0)
    random_state = kwargs.get('random_state',42)
    scoring = kwargs.get('scoring','roc_auc')

    if base_model is None:
        if model_kind == 'logistic_regression':
            base_model = LogisticRegression(max_iter=1000)
            if param_grid is None:
                param_grid = logistic_regression_param_grid
        elif model_kind == 'random_forest':
            base_model = RandomForestClassifier()
            if param_grid is None:
                param_grid = random_forest_param_grid
        elif model_kind == 'decision_tree':
            base_model = DecisionTreeClassifier()
            if param_grid is None:
                param_grid = decision_tree_param_grid
        elif model_kind == 'svc':
            base_model = SVC(probability=True)
            if param_grid is None:
                param_grid = svc_param_grid
        else:
            raise ValueError('The model name is not recognized.')

    if model_name is None:
        model_name = model_kind +'_RS'

    if phase_list is None:
        phase_list = list(data_dict.keys())

    # load the data
    X_train = data_dict['train'].X.numpy()
    y_train = data_dict['train'].y.numpy()

    if feat_subset is not None:
        raise NotImplementedError('Feature subset selection is not yet implemented.')

    if num_classes is None:
        num_classes = len(np.unique(y_train))

    if num_classes > 2:
        task = 'multi'
    else:
        task = 'binary'

    output_file_path = os.path.join(save_dir, f'{model_name}_output.json')


    # create the model
    if (param_grid is not None) and (len(param_grid) > 0):
        model = RandomizedSearchCV(base_model, param_distributions=param_grid, 
                                n_iter=n_iter, cv=cv, verbose=verbose, random_state=random_state, n_jobs=-1, scoring=scoring)
    
    
        model.fit(X_train, y_train)
        # save the model results
        model_results = pd.DataFrame(model.cv_results_)
        model_results.to_csv(os.path.join(save_dir, f'{model_kind}_randsearchcv_results.csv'))
        best_model = model.best_estimator_
    else:
        best_model = base_model

        best_model.fit(X_train, y_train)

    phase_sizes = {phase: len(data_dict[phase]) for phase in phase_list}
    end_state_auroc = {}
    end_state_acc = {}
    end_state_auroc2 = {}
    end_state_acc2 = {}
    metrics_accu = Accuracy(task=task)
    metrics_auroc = AUROC(task=task,average='weighted')  

    for phase in phase_list:
        metrics_accu.reset()
        metrics_auroc.reset()

        X = data_dict[phase].X.numpy()
        y = data_dict[phase].y.numpy()

        y_pred = best_model.predict(X)
        y_prob = best_model.predict_proba(X)[:,1]

        # currently using sklearn, but maybe I should use torch metrics to be consistent?
        end_state_auroc2[phase] = roc_auc_score(y, y_prob, average='weighted')
        end_state_acc2[phase] = accuracy_score(y, y_pred)

        metrics_accu(torch.tensor(y_pred),torch.tensor(y))
        metrics_auroc(torch.tensor(y_prob),torch.tensor(y))

        end_state_auroc[phase] = metrics_auroc.compute().item()
        end_state_acc[phase] = metrics_accu.compute().item()
    
    # save the model results
    output_data = {
        'model_name': model_name,
        'model_kind': model_kind,
        'model_hyperparameters': best_model.get_params(),
        'phase_sizes': phase_sizes,
        'end_state_auroc': end_state_auroc,
        'end_state_acc': end_state_acc,
        'end_state_auroc2': end_state_auroc2,
        'end_state_acc2': end_state_acc2,
        'param_grid': param_grid,
        'feat_subset': feat_subset,
    
    }
    
    # save the output data to json
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f)



    # save the model
    # best_model_path = os.path.join(save_dir, f'{model_kind}_model.pkl')
    # joblib.dump(best_model, best_model_path)
    
    return output_data




def sklearn_fitCV_eval_wrapper(data_dict, model_kind, 
                      param_grid=None, cv=None, n_iter=10,
                      model_name=None, output_dir=None, 
                      scoring_func=None, verbose=0):
    """
    Fits a machine learning model using cross-validation and evaluates its performance.

    Parameters:
    - data_dict (dict): A dictionary containing the training, validation, and test data.
    - model_kind (str): The kind of model to use.
    - param_grid (dict): A dictionary of hyperparameters to search over.
    - cv (object, optional): The cross-validation strategy to use. If not provided, RepeatedStratifiedKFold with 5 splits and 20 repeats will be used.
    - n_iter (int, optional): The number of iterations for randomized search. Default is 10.
    - model_name (str, optional): The name of the model. If not provided, it will be set to the model_kind.
    - output_dir (str, optional): The directory to save the model summary JSON file. If not provided, the summary will not be saved.
    - scoring_func (object, optional): The scoring function to use for evaluation. If not provided, roc_auc_score with weighted average and needs_proba=True will be used.
    - verbose (int, optional): Verbosity level. Default is 0.

    Returns:
    - model_summary (dict): A dictionary containing the summary of the model's performance.

    """
         
    if cv is None:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    
    if scoring_func is None:
        # depends on which sklearn version we are using
        # scoring_func = make_scorer(roc_auc_score, average='weighted', response_method='predict_proba')
        scoring_func = make_scorer(roc_auc_score, average='weighted', needs_proba=True)

    if model_name is None:
        model_name = model_kind

    if param_grid is None:
        param_grid = get_default_param_grid(model_kind)

    base_model = get_base_model(model_kind)

    model = RandomizedSearchCV(base_model, 
                               param_distributions=param_grid, 
                               n_iter=n_iter, 
                               cv=cv, 
                               scoring=scoring_func, 
                               verbose=verbose, 
                               random_state=1010,
                               n_jobs=1)
    X_train, y_train = data_dict['X_train'], data_dict['y_train']

    model.fit(X_train, y_train)

    model_results = pd.DataFrame(model.cv_results_)

    model_results.sort_values('rank_test_score', inplace=True)
    model_results.iloc[0]['mean_test_score']
    cv_score = model_results.iloc[0]['mean_test_score']
    cv_score_std = model_results.iloc[0]['std_test_score']
    print(f'Train CV score: {cv_score} +/- {cv_score_std}')

    y_train_proba = model.predict_proba(X_train)[:,1]
    train_score = roc_auc_score(y_train, y_train_proba, average='weighted')
    print(f'Training score: {train_score}')

    if 'X_val' in data_dict:
        X_val, y_val = data_dict['X_val'], data_dict['y_val']
        y_val_proba = model.predict_proba(X_val)[:,1]
        val_score = roc_auc_score(y_val, y_val_proba, average='weighted')
        print(f'Validation score: {val_score}')
    else:
        val_score = None

    if 'X_test' in data_dict:
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        y_test_proba = model.predict_proba(X_test)[:,1]
        test_score = roc_auc_score(y_test, y_test_proba, average='weighted')
        print(f'Test score: {test_score}')
    else:
        test_score = None

    if ('X_val' in data_dict) and ('X_test' in data_dict):
        X_train_val = np.concatenate([X_train, X_val], axis=0)
        y_train_val = np.concatenate([y_train, y_val], axis=0)

        # refit the best model
        base_model.set_params(**model.best_params_)
        base_model.fit(X_train_val, y_train_val)
        y_test_proba = base_model.predict_proba(X_test)[:,1]
        test_score_2 = roc_auc_score(y_test, y_test_proba, average='weighted')
        print(f'Test score (refit on Train and Val): {test_score_2}')
    else:
        test_score_2 = None


    model_summary = {
        'model_kind': model_kind,
        'model_name': model_name,
        'n_input ft': X_train.shape[1],
        'param_grid' : param_grid,
        'best_params': model.best_params_,
        'score name': 'roc_auc (weighted)',
        'cv_score': cv_score,
        'cv_score_std': cv_score_std,
        'train_score': train_score,
        'val_score': val_score,
        'test_score': test_score,
        'test_score (fit on train+val)': test_score_2,
        'train sz': X_train.shape[0],
        'val sz': X_val.shape[0],
        'test sz': X_test.shape[0],
        'n_trials': n_iter,
        'n_folds': cv.get_n_splits(),
    }

    if output_dir is not None:
        save_json(model_summary, os.path.join(output_dir, f'{model_name}_summary.json'))

    return model_summary



def sklearn_fit_eval_wrapper(data_dict, model_kind, 
                      param_grid=None, n_iter=10,
                      model_name=None, output_dir=None, 
                      scoring_func=None, verbose=0):
    """
    Fits and evaluates a sklearn model using different parameter combinations.

    Parameters:
    - data_dict (dict): A dictionary containing the training, validation, and test data.
                        It should have the following keys: 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'.
    - model_kind (str): The kind of sklearn model to use.
    - param_grid (dict): A dictionary containing the hyperparameter grid for the model.
    - n_iter (int): The number of iterations to perform.
    - model_name (str): The name of the model. If None, the model_kind will be used as the name.
    - output_dir (str): The directory to save the results. If None, the results will not be saved.
    - scoring_func (callable): The scoring function to use for evaluation. If None, roc_auc_score with average='weighted' and needs_proba=True will be used.
    - verbose (int): The level of verbosity. Set to 0 for no output.

    Returns:
    - res_summary (pd.DataFrame): A DataFrame containing the evaluation results for each parameter combination.

    """


    if scoring_func is None:
        # depends on which sklearn version we are using
        # scoring_func = make_scorer(roc_auc_score, average='weighted', response_method='predict_proba')
        scoring_func = make_scorer(roc_auc_score, average='weighted', needs_proba=True)
    
    if model_name is None:
        model_name = model_kind

    if param_grid is None:
        param_grid = get_default_param_grid(model_kind)
    model = get_base_model(model_kind)
    all_res = []
    
    # cycle over all parameter combinations
    param_combinations = []
    for iter in range(4*n_iter):
        param_kwargs = {}
        for param_name in param_grid.keys():
            param_kwargs[param_name] = np.random.choice(param_grid[param_name])
            if isinstance(param_kwargs[param_name], float):
                #if it should be an integer, make it an integer
                if param_kwargs[param_name] == int(param_kwargs[param_name]):
                    param_kwargs[param_name] = int(param_kwargs[param_name])

        param_combinations.append(param_kwargs)

    #remove duplicates
    param_combinations = [dict(t) for t in {tuple(d.items()) for d in param_combinations}]
    print(len(param_combinations))
    if len(param_combinations) > n_iter:
        param_combinations = param_combinations[:n_iter]
    if len(param_combinations) < n_iter:
        print('warning, running on fewer iterations {} than requested {}'.format(len(param_combinations), n_iter))

    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']


    for model_param in param_combinations:

        param = {k:v for k,v in model_param.items()}
        model.set_params(**model_param)
        model.fit(X_train, y_train)
        train_score = scoring_func(model, X_train, y_train)
        # y_train_proba = model.predict_proba(X_train)[:,1]
        # train_score = roc_auc_score(y_train, y_train_proba, average='weighted')
        param['train_score (fit on train)'] = train_score


        # y_val_proba = model.predict_proba(X_val)[:,1]
        # val_score = roc_auc_score(y_val, y_val_proba, average='weighted')
        val_score = scoring_func(model, X_val, y_val)
        param['val_score (fit on train)'] = val_score

        # y_test_proba = model.predict_proba(X_test)[:,1]
        # test_score = roc_auc_score(y_test, y_test_proba, average='weighted')
        test_score = scoring_func(model, X_test, y_test)
        param['test_score (fit on train)'] = test_score

        # join the train and val sets
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)
        model.fit(X_train_val, y_train_val)
        # y_test_proba = model.predict_proba(X_test)[:,1]
        # test_score2 = roc_auc_score(y_test, y_test_proba, average='weighted')
        test_score2 = scoring_func(model, X_test, y_test)
        param['test_score (fit on train+val)'] = test_score2

        all_res.append(param)


    res_summary = pd.DataFrame(all_res)
    res_summary['train sz'] = X_train.shape[0]
    res_summary['val sz'] = X_val.shape[0]
    res_summary['test sz'] = X_test.shape[0]
    res_summary['n_trials'] = n_iter
    res_summary['n_peaks'] = X_train.shape[1]
    res_summary.sort_values('train_score (fit on train)', ascending=False, inplace=True)
    res_summary.sort_values('val_score (fit on train)', ascending=False, inplace=True)

    if output_dir is not None:
        res_summary.to_csv(os.path.join(output_dir, f'{model_name}_summary.csv'))


    return res_summary