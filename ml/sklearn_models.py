import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from sklearn.model_selection import RandomizedSearchCV,  StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from torchmetrics import Accuracy, AUROC
# from sklearn.externals import joblib
from prep import ClassifierDataset

logistic_regression_param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.005, 0.1, 0.5, 1, 2, 5, 10],
            'solver' : ['liblinear'],
            'class_weight': ['balanced']
            }

random_forest_param_grid = {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 8],
                'bootstrap': [True, False],
                'class_weight': ['balanced']
}

decision_tree_param_grid = {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 8],
                'class_weight': ['balanced']
}

svc_param_grid = {
                'C': [0.005, 0.1, 0.5, 1,2, 5, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                # 'early_stopping': [True],
                'class_weight': ['balanced'],
                'probability': [True]
                # 'validation_fraction': [0.1, 0.2]
            }



def run_train_sklearn_model(data_dict,save_dir,**kwargs):

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
                                n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
    
    
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