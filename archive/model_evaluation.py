# Helper Functions for Model Evaluation


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import get_scorer
import pandas as pd
import numpy as np



def get_model_predictions(model,X,model_type='sklearn_proba'):
    if model_type=='sklearn_proba':
        predictions = model.predict_proba(X)
        predictions = predictions[:,1]
    if model_type=='keras':
        predictions = model.predict(X)
        predictions = predictions[:,0]
    else:
        raise ValueError({model_type} + ' is not a valid model type')
    return predictions


def evaluate_model(model, X_train, y_train, X_test=None, y_test=None, 
                   cv_folds=5, cv_repeats=3, metric_list=['auc'], 
                   stratified=False, random_state=None, 
                   model_type='sklearn_proba',custom_fit_func=None):
    
    # Initialize the scorer
    # scorer = get_scorer(metric)
    
    # Initialize a list to store the results for each fold and repeat
    results = []

    # Repeat the cross-validation
    for repeat in range(cv_repeats):

        # Initialize the cross-validator
        if stratified:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state+repeat)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state+repeat)

        
        # Perform cross-validation
        for fold, (train_index, val_index) in enumerate(cv.split(X_train, y_train)):
            # Split the data
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            # Fit the model
            if custom_fit_func is not None:
                # allows for saving the model history
                model = custom_fit_func(model,X_train_fold,y_train_fold,X_val_fold,y_val_fold)
            else:
                model.fit(X_train_fold, y_train_fold)
            
            # Compute the scores for each subset
            for subset in ['train','val','test']:
                if subset=='train':
                    X_subset = X_train_fold
                    y_subset = y_train_fold
                elif subset=='val':
                    X_subset = X_val_fold
                    y_subset = y_val_fold
                elif subset=='test':
                    if X_test is not None and y_test is not None:
                        X_subset = X_test
                        y_subset = y_test
                    else:
                        continue
                else:
                    raise ValueError(f'{subset} is not a valid subset')

                res = { 'Repeat': repeat,
                        'Fold': fold,
                        'Subset': subset}
                
                pred = get_model_predictions(model,X_subset,model_type=model_type)
                for metric in metric_list:
                    scorer = get_scorer(metric)
                    score_val = scorer(y_subset, pred)
                    
                    res = { 'Repeat': repeat,
                        'Fold': fold,
                        'Subset': subset,
                        'Metric': metric,
                        'Value': score_val}
                
                    results.append(res)
            
    
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df