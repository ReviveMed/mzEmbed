#Util functions for runnign the Logistic Regression model


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score




def log_reg_multi_class(x_train, y_train, x_val, y_val):

    # Define the parameter grid
    param_grid = {
        'C': [1, 10, 100], #[0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'  ],  # 'l1' only works with solvers like 'liblinear' or 'saga'
        'solver': ['lbfgs']  ,  # solvers that support 'l1' penalty
        #'class_weight': [None, 'balanced']
    }


    best_val_accuracy = 0
    best_model = None

    # Manual grid search over hyperparameters
    for C in param_grid['C']:
        for penalty in param_grid['penalty']:
            for solver in param_grid['solver']:
                if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                    continue  # Skip incompatible combinations
                
                # Initialize and train the model
                clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)
                #clf = LogisticRegression(C=C, penalty=penalty, solver=solver, #multi_class='multinomial', max_iter=1000, random_state=42)
                clf.fit(x_train, y_train)
                
                # Evaluate on the validation set
                y_val_pred = clf.predict(x_val)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                
                # Check if this is the best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = clf
                    #print(f'New best model found: C={C}, penalty={penalty}, solver={solver}, Validation Accuracy: {val_accuracy:.4f}')

    # Final evaluation on the test set using the best model
    #y_test_pred = best_model.predict(x_test)
    #test_accuracy = accuracy_score(y_test, y_test_pred)
    #print(f'Val Accuracy with best model: {best_val_accuracy:.4f}')

    return best_model, best_val_accuracy




import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np

def cox_proportional_hazards(X_train, Y_train_OS, Y_train_event, X_val, Y_val_OS, Y_val_event, X_test, Y_test_OS, Y_test_event):
    """
    Train a Cox Proportional Hazards model to predict survival time,
    evaluate performance on the validation set, and calculate the C-index.

    Parameters:
    X_train: Training feature vectors
    Y_train: Training target values
    X_val: Validation feature vectors
    Y_val: Validation target values

    Returns:
    c_index: C-index on the validation set
    """


    # Define a grid of parameters to search over
    param_grid = {
        'penalizer': [0.01, 0.1, 1, 10],  # L2 regularization strength
        'l1_ratio': [0]  # Mix of L1 and L2 regularization as [0, 0.1, 0.5, 0.9, 1]
        #zero emphasizes L2 regularization
    }


    # Combine X_train with Y_train_OS and Y_train_event to create the training data
    train_data = X_train.copy()
    train_data['OS'] = Y_train_OS
    train_data['event'] = Y_train_event

    # Filter out rows where OS is missing
    train_data = train_data.dropna(subset=['OS'])

    # Combine X_val with Y_val_OS and Y_val_event to create the validation data
    val_data = X_val.copy()
    val_data['OS'] = Y_val_OS
    val_data['event'] = Y_val_event

    # Filter out rows where OS is missing
    val_data = val_data.dropna(subset=['OS'])

    # Combine X_val with Y_val_OS and Y_val_event to create the TEST data
    test_data = X_test.copy()
    test_data['OS'] = Y_test_OS
    test_data['event'] = Y_test_event

    # Filter out rows where OS is missing
    test_data = test_data.dropna(subset=['OS'])

    # Grid search for the best parameters
    best_val_c_index = -np.inf
    best_params = {}

    for penalizer in param_grid['penalizer']:
        for l1_ratio in param_grid['l1_ratio']:
            # Initialize the CoxPH model with the current parameters
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            
            # Fit the model on the training data
            cph.fit(train_data, duration_col='OS', event_col='event')
            
            # Predict risk scores on the validation data
            val_data['risk_score'] = cph.predict_partial_hazard(val_data)
            
            # Calculate the C-index on the validation data
            c_index = concordance_index(val_data['OS'], -val_data['risk_score'], val_data['event'])
            
            # Check if this is the best model so far
            if c_index > best_val_c_index:
                best_val_c_index = c_index
                best_params = {'penalizer': penalizer, 'l1_ratio': l1_ratio}


    # Evaluate the model with the best parameters on the test set
    cph_best = CoxPHFitter(penalizer=best_params['penalizer'], l1_ratio=best_params['l1_ratio'])
    cph_best.fit(train_data, duration_col='OS', event_col='event')

    # Predict risk scores on the test data
    test_data['risk_score'] = cph_best.predict_partial_hazard(test_data)
    
    # Calculate the C-index on the test data
    best_test_c_index = concordance_index(test_data['OS'], -test_data['risk_score'], test_data['event'])

    # Return the best C-index on validation, C-index on test, and best parameters
    return best_val_c_index, best_test_c_index, best_params

   