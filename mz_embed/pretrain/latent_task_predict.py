#Util functions for runnign the Logistic Regression model


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score


def log_reg_multi_class(task, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test):
    # Drop rows with NaN values in the target columns
    valid_train_indices = y_data_train[task].dropna().index
    valid_val_indices = y_data_val[task].dropna().index
    valid_test_indices = y_data_test[task].dropna().index

    # Filter y and Z datasets to remove NaNs
    y_train = y_data_train.loc[valid_train_indices, task]
    y_val = y_data_val.loc[valid_val_indices, task]
    y_test = y_data_test.loc[valid_test_indices, task]

    x_train = x_data_train.loc[valid_train_indices]
    x_val = x_data_val.loc[valid_val_indices]
    x_test = x_data_test.loc[valid_test_indices]
    #print (y_train.shape, x_train.shape, y_val.shape, x_val.shape, y_test.shape, x_test.shape)

    # Encode labels
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    #print (y_train.shape, x_train.shape, y_val.shape, x_val.shape, y_test.shape, x_test.shape)

    # Define the parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
    }

    best_val_accuracy = 0
    best_val_auc = 0
    best_model = None
    test_auc=0
    val_auc=0

    # Manual grid search over hyperparameters
    for C in param_grid['C']:
        for penalty in param_grid['penalty']:
            for solver in param_grid['solver']:
                clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)
                clf.fit(x_train, y_train)

                # Evaluate on the validation set
                y_val_pred = clf.predict(x_val)
                val_accuracy = accuracy_score(y_val, y_val_pred)

                # Get probability predictions for AUC calculation
                y_val_prob = clf.predict_proba(x_val)
                if len(label_encoder.classes_) == 2:  # Binary classification
                    y_val_prob = y_val_prob[:, 1]  # Extract probability for the positive class

                val_auc = roc_auc_score(y_val, y_val_prob, average='macro', multi_class='ovr')

                # Check if this is the best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_val_auc = val_auc
                    best_model = clf


    # Final evaluation on the test set using the best model
    y_test_pred = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Get probability predictions for AUC calculation on the test set
    y_test_prob = best_model.predict_proba(x_test)

    # Handle binary and multi-class classification separately
    if len(label_encoder.classes_) == 2:  # Binary classification
        y_test_prob = y_test_prob[:, 1]  # Extract probability for the positive class

    # Compute test AUC
    test_auc = roc_auc_score(y_test, y_test_prob, average='macro', multi_class='ovr')

    print(f'Test Accuracy with best model: {test_accuracy:.4f}')
    print(f'Test AUC with best model: {test_auc:.4f}')

    return best_val_accuracy, best_val_auc, test_accuracy, test_auc


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

   


def ridge_regression_predict(task, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test):
    """
    Train a Ridge regression model to predict a numerical value (e.g., age),
    tune the alpha parameter using the validation set, and evaluate performance on the test set.

    Parameters:
    x_train: Training feature vectors
    y_train: Training target values
    x_val: Validation feature vectors
    y_val: Validation target values
    x_test: Test feature vectors
    y_test: Test target values

    Returns:
    best_model: Trained regression model with the best alpha
    val_mse: Mean Squared Error on the validation set
    test_mse: Mean Squared Error on the test set
    test_r2: R^2 score on the test set
    best_alpha: The alpha value that gave the best validation performance
    """

    #preparing the input by removing the NA values
    
    # y_train = y_data_train[task]
    # non_nan_indices = y_train.dropna().index
    # y_train = y_train.dropna()
    # x_train = x_data_train.loc[non_nan_indices]

    # y_val = y_data_val[task]
    # non_nan_indices = y_val.dropna().index
    # y_val = y_val.dropna()
    # x_val = x_data_val.loc[non_nan_indices]

    # y_test = y_data_test[task]
    # non_nan_indices = y_test.dropna().index
    # y_test = y_test.dropna()
    # x_test = x_data_test.loc[non_nan_indices]

    # Preparing the input by removing the NA values
    y_train = y_data_train[task].dropna()
    x_train = x_data_train.loc[y_train.index]

    y_val = y_data_val[task].dropna()
    x_val = x_data_val.loc[y_val.index]

    y_test = y_data_test[task].dropna()
    x_test = x_data_test.loc[y_test.index]

    # Ensure that the input data is 2D
    if isinstance(x_train, pd.Series):
        x_train = x_train.to_frame()
    if isinstance(x_val, pd.Series):
        x_val = x_val.to_frame()
    if isinstance(x_test, pd.Series):
        x_test = x_test.to_frame()



    # Define a range of alpha values to search over
    alpha_range = [0.001, 0.01, 0.1, 1, 10, 100]
    
    best_alpha = None
    best_val_mse = float('inf')
    best_model = None
    

    # Iterate over the range of alpha values
    for alpha in alpha_range:
        # Initialize and train Ridge regression model with the current alpha
        ridge = Ridge(alpha=alpha)
        ridge.fit(x_train, y_train)
        
        # Evaluate on the validation set
        y_val_pred = ridge.predict(x_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        
        # If the current model has the lowest validation MSE, save it
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = alpha
            best_model = ridge


    # Evaluate the best model on the validation set using MAE
    y_val_pred = best_model.predict(x_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(x_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return best_val_mse, val_mae, val_r2, test_mse, test_mae, test_r2
