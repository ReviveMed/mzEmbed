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

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


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
    best_val_f1 = 0
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
                val_f1 = f1_score(y_val, y_val_pred, average='weighted')

                # Get probability predictions for AUC calculation
                y_val_prob = clf.predict_proba(x_val)
                if len(label_encoder.classes_) == 2:  # Binary classification
                    y_val_prob = y_val_prob[:, 1]  # Extract probability for the positive class

                val_auc = roc_auc_score(y_val, y_val_prob, average='weighted', multi_class='ovr')

                # Check if this is the best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_val_auc = val_auc
                    best_val_f1 = val_f1
                    best_model = clf


    # Final evaluation on the test set using the best model
    y_test_pred = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Get probability predictions for AUC calculation on the test set
    y_test_prob = best_model.predict_proba(x_test)

    # Handle binary and multi-class classification separately
    if len(label_encoder.classes_) == 2:  # Binary classification
        y_test_prob = y_test_prob[:, 1]  # Extract probability for the positive class

    # Compute test AUC
    test_auc = roc_auc_score(y_test, y_test_prob, average='weighted', multi_class='ovr')

    print(f'Test Accuracy with best model: {test_accuracy:.4f}')
    print(f'Test AUC with best model: {test_auc:.4f}')
    print(f'Test F1 Score with best model: {test_f1:.4f}')

    return best_val_f1, best_val_auc, test_f1, test_auc



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

   


def cox_proportional_hazards_l1_sksurv(X_train, Y_train_OS, Y_train_event, 
                                       X_val, Y_val_OS, Y_val_event, 
                                       X_test, Y_test_OS, Y_test_event):
    """
    Train a Cox Proportional Hazards model with L1 regularization using scikit-survival,
    remove samples with NaN values, evaluate performance on the validation set, 
    and calculate the C-index.

    Parameters:
    X_train: Training feature vectors (DataFrame)
    Y_train_OS: Training survival time
    Y_train_event: Training event status (True if event occurred, False otherwise)
    X_val: Validation feature vectors (DataFrame)
    Y_val_OS: Validation survival time
    Y_val_event: Validation event status
    X_test: Test feature vectors (DataFrame)
    Y_test_OS: Test survival time
    Y_test_event: Test event status

    Returns:
    best_val_c_index: Best C-index on the validation set
    best_test_c_index: C-index on the test set with the best model
    best_alpha: Best alpha parameter for the Cox model (L1 regularization strength)
    """
    
    # Remove rows with NaN values from training, validation, and test sets
    train_data = pd.concat([X_train, pd.Series(Y_train_OS, name='OS'), pd.Series(Y_train_event, name='event')], axis=1).dropna()
    X_train_clean = train_data.drop(columns=['OS', 'event'])
    Y_train_OS_clean = train_data['OS'].values
    Y_train_event_clean = train_data['event'].values

    val_data = pd.concat([X_val, pd.Series(Y_val_OS, name='OS'), pd.Series(Y_val_event, name='event')], axis=1).dropna()
    X_val_clean = val_data.drop(columns=['OS', 'event'])
    Y_val_OS_clean = val_data['OS'].values
    Y_val_event_clean = val_data['event'].values

    test_data = pd.concat([X_test, pd.Series(Y_test_OS, name='OS'), pd.Series(Y_test_event, name='event')], axis=1).dropna()
    X_test_clean = test_data.drop(columns=['OS', 'event'])
    Y_test_OS_clean = test_data['OS'].values
    Y_test_event_clean = test_data['event'].values

    # Convert the cleaned labels into a structured array format (required by sksurv)
    Y_train = np.array([(event, os) for event, os in zip(Y_train_event_clean, Y_train_OS_clean)],
                       dtype=[('event', bool), ('OS', float)])

    Y_val = np.array([(event, os) for event, os in zip(Y_val_event_clean, Y_val_OS_clean)],
                     dtype=[('event', bool), ('OS', float)])

    Y_test = np.array([(event, os) for event, os in zip(Y_test_event_clean, Y_test_OS_clean)],
                      dtype=[('event', bool), ('OS', float)])

    # Define a grid of alpha (L1 regularization strength) to search over
    param_grid = {
        'alphas': [[0.01], [0.1], [1], [10]]  # L1 regularization strength [[0.001], [0.01], [0.1], [1], [10]]  
    }

    # Perform grid search to find the best regularization strength
    best_val_c_index = -np.inf
    best_alpha = None
    best_model = None

    for alpha_list in param_grid['alphas']:
        # Create and fit CoxnetSurvivalAnalysis model with L1 regularization
        model = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=alpha_list)
        model.fit(X_train_clean, Y_train)
        
        # Predict risk scores on the validation data
        risk_scores_val = model.predict(X_val_clean)
        
        # Calculate the C-index on the validation data
        concordance_val = concordance_index_censored(Y_val['event'], Y_val['OS'], risk_scores_val)[0]
        
        # Check if this is the best model so far
        if concordance_val > best_val_c_index:
            best_val_c_index = concordance_val
            best_alpha = alpha_list[0]
            best_model = model

    # Predict risk scores on the test data with the best model
    risk_scores_test = best_model.predict(X_test_clean)
    
    # Calculate the C-index on the test data
    best_test_c_index = concordance_index_censored(Y_test['event'], Y_test['OS'], risk_scores_test)[0]

    # Return the best C-index on validation, C-index on test, and best alpha
    return best_val_c_index, best_test_c_index, best_alpha



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
