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

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

import pandas as pd



def log_reg_multi_class(task, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test):

    # preparing the data by removing NA values
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

    # Encode labels
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)


    # Define the parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
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
    y_test_pred = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    #print(f'Test Accuracy with best model: {test_accuracy:.4f}')

    return best_val_accuracy, test_accuracy







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
