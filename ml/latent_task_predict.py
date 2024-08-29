#Util functions for runnign the Logistic Regression model


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import RandomizedSearchCV


def log_reg_multi_class(x_train, y_train, x_val, y_val, x_test, y_test):

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

    return best_model, val_accuracy, test_accuracy





from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def ridge_regression_predict(x_train, y_train, x_val, y_val, x_test, y_test):
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
    
    # Define a range of alpha values to search over
    alpha_range = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = {'alpha': alpha_range}
    
    # Initialize Ridge regression model
    ridge = Ridge()
    
    # Use GridSearchCV to find the best alpha value
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    
    # Get the best model and its alpha value
    best_model = grid_search.best_estimator_
    best_alpha = grid_search.best_params_['alpha']
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(x_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    
    # Evaluate on test set
    y_test_pred = best_model.predict(x_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return best_model, val_mse, test_mse, test_r2, best_alpha
