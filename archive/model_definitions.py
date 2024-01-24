# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# %%
##############################
# Default parameter grids
##############################

# %%
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


# %%

##############################
# Random Search sklearn models
##############################

# %%
class RndSearchModel_Base:
    def __init__(self,base_model,param_grid):
        self.base_model = base_model
        self.param_grid = param_grid
        self.model = RandomizedSearchCV(
            base_model(),
            param_distributions=param_grid,
            n_iter=20,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=39,
            return_train_score=True,
        )

    def fit(self,X,y):
        self.model.fit(X,y)
        return self
    
    def predict(self,X):
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)
     

class RndSearchModel_RF(RndSearchModel_Base):
    def __init__(self,param_grid=random_forest_param_grid):
        super().__init__(RandomForestClassifier,param_grid)
        
class RndSearchModel_DT(RndSearchModel_Base):
    def __init__(self,param_grid=decision_tree_param_grid):
        super().__init__(DecisionTreeClassifier,param_grid)

class RndSearchModel_LR(RndSearchModel_Base):
    def __init__(self,param_grid=logistic_regression_param_grid):
        super().__init__(LogisticRegression,param_grid)

class RndSearchModel_SVC(RndSearchModel_Base):
    def __init__(self,param_grid=svc_param_grid):
        super().__init__(SVC,param_grid)


# %%

##############################
# Keras binary classifier models
##############################

# %%

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class KerasBinaryClassifier_Base:
    def __init__(self,input_size,learning_rate=0.001):
        self.input_size = input_size
        self.loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.auc_metric = tf.keras.metrics.AUC(from_logits=True)
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model = Sequential()
        self.model.add(Dense(1, input_dim=input_size))
        self.model.compile(self.loss_func, optimizer=self.optimizer,
               metrics=[self.auc_metric])
        self.history = None
        
    def fit(self,X,y,epochs=100,batch_size=32,val_data=None,callbacks=None):
        self.model.fit(X,y,epochs=epochs,batch_size=batch_size,validation_data=val_data,callbacks=callbacks)
        self.history = self.model.history.history
        return self
    
    def predict_logits(self,X):
        return self.model.predict(X)

    def predict(self,X):
        preds = self.model.predict(X)
        return tf.nn.sigmoid(preds).numpy()
