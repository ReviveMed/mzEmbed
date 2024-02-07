import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from sklearn.preprocessing import LabelEncoder, StandardScaler,FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV


import keras
from keras import layers
from keras import regularizers
from keras.losses import BinaryCrossentropy
import random
import datetime
from scipy.stats import uniform, randint
import time

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFpr, f_classif


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2, l1, l2


from sklearn_models.evaluation import generate_model_summary

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
def create_sklearn_model_summary(base_model,model_name,param_grid,input_dir):
    print(f'Running {model_name} model')
    # X_pretrain = pd.read_csv(f'{input_dir}/X_pretrain.csv',index_col=0)
    X_train = pd.read_csv(f'{input_dir}/X_train.csv',index_col=0)
    X_test = pd.read_csv(f'{input_dir}/X_test.csv',index_col=0)
    y_train = pd.read_csv(f'{input_dir}/y_train.csv',index_col=0)
    y_test = pd.read_csv(f'{input_dir}/y_test.csv',index_col=0)
    if os.path.exists(f'{input_dir}/X_val.csv'):
        X_val = pd.read_csv(f'{input_dir}/X_val.csv',index_col=0)
        y_val = pd.read_csv(f'{input_dir}/y_val.csv',index_col=0)
    if os.path.exists(f'{input_dir}/X_test_alt.csv'):
        X_test_alt = pd.read_csv(f'{input_dir}/X_test_alt.csv',index_col=0)
        y_test_alt = pd.read_csv(f'{input_dir}/y_test_alt.csv',index_col=0)

    output_dir = os.path.join(input_dir,'classical_models')
    os.makedirs(output_dir,exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    rs = RandomizedSearchCV(
        base_model(),
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=39,
        return_train_score=True,
    )

    # Fit the RandomizedSearchCV object to the data
    rs.fit(X_train, y_train.values.ravel())

    results_list = []
    for ii in range(len(rs.cv_results_['params'])):
        results_list.append({'val_scores': [], 'test_scores': [], 'train_scores': [], 'test_alt_scores': []})

    num_repeats = 5
    if os.path.exists(f'{input_dir}/X_val.csv'):
        for jj in range(num_repeats):
            for ii, params in enumerate(rs.cv_results_['params']):
                model = base_model(**params)
                # shuffle order of training data
                y_train = y_train.sample(frac=1,random_state=jj)
                X_train = X_train.loc[y_train.index,:]
                model.fit(X_train, y_train.values.ravel())

                pred_proba = model.predict_proba(X_val)[:,1]
                results_list[ii]['val_scores'].append(roc_auc_score(y_val.values, pred_proba))

                pred_proba = model.predict_proba(X_test)[:,1]
                results_list[ii]['test_scores'].append(roc_auc_score(y_test.values, pred_proba))

                pred_proba = model.predict_proba(X_train)[:,1]
                results_list[ii]['train_scores'].append(roc_auc_score(y_train.values, pred_proba))

                if os.path.exists(f'{input_dir}/X_test_alt.csv'):
                    pred_proba = model.predict_proba(X_test_alt)[:,1]
                    results_list[ii]['test_alt_scores'].append(roc_auc_score(y_test_alt.values, pred_proba))
    
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(X_train, y_train):
            for ii, params in enumerate(rs.cv_results_['params']):
                model = base_model(**params)
                model.fit(X_train.iloc[train_index], y_train.iloc[train_index].values.ravel())
                
                pred_proba = model.predict_proba(X_train.iloc[test_index])[:,1]
                results_list[ii]['val_scores'].append(roc_auc_score(y_train.iloc[test_index].values, pred_proba))

                pred_proba = model.predict_proba(X_test)[:,1]
                results_list[ii]['test_scores'].append(roc_auc_score(y_test.values, pred_proba))

                pred_proba = model.predict_proba(X_train.iloc[train_index])[:,1]
                results_list[ii]['train_scores'].append(roc_auc_score(y_train.iloc[train_index].values, pred_proba))

                if os.path.exists(f'{input_dir}/X_test_alt.csv'):
                    pred_proba = model.predict_proba(X_test_alt)[:,1]
                    results_list[ii]['test_alt_scores'].append(roc_auc_score(y_test_alt.values, pred_proba))


    params_df = pd.DataFrame(rs.cv_results_['params'])
    # if os.path.exists(f'{input_dir}/X_test_alt.csv'):
    #     summary_df = pd.DataFrame(rs.cv_results_)[['mean_test_score','std_test_score','mean_train_score', 'std_train_score',
    #                                                'mean_'
    # else:
    summary_df = pd.DataFrame(rs.cv_results_)[['mean_test_score','std_test_score','mean_train_score', 'std_train_score']]
    summary_df.columns = ['RndGrid '+ col for col in summary_df.columns]
    summary_df = pd.concat([params_df,summary_df],axis=1)
    summary_df['Model'] = model_name


    result_df_means_list = []
    result_df_stds_list = []
    for res in results_list:
        df = pd.DataFrame(res)
        result_df_means_list.append(df.mean())
        result_df_stds_list.append(df.std())

    result_df_means = pd.concat(result_df_means_list,axis=1).T
    result_df_stds = pd.concat(result_df_stds_list,axis=1).T
    result_df_means.columns = [col+ '_mean' for col in result_df_means.columns]
    result_df_stds.columns = [col+ '_std' for col in result_df_stds.columns]

    result_df = pd.concat([result_df_means,result_df_stds],axis=1).round(3)
    # result_df['Model'] = model_name

    summary_df = pd.concat([result_df,summary_df],axis=1)

    summary_df.to_csv(os.path.join(output_dir,f'{model_name}_summary_{date_str}.csv'))

    return summary_df


def create_autoencoder_model(input_dir,
                         encoder_name,
                         epochs=50,
                         batch_size=50,
                         encoder_patience=10,
                         activation='sigmoid',
                         l1_val=0.00001,
                         dropout_val=0,
                         bottleneck_sz=8,
                         neuron_num_list=[],
                         pretrained_autoenoder=None,
                         pretrain_subset_idx=None,
                         include_training_data=False,
                         my_output_dir=None,
                         pretrain_filename='X_pretrain.csv'):

    if my_output_dir is None:
        my_output_dir = input_dir

    if pretrain_filename is None:
        pretrain_filename = 'X_pretrain.csv'

    encoder_path = f'{my_output_dir}/pretrained_{encoder_name}.keras'
    if os.path.exists(encoder_path):
        print(f'Encoder {encoder_name} already exists')
        return

    X_pretrain = pd.read_csv(f'{input_dir}/{pretrain_filename}',index_col=0)
    if include_training_data:
        X_pretrain = pd.concat([X_pretrain,pd.read_csv(f'{input_dir}/X_train.csv',index_col=0)])

    if pretrain_subset_idx is not None:
        pretrain_subset_idx = np.intersect1d(pretrain_subset_idx,X_pretrain.index)
        if len(pretrain_subset_idx) < 20:
            print(f'Not enough samples to pretrain {encoder_name}')
            return
        X_pretrain = X_pretrain.loc[pretrain_subset_idx,:]

    if pretrained_autoenoder is None:
        input_sz = X_pretrain.shape[1]
        l1_val = 0.00001

        num_layers = len(neuron_num_list)

        if l1_val > 0:
            kernel_regularizer = regularizers.l1(l1_val)
        else:
            kernel_regularizer = None


        encoder = Sequential()
        encoder.add(Input(shape=(input_sz,)))
        if num_layers > 0:
            for i in range(num_layers):
                neuron_num = neuron_num_list[i]
                encoder.add(Dense(neuron_num, activation=activation, kernel_regularizer=kernel_regularizer))
                if dropout_val > 0:
                    encoder.add(Dropout(dropout_val))

        encoder.add(Dense(bottleneck_sz, activation=activation, kernel_regularizer=kernel_regularizer))


        decoder = Sequential()
        decoder.add(Input(shape=(bottleneck_sz,)))
        if num_layers > 0:
            for i in range(num_layers,0,-1):
                neuron_num = neuron_num_list[i-1]
                decoder.add(Dense(neuron_num, activation=activation, kernel_regularizer=kernel_regularizer))
                if dropout_val > 0:
                    decoder.add(Dropout(dropout_val))

        decoder.add(Dense(input_sz, activation='tanh'))     

        autoencoder = Sequential()
        autoencoder.add(encoder)
        autoencoder.add(decoder)

        autoencoder.compile(optimizer='adam', loss='mse')

        encoder.save(f'{my_output_dir}/untrained_{encoder_name}.keras')

        X_pretrain_train, X_pretrain_val, _, _ = train_test_split(X_pretrain,X_pretrain.iloc[:,0] ,test_size=0.2, random_state=42)

        h= autoencoder.fit(X_pretrain_train, X_pretrain_train,epochs=epochs,batch_size=batch_size,validation_data=(X_pretrain_val, X_pretrain_val),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=encoder_patience, restore_best_weights=True, mode='auto')])

        encoder.save(f'{my_output_dir}/pretrained_{encoder_name}.keras')
        decoder.save(f'{my_output_dir}/pretrained_decoder_{encoder_name}.keras')
    else:
        encoder = load_model(f'{my_output_dir}/pretrained_{pretrained_autoenoder}.keras')
        decoder = load_model(f'{my_output_dir}/pretrained_decoder_{pretrained_autoenoder}.keras')
        autoencoder = Sequential()
        autoencoder.add(encoder)
        autoencoder.add(decoder)
        autoencoder.compile(optimizer='adam', loss='mse')

        X_pretrain_train, X_pretrain_val, _, _ = train_test_split(X_pretrain,X_pretrain.iloc[:,0] ,test_size=0.2, random_state=42)

        h= autoencoder.fit(X_pretrain_train, X_pretrain_train,epochs=epochs,batch_size=batch_size,validation_data=(X_pretrain_val, X_pretrain_val),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=encoder_patience, restore_best_weights=True, mode='auto')])

        encoder.save(f'{my_output_dir}/pretrained_{encoder_name}.keras')
        decoder.save(f'{my_output_dir}/pretrained_decoder_{encoder_name}.keras')


    autoencoder_stats = {}
    autoencoder_stats['number of encoder training samples'] = X_pretrain.shape[0]
    autoencoder_stats['include training data'] = include_training_data
    autoencoder_stats['encoder epochs'] = epochs
    autoencoder_stats['encoder batch_size'] = batch_size
    autoencoder_stats['encoder patience'] = encoder_patience
    autoencoder_stats['encoder val loss'] = h.history['val_loss'][-1]
    autoencoder_stats['encoder train loss'] = h.history['loss'][-1]
    autoencoder_stats['pre-trained autoencoder'] = pretrained_autoenoder
    autoencoder_stats = pd.Series(autoencoder_stats)
    autoencoder_stats.to_csv(f'{my_output_dir}/autoencoder_stats_{encoder_name}.csv')

    plt.plot(h.history['loss'], color='blue',label='loss')
    plt.plot(h.history['val_loss'], color='orange',label='val_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Loss')
    plt.savefig(f'{my_output_dir}/autoencoder_loss_{encoder_name}.png')
    plt.close()
    return


def create_classifier_model(input_sz,
                            input_dir,
                                n_neurons=32,
                              pre_trained=True,
                              activation='relu',
                              dropout_val=0,
                              no_encoder=False,
                              batch_norm=False,
                              n_layers=1,
                              learning_rate=0.0001,
                              custom_encoder_path=None,
                              encoder_name = 'encoder_Aug23',
                              encoder_trainable=True,
                              my_output_dir=None,
                              use_predefined_val=None,
                              **kwargs):

    if my_output_dir is None:
        my_output_dir = input_dir

    if custom_encoder_path is not None:
        encoder = load_model(custom_encoder_path)
        encoder.trainable = True
    
    else:
        if pre_trained:
            if os.path.exists(f'{my_output_dir}/pretrained_{encoder_name}.keras'):
                encoder = load_model(f'{my_output_dir}/pretrained_{encoder_name}.keras')
            else:
                print(f'pre-trained Encoder {encoder_name} does not exist')
                return None
        else:
            if os.path.exists(f'{my_output_dir}/untrained_{encoder_name}.keras'):
                encoder = load_model(f'{my_output_dir}/untrained_{encoder_name}.keras')
            else:
                print(f'untrained Encoder {encoder_name} does not exist')
                return None
    
    if encoder_trainable:
        encoder.trainable = True
    else:
        encoder.trainable = False
    
    if no_encoder:
        encoder = Input(shape=(input_sz,))
    
    
    classifier_model = Sequential()
    classifier_model.add(encoder)
    # add a batch normalization layer
    if batch_norm:
        classifier_model.add(BatchNormalization())
    for ii in range(n_layers):
        if dropout_val>0:
            classifier_model.add(Dropout(dropout_val))
        classifier_model.add(Dense(n_neurons, activation=activation)) 

    # classifier_model.add(Dense(1, activation='sigmoid'))
    classifier_model.add(Dense(1))

    # classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    loss_func = BinaryCrossentropy(from_logits=True)
    auc_metric = keras.metrics.AUC(from_logits=True)
    classifier_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=loss_func, metrics=[auc_metric])
    # classifier_model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['AUC'])

    return classifier_model


def run_network_tests(input_dir,epochs=50,batch_size=100,**kwargs):
    if 'use_predefined_val' not in kwargs:
        use_predefined_val = False
    else:
        use_predefined_val = kwargs['use_predefined_val']
    use_alt_test = kwargs.get('use_alt_test',False)
    if 'custom_encoder_path' in kwargs:
        custom_encoder_path = kwargs['custom_encoder_path']
    else:
        custom_encoder_path = None

    start_time = time.time()
    print(f'Running {kwargs["encoder_name"]} {kwargs["n_neurons"]} {kwargs["n_layers"]} {kwargs["activation"]} {epochs} {batch_size}')
    # print('run network tests: ', kwargs)
    if 'my_output_dir' in kwargs:
        my_output_dir = kwargs['my_output_dir']
    else:
        my_output_dir = input_dir
    
    X_train = pd.read_csv(f'{input_dir}/X_train.csv',index_col=0)
    y_train = pd.read_csv(f'{input_dir}/y_train.csv',index_col=0)
    if use_alt_test:
        X_test_alt = pd.read_csv(f'{input_dir}/X_test_alt.csv',index_col=0)
        y_test_alt = pd.read_csv(f'{input_dir}/y_test_alt.csv',index_col=0)
    X_test = pd.read_csv(f'{input_dir}/X_test.csv',index_col=0)
    y_test = pd.read_csv(f'{input_dir}/y_test.csv',index_col=0)
    if use_predefined_val:
        X_val = pd.read_csv(f'{input_dir}/X_val.csv',index_col=0)
        y_val = pd.read_csv(f'{input_dir}/y_val.csv',index_col=0)

    # check the data type of the y_train
    if y_train.dtypes[0] == 'object':
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train.values.ravel()),index=y_train.index)
        y_test = pd.Series(le.transform(y_test.values.ravel()),index=y_test.index)
        if use_predefined_val:
            y_val = pd.Series(le.transform(y_val.values.ravel()),index=y_val.index)


    my_result_output_dir = os.path.join(my_output_dir, 'results')
    os.makedirs(my_result_output_dir, exist_ok=True)

    input_sz = X_train.shape[1]

    no_encoder_results = {'model_name': 'no_encoder'}
    no_pretrain_results = {'model_name': 'no_pretrain'}
    yes_pretrain_results = {'model_name': 'yes_pretrain'}
    fixed_pretrain_results = {'model_name': 'fixed_pretrain'}
    
    # save the kwargs used to create the model to a csv file
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_kwards = pd.DataFrame(kwargs, index=[0])
    # model_kwards.to_csv(f'{my_result_output_dir}/model_kwargs_{date_str}.csv')
    model_kwards['epochs'] = epochs
    model_kwards['batch_size'] = batch_size
    encoder_name = kwargs['encoder_name']
    # model_kwards.append({'epochs': epochs, 'batch_size': batch_size}, ignore_index=True)
    model_kwards.to_csv(f'{my_result_output_dir}/model_kwargs_{encoder_name}_{date_str}.csv')

    # results_list = [no_encoder_results, no_pretrain_results, yes_pretrain_results,fixed_pretrain_results]
    if custom_encoder_path:
        results_list = [yes_pretrain_results]
    else:
        # results_list = [no_encoder_results, no_pretrain_results, yes_pretrain_results]
        results_list = [no_pretrain_results, yes_pretrain_results]

    for results in results_list:
        results['val_scores'] = []
        results['test_scores'] = []
        results['train_scores'] = []
        results['test_alt_scores'] = []

    if use_predefined_val:
        n_repeats = 10
        for i in range(n_repeats):
            # randomly permutate the training data
            y_train = y_train.sample(frac=1,random_state=i)
            X_train = X_train.loc[y_train.index,:]

            if custom_encoder_path:
                model_2 = create_classifier_model(input_sz,input_dir,pre_trained=True,**kwargs)
                model_list = [model_2]
            else:
                # model_0 = create_classifier_model(input_sz,input_dir,no_encoder=True,**kwargs)
                model_1 = create_classifier_model(input_sz,input_dir,pre_trained=False,**kwargs)
                model_2 = create_classifier_model(input_sz,input_dir,pre_trained=True,**kwargs)
                # model_list = [model_0, model_1, model_2]
                model_list = [model_1, model_2]

            for ii,model in enumerate(model_list):
                    if model is None:
                        continue
                    
                    results = results_list[ii]
                    model_name = results["model_name"]
                    print(f'model {encoder_name}, {model_name}, repeat {i}')
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))
                    # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                    # save the model training history
                    pd.DataFrame(model.history.history).to_csv(f'{my_result_output_dir}/model_history_{encoder_name}_{model_name}_{date_str}.csv')

                    preds = model.predict(X_val)
                    # transform the preds with a sigmoid
                    preds = 1/(1+np.exp(-preds))
                    results['val_scores'].append(roc_auc_score(y_val, preds))

                    preds = model.predict(X_train)
                    # transform the preds with a sigmoid
                    preds = 1/(1+np.exp(-preds))
                    results['train_scores'].append(roc_auc_score(y_train, preds))

                    preds = model.predict(X_test)
                    # transform the preds with a sigmoid
                    preds = 1/(1+np.exp(-preds))
                    results['test_scores'].append(roc_auc_score(y_test, preds))

                    if use_alt_test:
                        preds = model.predict(X_test_alt)
                        # transform the preds with a sigmoid
                        preds = 1/(1+np.exp(-preds))
                        results['test_alt_scores'].append(roc_auc_score(y_test_alt, preds))


    else:        
        k = 5
        skf = StratifiedKFold(n_splits=k,random_state=42,shuffle=True)

        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            
            if custom_encoder_path:
                model_2 = create_classifier_model(input_sz,input_dir,pre_trained=True,**kwargs)
                model_list = [model_2]
            else:
                # model_0 = create_classifier_model(input_sz,input_dir,no_encoder=True,**kwargs)
                model_1 = create_classifier_model(input_sz,input_dir,pre_trained=False,**kwargs)
                model_2 = create_classifier_model(input_sz,input_dir,pre_trained=True,**kwargs)
                # model_list = [model_0, model_1, model_2]
                model_list = [model_1, model_2]

            # model_3 = create_classifier_model_2(pre_trained=True, encoder_trainable=False,**kwargs)
            # model_list = [model_0, model_1, model_2, model_3]
            # model_list = [model_0, model_1, model_2]

            for ii,model in enumerate(model_list):
                if model is None:
                    continue
                
                results = results_list[ii]
                model_name = results["model_name"]
                print(f'model {encoder_name}, {model_name} fold {i}')
                model.fit(X_train.iloc[train_index], y_train.iloc[train_index], epochs=epochs, batch_size=batch_size, verbose=0)
                #TODO: save with conversion to probability from logits
                # preds = model.predict(X_train.iloc[test_index])
                # results['val_scores'].append(roc_auc_score(y_train.iloc[test_index], preds))

                # train_preds = model.predict(X_train.iloc[train_index])
                # results['train_scores'].append(roc_auc_score(y_train.iloc[train_index], train_preds))

                # test_preds = model.predict(X_test)
                # results['test_scores'].append(roc_auc_score(y_test, test_preds))

                # if use_alt_test:
                #     test_preds = model.predict(X_test_alt)
                #     results['test_alt_scores'].append(roc_auc_score(y_test_alt, test_preds))

    for result in results_list:
        print(result['model_name'])
        print('Train: ', np.mean(result['train_scores']))
        print('Val: ', np.mean(result['val_scores']))
        print('Test: ', np.mean(result['test_scores']))
        if use_alt_test:
            print('Test Alt: ', np.mean(result['test_alt_scores']))

    # save model visualization
    keras.utils.plot_model(model, to_file=f'{my_result_output_dir}/model_{encoder_name}_{date_str}.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)

    all_results = []
    for result in results_list:
        all_results.append(pd.DataFrame(result))
    result_df = pd.concat(all_results)
    result_df.to_csv(f'{my_result_output_dir}/results_{encoder_name}_{date_str}.csv')

    result_df_summary = result_df.groupby('model_name').mean().round(3)
    result_df_summary.rename(columns={'val_scores': 'Val AUC', 'test_scores': 'Test AUC', 'train_scores': 'Train AUC', 'test_alt_scores': 'Test Alt AUC'}, inplace=True)
    # sort the columns
    if use_alt_test:
        result_df_summary = result_df_summary[['Train AUC','Val AUC','Test AUC','Test Alt AUC']]
    else:
        result_df_summary = result_df_summary[['Train AUC','Val AUC','Test AUC']]
    result_df_summary['Train AUC Std'] = result_df.groupby('model_name').std()['train_scores'].round(3)
    result_df_summary['Val AUC Std'] = result_df.groupby('model_name').std()['val_scores'].round(3)
    result_df_summary['Test AUC Std'] = result_df.groupby('model_name').std()['test_scores'].round(3)
    if use_alt_test:
        result_df_summary['Test Alt AUC Std'] = result_df.groupby('model_name').std()['test_alt_scores'].round(3)
    result_df_summary.to_csv(f'{my_result_output_dir}/results_summary_{encoder_name}_{date_str}.csv')


    elapsed_time_mins = (time.time() - start_time)/60
    print(f'Elapsed time: {elapsed_time_mins} minutes')

    return result_df



# %%
def run_performance_eval(n_layers,n_neurons,activation,
                         input_dir,
                         output_subdir='keras_autoencoder_models',
                         epoch_list=None,
                         custom_encoder_path=None,
                         pretrain_filename=None,
                         dropout_val=0,
                         use_alt_test=False):
    
    if epoch_list is None:
        # epoch_list= [10,20,30,40,50,60,70,80,90,100,120,150,200,250]
        # epoch_list = [5,25,50,75,100,125,150,200,250,300,400,500,600,750,1000]
        # epoch_list = [50,100,150,250,500]
        # epoch_list = [75,125,250]
        epoch_list = [125,250]
        # epoch_list = [1000]

    output_dir = os.path.join(input_dir,output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    #n_layers is the number of hidden layers

    if os.path.exists(os.path.join(input_dir,'X_val.csv')):
        use_predefined_val = True
    else:
        use_predefined_val = False

    if custom_encoder_path is not None:
        for epochs in epoch_list:
            # for activation in ['sigmoid','relu']:
                encoder_name = os.path.basename(custom_encoder_path).replace('.keras','')
                run_network_tests(input_dir,
                                n_neurons=0,
                                encoder_name=encoder_name,
                                n_layers=0,
                                activation=activation,
                                epochs=epochs,
                                batch_size=50,
                                my_output_dir=output_dir,
                                use_predefined_val=use_predefined_val,
                                custom_encoder_path=custom_encoder_path)

    else:
        for encoder_patience in [15]:#,20]:
            if pretrain_filename is None:
                encoder_name = f'{n_layers}_{n_neurons}_{activation}_{dropout_val}_L1_{encoder_patience}'
            else:
                encoder_name = f'{n_layers}_{n_neurons}_{activation}_{dropout_val}_L1_{encoder_patience}_{pretrain_filename}'
            
            create_autoencoder_model(input_dir,
                                encoder_name,
                                epochs=1000,
                                batch_size=50,
                                encoder_patience=encoder_patience,
                                activation=activation,
                                l1_val=0.00001,
                                dropout_val=dropout_val,
                                bottleneck_sz=n_neurons,
                                neuron_num_list=[n_neurons]*n_layers,
                                pretrained_autoenoder=None,
                                my_output_dir=output_dir,
                                pretrain_filename=pretrain_filename)
        
            for epochs in epoch_list:
                # for activation in ['sigmoid','relu']:

                    run_network_tests(input_dir,
                                    n_neurons=0,
                                    encoder_name=encoder_name,
                                    n_layers=0,
                                    activation=activation,
                                    epochs=epochs,
                                    batch_size=32,
                                    my_output_dir=output_dir,
                                    use_predefined_val=use_predefined_val,
                                    use_alt_test=use_alt_test)

    return



# %%
base_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton/development_CohortCombination'
if not os.path.exists(base_dir):
    base_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination'
# input_dir = f'{base_dir}/data_2023_october/selected_studiesX 13/combat_mskcc_binary_task'
# input_dir = f'{base_dir}/data_2023_october/selected_studiesX 13/combat_survival_class_task'
# input_dir = f'{base_dir}/hilic_pos_2023_nov_21/subset all_studies/num_cohorts_thresh_0.5/combat_Sex'
# input_dir = f'{base_dir}/hilic_pos_2023_nov_21/subset all_studies/num_cohorts_thresh_0.5/combat_survival class'
# input_dir = f'{base_dir}/hilic_pos_2024_jan_25/subset all_studies from Jan25_alignments/num_cohorts_thresh_0.5/combat_Benefit'
# input_dir = f'{base_dir}/data_2023_october/selected_studiesX 13/combat_mskcc_binary_task'
# data_engine_path = '/Users/jonaheaton/Desktop/Data-engine'

date_name = 'hilic_pos_2024_feb_01_read_norm'
# study_subset_name = 'subset all_studies with align score 0 from Eclipse_align_80_40_default'
# study_subset_name = 'subset all_studies with align score 0 from Merge_Jan25_align_80_40_default'
# study_subset_name = 'subset all_studies with align score 0.25 from Merge_Jan25_align_80_40_default'
feat_subset_name = 'num_cohorts_thresh_0.5'
task_name = 'combat_Benefit'

study_subset_name_list = ['subset all_studies with align score 0.25 from Merge_Jan25_align_80_40_default']#,'subset all_studies with align score 0 from Eclipse_align_80_40_default']

for study_subset_name in study_subset_name_list:

    input_dir = f'{base_dir}/{date_name}/{study_subset_name}/{feat_subset_name}/{task_name}'

    skip_sklearn = False
    # check if the sklearn models have already been run
    sklearn_output_dir = os.path.join(input_dir,'classical_models')
    if os.path.exists(sklearn_output_dir):
        files_in_sklearn = [file for file in os.listdir(sklearn_output_dir) if file.endswith('.csv')]
        for file in files_in_sklearn:
            if 'decision_tree_summary' in file:
                print('sklearn models already run')
                skip_sklearn= True
                continue


    if (not skip_sklearn):

        # my_output_dir = f'/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination/simple_networks_Aug_2023/datasets/Aug22_{freq_th}'
        # input_dir = data_engine_path


        base_model = LogisticRegression
        model_name = 'logistic_regression'
        param_grid = logistic_regression_param_grid
        create_sklearn_model_summary(base_model,model_name,param_grid,input_dir)

        base_model = RandomForestClassifier
        model_name = 'random_forest'
        param_grid = random_forest_param_grid
        create_sklearn_model_summary(base_model,model_name,param_grid,input_dir)

        base_model = SVC
        model_name = 'svc'
        param_grid = svc_param_grid
        create_sklearn_model_summary(base_model,model_name,param_grid,input_dir)

        base_model = DecisionTreeClassifier
        model_name = 'decision_tree'
        param_grid = decision_tree_param_grid
        create_sklearn_model_summary(base_model,model_name,param_grid,input_dir)

        generate_model_summary(sklearn_output_dir)
    else:
        print('skipping rerun of sklearn models')
        generate_model_summary(sklearn_output_dir)



    # %%
    if False:
        run_performance_eval(n_layers=0,n_neurons=0,activation='sigmoid',
                                input_dir=input_dir,
                                output_subdir='adde_autoencoder_models3',
                                custom_encoder_path='/Users/jonaheaton/Desktop/cohort_combine_oct16/selected_studiesX 13/ADV_DE_Model_on_combat/ADV_encoder_ALL_DATASETS_EXCEPT_[0, 1]_32L_lam1_fold42_epochs1000.keras'
                                # custom_encoder_path= '/Users/jonaheaton/Desktop/cohort_combine_oct16/selected_studiesX 13/ADV_DE_Model_on_combat1/ADV_encoder_ALL_DATASETS_EXCEPT_[0, 1]_32L_lam1_fold42.keras'
        )


    if False:
        # the number of hidden layers
        for n_layers in [0]:
            for n_neurons in [128]:
                for dropout_val in [0,0.2]:
                    run_performance_eval(n_layers,n_neurons,'sigmoid',
                                        input_dir=input_dir,
                                        epoch_list = [125,250],
                                        dropout_val=dropout_val,
                                        output_subdir='keras_autoencoder_models5',
                                        # pretrain_filename='X_pretrain_farmm_and_rcc3_bs.csv')
                                        # pretrain_filename='X_pretrain_rcc3_bs.csv')
                                        pretrain_filename='X_pretrain.csv',
                                        use_alt_test=True)


        for n_layers in []:
            for n_neurons in [32,64]:
                for dropout_val in [0,0.2]:
                    run_performance_eval(n_layers,n_neurons,'sigmoid',
                                        input_dir=input_dir,
                                        epoch_list = [250,500],
                                        dropout_val=dropout_val,
                                        output_subdir='keras_autoencoder_models5',
                                        # pretrain_filename='X_pretrain_farmm_and_rcc3_bs.csv')
                                        # pretrain_filename='X_pretrain_rcc3_bs.csv')
                                        pretrain_filename='X_pretrain.csv',
                                        use_alt_test=True)

                # run_performance_eval(n_layers,n_neurons,'sigmoid',
                #                     input_dir=input_dir,
                #                     output_subdir='keras_autoencoder_models5',
                #                     # pretrain_filename='X_pretrain_farmm_and_rcc3_bs.csv')
                #                     pretrain_filename='X_pretrain_rcc3_bs.csv')
                #                     # pretrain_filename='X_pretrain.csv')


    # # %%
    # data_engine_path = '/Users/jonaheaton/Desktop/Data-engine'
    # input_dir = data_engine_path

    # output_dir = os.path.join(input_dir,'autoencoder_models')
    # os.makedirs(output_dir, exist_ok=True)
    # encoder_name = '16_16_16_relu_L1'

    # create_autoencoder_model(input_dir,
    #                         encoder_name,
    #                         epochs=500,
    #                         batch_size=50,
    #                         encoder_patience=10,
    #                         activation='relu',
    #                         l1_val=0.00001,
    #                         dropout_val=0,
    #                         bottleneck_sz=16,
    #                         neuron_num_list=[16,16],
    #                         pretrained_autoenoder=None,
    #                         my_output_dir=output_dir)

    # # %%
    # encoder_list = ['16_16_16_relu_L1']

    # for encoder_id in encoder_list:
    #                 for epochs in [10,20,50,100,200]:
    #                     for activation in ['relu']:
    #                     # for activation in ['sigmoid','relu']:

    #                         run_network_tests(input_dir,
    #                                         n_neurons=0,
    #                                         encoder_name=encoder_id,
    #                                         n_layers=0,
    #                                         activation=activation,
    #                                         epochs=epochs,
    #                                         batch_size=50,
    #                                         my_output_dir=output_dir)