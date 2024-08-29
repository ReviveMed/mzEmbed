
'''
evaluting pre-trained latnet space to predict new tasks

input: pre-trained latnet space, new tasks

output: evaluation results of the new tasks saved in csv file


'''

import os
ml_code_path='/home/leilapirhaji/mz_embed_engine/ml'
os.chdir(ml_code_path)

import pandas as pd
import importlib
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


from get_pretrain_encoder import get_pretrain_encoder_from_modelID

from latent_task_predict import log_reg_multi_class, ridge_regression_predict



def evalute_pretrain_latent_extra_task(model_id_list, input_data_location, pretrain_save_dir, task_list_cat, task_list_num):

    # Initialize an empty list to collect all results across models
    all_results = []

    for model_id in model_id_list:

        # getting the latent space for the model
        (encoder, Z_all, Z_train, Z_val, Z_test, y_data_all, y_data_train, y_data_val, y_data_test)=get_pretrain_encoder_from_modelID(model_id, input_data_location, pretrain_save_dir, ml_code_path)

        print (Z_all.shape, Z_train.shape, Z_val.shape, Z_test.shape, y_data_all.shape, y_data_train.shape, y_data_val.shape, y_data_test.shape)

        # Now use the latnet space to predict the tasks
        
        model_results = {'Model ID': model_id}

        # Predict the categorical tasks

        for task in task_list_cat:

            # Drop rows with NaN values in the target columns
            valid_train_indices = y_data_train[task].dropna().index
            valid_val_indices = y_data_val[task].dropna().index
            valid_test_indices = y_data_test[task].dropna().index

            # Filter y and Z datasets to remove NaNs
            y_train = y_data_train.loc[valid_train_indices, task]
            y_val = y_data_val.loc[valid_val_indices, task]
            y_test = y_data_test.loc[valid_test_indices, task]

            Z_train_filtered = Z_train.loc[valid_train_indices]
            Z_val_filtered = Z_val.loc[valid_val_indices]
            Z_test_filtered = Z_test.loc[valid_test_indices]

            # Encode labels
            y_train = label_encoder.fit_transform(y_train)
            y_val = label_encoder.transform(y_val)
            y_test = label_encoder.transform(y_test)

            # Train and evaluate the model
            [best_model, val_accuracy, test_accuracy] = log_reg_multi_class(Z_train_filtered, y_train, Z_val_filtered, y_val, Z_test_filtered, y_test)


            # Store the results in the dictionary
            model_results[f'{task} Val Accuracy'] = val_accuracy
            model_results[f'{task} Test Accuracy'] = test_accuracy

            print(f'{task} Val Accuracy: {val_accuracy:.4f}')
            print(f'{task} Test Accuracy: {test_accuracy:.4f}')


        
        #now evaluting numercal task predictions
        for task in task_list_num:

            y_train = y_data_train[task]
            non_nan_indices = y_train.dropna().index
            y_train = y_train.dropna()
            z_train_task = Z_train.loc[non_nan_indices]

            y_val = y_data_val[task]
            non_nan_indices = y_val.dropna().index
            y_val = y_val.dropna()
            Z_val_task = Z_val.loc[non_nan_indices]

            y_test = y_data_test[task]
            non_nan_indices = y_test.dropna().index
            y_test = y_test.dropna()
            Z_test_task = Z_test.loc[non_nan_indices]


            [best_model, val_mse, test_mse, test_r2, best_alpha]=ridge_regression_predict(z_train_task, y_train, Z_val_task, y_val, Z_test_task, y_test)

            # Store the results in the dictionary
            model_results[f'{task} Val MSE'] = val_mse
            model_results[f'{task} Test MSE'] = test_mse
            model_results[f'{task} Test R2'] = test_r2

            print(f'{task} Val MSE : {val_mse:.4f}')
            print(f'{task} Test MSE : {test_mse:.4f}')
            print(f'{task} Test R2 : {test_r2:.4f}')



        # Append the model results to the list of all results
        all_results.append(model_results)



    # Convert the all_results list to a Pandas DataFrame
    final_results_df = pd.DataFrame(all_results)

    # Save the DataFrame to a CSV file
    final_results_df.to_csv(f'{pretrain_save_dir}/final_pretrain_latent_results.csv', index=False)





def main():

    #input data
    model_id_list=['RCC-7723']
    input_data_location='/home/leilapirhaji/PROCESSED_DATA_2'
    pretrain_save_dir='/home/leilapirhaji/pretrained_models' 


    #tasks to predict using encoder
    task_list_cat=[ 'Study ID', 'is Female', 'is Pediatric', 'Cohort Label v0','Smoking Status', 'Cancer Risk' ]

    task_list_num=[ 'BMI', 'Age' ]
    
    evalute_pretrain_latent_extra_task(model_id_list, input_data_location, pretrain_save_dir, task_list_cat, task_list_num)




if __name__ == '__main__':
    main()






            