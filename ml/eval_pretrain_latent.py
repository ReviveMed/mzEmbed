
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
from latent_task_predict_pretrain import log_reg_multi_class, ridge_regression_predict




def evalute_pretrain_latent_extra_task(model_id, input_data_location, pretrain_save_dir, task_list_cat, task_list_num):

    print (f'model_id is {model_id}')


    # getting the latent space for the model
    (encoder, Z_all, Z_train, Z_val, Z_test, y_data_all, y_data_train, y_data_val, y_data_test)=get_pretrain_encoder_from_modelID(model_id, input_data_location, pretrain_save_dir, ml_code_path)
    

    print (Z_all.shape, Z_train.shape, Z_val.shape, Z_test.shape, y_data_all.shape, y_data_train.shape, y_data_val.shape, y_data_test.shape)

    # evaluating the avg latent space for the reconstruction loss

    # Now use the latnet space to predict the tasks
    model_results = {'Model ID': model_id}

    # Predict the categorical tasks

    for task in task_list_cat:

        (val_accuracy, test_accuracy) = log_reg_multi_class(task, Z_train, y_data_train, Z_val, y_data_val, Z_test, y_data_test)

        # Store the results in the dictionary
        model_results[f'{task} Val Accuracy'] = val_accuracy
        model_results[f'{task} Test Accuracy'] = test_accuracy

        print(f'{task} Val Accuracy: {val_accuracy:.4f}')
        print(f'{task} Test Accuracy: {test_accuracy:.4f}')

    
    #now evaluting numercal task predictions
    for task in task_list_num:

        (val_mse, val_mae, val_r2, test_mse, test_mae, test_r2)= ridge_regression_predict(task, Z_train, y_data_train, Z_val, y_data_val, Z_test, y_data_test)

        # Store the results in the dictionary
        # model_results[f'{task} Val MSE'] = val_mse
        model_results[f'{task} Val MAE'] = val_mae
        # model_results[f'{task} Val R2'] = val_r2
        # model_results[f'{task} Test MSE'] = test_mse
        model_results[f'{task} Test MAE'] = test_mae
        # model_results[f'{task} Test R2'] = test_r2

        print(f'{task} Val MAE : {val_mae:.4f}')
        print(f'{task} Test MAE : {test_mae:.4f}')
        


    return model_results





def main():

    #input data
    model_id_list=['RCC-10290', 'RCC-11291', 'RCC-11291', 'RCC-11472', 'RCC-11517', 'RCC-11174', 'RCC-7620', 'RCC-10860', 'RCC-10737', 'RCC-11381', 'RCC-10830', 'RCC-10256', 'RCC-11222', 'RCC-11511', 'RCC-10662', 'RCC-11037', 'RCC-10473', 'RCC-10121', 'RCC-7426', 'RCC-10192', 'RCC-9566', 'RCC-10343']

    input_data_location='/home/leilapirhaji/PROCESSED_DATA_2'
    pretrain_save_dir='/home/leilapirhaji/pretrained_models' 


    #tasks to predict using encoder
    task_list_cat=[ 'Study ID', 'is Female', 'is Pediatric', 'Cohort Label v0','Smoking Status', 'Cancer Risk' ]

    task_list_num=[ 'BMI', 'Age' ]

    # Initialize an empty list to collect all results across models
    all_results = []
    

    for model_id in model_id_list:

        model_results = evalute_pretrain_latent_extra_task(model_id_list, input_data_location, pretrain_save_dir, task_list_cat, task_list_num)


    # Append the model results to the list of all results
    all_results.append(model_results)

    # Convert the all_results list to a Pandas DataFrame
    final_results_df = pd.DataFrame(all_results)

    # Save the DataFrame to a CSV file
    final_results_df.to_csv(f'{pretrain_save_dir}/final_pretrain_latent_results.csv', index=False)



if __name__ == '__main__':
    main()






            