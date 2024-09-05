
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



#importing fundtion to get encoder info and perfrom tasks 
from latent_task_predict_finetune import log_reg_multi_class, ridge_regression_predict, cox_proportional_hazards

from get_finetune_encoder import get_finetune_encoder_from_modelID, get_input_data


#importing Jonha's funtions 
from models import get_model, Binary_Head, Dummy_Head, MultiClass_Head, MultiHead, Regression_Head, Cox_Head, get_encoder
from viz import generate_latent_space, generate_umap_embedding, generate_pca_embedding







def evalute_pretrain_latent_categorical_tasks(X_data_train, Z_train_real, Z_train_rand, X_data_val, Z_val_real, Z_val_rand, y_data_train, y_data_val, task):

    # Drop rows with NaN values in the target columns
    valid_train_indices = y_data_train[task].dropna().index
    valid_val_indices = y_data_val[task].dropna().index

    # Filter y to remove NaNs
    y_train = y_data_train.loc[valid_train_indices, task]
    y_val = y_data_val.loc[valid_val_indices, task]

    # Encode Y labels
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)


    #full input data
    X_train_filtered = X_data_train.loc[valid_train_indices]
    X_val_filtered = X_data_val.loc[valid_val_indices]

    #latent space - fine tuned initiation
    Z_train_filtered_real = Z_train_real.loc[valid_train_indices]
    Z_val_filtered_real = Z_val_real.loc[valid_val_indices]

    #latent space - rand initiation
    Z_train_filtered_rand = Z_train_rand.loc[valid_train_indices]
    Z_val_filtered_rand = Z_val_rand.loc[valid_val_indices]


    # Train and evaluate the model - latent space
    [best_model, val_accuracy_X] = log_reg_multi_class(X_train_filtered, y_train, X_val_filtered, y_val,)
    [best_model, val_accuracy_Z_real] = log_reg_multi_class(Z_train_filtered_real, y_train, Z_val_filtered_real, y_val,)
    [best_model, val_accuracy_Z_rand] = log_reg_multi_class(Z_train_filtered_rand, y_train, Z_val_filtered_rand, y_val,)

    return (val_accuracy_X, val_accuracy_Z_real, val_accuracy_Z_rand)


    


def evalute_pretrain_latent_survival_tasks( X_data_train, X_data_val, Z_train_real, Z_val_real,Z_train_rand, Z_val_rand, y_data_train, y_data_val, survival_task, survival_event_task):
    
    Y_train_OS=y_data_train[survival_task]
    Y_train_event=y_data_train[survival_event_task]

    Y_val_OS=y_data_val[survival_task]
    Y_val_event=y_data_val[survival_event_task]


    #measuring C index for input data
    (best_c_index_X, best_params) = cox_proportional_hazards(X_data_train, Y_train_OS, Y_train_event, X_data_val, Y_val_OS, Y_val_event)

    #measuring C index for latent space - fine tuned initiation
    (best_c_index_Z_real, best_params) = cox_proportional_hazards(Z_train_real, Y_train_OS, Y_train_event, Z_val_real, Y_val_OS, Y_val_event)

    #measuring C index for latent space - rand initiation
    (best_c_index_Z_rand, best_params) = cox_proportional_hazards(Z_train_rand, Y_train_OS, Y_train_event, Z_val_rand, Y_val_OS, Y_val_event)

    return (best_c_index_X, best_c_index_Z_real, best_c_index_Z_rand)





def main():

    #input data
    input_data_location='/home/leilapirhaji/PROCESSED_DATA_2'
    finetune_save_dir='/home/leilapirhaji/finetune_models' 


    #tasks to predict using encoder
    task_list_cat=['Benefit BINARY', 'Nivo Benefit BINARY', 'MSKCC BINARY', 'IMDC BINARY', 'Benefit ORDINAL', 'MSKCC ORDINAL', 'IMDC ORDINAL', 'ORR', 'Benefit', 'IMDC', 'MSKCC', 'Prior_2' ]

    #survival tasks
    task_list_survival=[ 'OS', 'OS_Event', 'PFS', 'PFS_Event' ]
    
    #pre-train model info  - 1
    model_id_pretrain_1='RCC-10290'

    #model info
    model_id_real_1='RCC-35857'
    model_neptune_path_real_1='fine-tune-optuna-RCC-10290-recon-real-modified-loss-Aug29'

    model_id_rand_1='RCC-35899'
    model_neptune_path_rand_1='fine-tune-optuna-RCC-10290-recon-random-modified-loss-Aug29'

    #pre-train model info - 2
    model_id_pretrain_2='RCC-11381'

    #model info
    model_id_real_2='RCC-35663'
    model_neptune_path_real_2='fine-tune-optuna-RCC-11381-recon-real-modified-loss-Aug29'

    model_id_rand_2='RCC-35692'
    model_neptune_path_rand_2='fine-tune-optuna-RCC-11381-recon-random-modified-loss-Aug29'



    # Dictionary to save the finetune model info
    finetune_dic = {
        model_id_pretrain_1: {
            'pretrain_model_id': model_id_pretrain_1,
            'model_id_real': model_id_real_1,
            'model_neptune_path_real': model_neptune_path_real_1,
            'model_id_rand': model_id_rand_1,
            'model_neptune_path_rand': model_neptune_path_rand_1
        },
        model_id_pretrain_2: {
            'pretrain_model_id': model_id_pretrain_2,
            'model_id_real': model_id_real_2,
            'model_neptune_path_real': model_neptune_path_real_2,
            'model_id_rand': model_id_rand_2,
            'model_neptune_path_rand': model_neptune_path_rand_2
        }
    }


    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=['Pretrained Model_ID', 'Task', 'Val_Accuracy_X',       
                                'Val_Accuracy_Z_real', 
                                'Val_Accuracy_Z_rand', 'Best_C_Index_X', 
                                'Best_C_Index_Z_real', 'Best_C_Index_Z_rand'])


    #looping through the pre-train model info
    # Loop through the pre-train model info
    for pretrain_id, model_info in finetune_dic.items():

        print (f'evaluating pre-train model: {pretrain_id}')
        model_id_real = model_info['model_id_real']
        model_neptune_path_real = model_info['model_neptune_path_real']
        model_id_rand = model_info['model_id_rand']
        model_neptune_path_rand = model_info['model_neptune_path_rand']

        # Output locations
        output_path_real = f'{finetune_save_dir}/{model_id_real}'
        os.makedirs(output_path_real, exist_ok=True)

        output_path_rand = f'{finetune_save_dir}/{model_id_rand}'
        os.makedirs(output_path_rand, exist_ok=True)


        #get real input data
        (X_data_train, y_data_train, X_data_val, y_data_val)=get_input_data(input_data_location)

        print (f'getting the encoder for: {model_id_rand}')
        #latent space with rand initization
        (encoder, Z_train_rand, Z_val_rand, y_data_train, y_data_val)=get_finetune_encoder_from_modelID(model_id_rand, input_data_location, output_path_rand, ml_code_path, model_neptune_path_rand )


        #latent space with fine-tuned initization
        print (f'getting the encoder for: {model_id_real}')
        (encoder, Z_train_real, Z_val_real, y_data_train, y_data_val)=get_finetune_encoder_from_modelID(model_id_real, input_data_location, output_path_rand, ml_code_path, model_neptune_path_real)


        # now using the encoder to predict the categorical tasks
        # Evaluate categorical tasks
        for task in task_list_cat:

            print (f'evaluating task: {task}')
            val_accuracy_X, val_accuracy_Z_real, val_accuracy_Z_rand = evalute_pretrain_latent_categorical_tasks(
                X_data_train, Z_train_real, Z_train_rand, X_data_val, Z_val_real, Z_val_rand, y_data_train, y_data_val, task)

            print (f'val_accuracy_X: {val_accuracy_X}, val_accuracy_Z_real: {val_accuracy_Z_real}, val_accuracy_Z_rand: {val_accuracy_Z_rand}')

            # Save results in DataFrame
            results_df = results_df.append({
                'Model_ID': pretrain_id,
                'Task': task,
                'Val_Accuracy_X': val_accuracy_X,
                'Val_Accuracy_Z_real': val_accuracy_Z_real,
                'Val_Accuracy_Z_rand': val_accuracy_Z_rand,
                'Best_C_Index_X': None,
                'Best_C_Index_Z_real': None,
                'Best_C_Index_Z_rand': None
            }, ignore_index=True)


        # Evaluate survival tasks
        for survival_task, survival_event_task in zip(task_list_survival[::2], task_list_survival[1::2]):
            
            print (f'evaluating survival task: {survival_task}')

            best_c_index_X, best_c_index_Z_real, best_c_index_Z_rand = evalute_pretrain_latent_survival_tasks(
                X_data_train, X_data_val, Z_train_real, Z_val_real, Z_train_rand, Z_val_rand, y_data_train, y_data_val, survival_task, survival_event_task)

            print (f'best_c_index_X: {best_c_index_X}, best_c_index_Z_real: {best_c_index_Z_real}, best_c_index_Z_rand: {best_c_index_Z_rand}')

            # Save results in DataFrame
            results_df = results_df.append({
                'Model_ID': pretrain_id,
                'Task': survival_task,
                'Val_Accuracy_X': None,
                'Val_Accuracy_Z_real': None,
                'Val_Accuracy_Z_rand': None,
                'Best_C_Index_X': best_c_index_X,
                'Best_C_Index_Z_real': best_c_index_Z_real,
                'Best_C_Index_Z_rand': best_c_index_Z_rand
            }, ignore_index=True)
   
    # Save the results DataFrame to a CSV file for future reference
    results_df.to_csv(f'{finetune_save_dir}/finetune_evaluation_results.csv', index=False)

    print("Evaluation completed. Results saved to DataFrame and CSV.")







if __name__ == '__main__':
    main()






            