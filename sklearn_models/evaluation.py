

import pandas as pd
import numpy as np
import os


def create_sklearn_model_summary(model_output_dir):
    """
    Create a summary of the model performance from the output files of the sklearn model
    each metric in the summary is a weighted average, weights given by the validation AUC scores
    #TODO: weights should be some combination of validation AUC and Train AUC
    """

    output_files = os.listdir(model_output_dir)
    output_files = [x for x in output_files if x.endswith('.csv')]
    output_files = [x for x in output_files if 'summary' in x]

    # summary_df = pd.DataFrame(columns=['Model','train_AUC','val_AUC','test_AUC','val_AUC_std','test_AUC_std'])
    summary_list = []

    for output_file in output_files:
        # model_name = output_file.split('_summary_')[0]
        df = pd.read_csv(os.path.join(model_output_dir, output_file),index_col=0)

        model_name = df['Model'].iloc[0]
        val_AUC = np.average(df['val_scores_mean'], weights=df['val_scores_mean'])
        train_AUC = np.average(df['train_scores_mean'], weights=df['val_scores_mean'])
        test_AUC = np.average(df['test_scores_mean'], weights=df['val_scores_mean'])

        val_AUC_std = np.sqrt( (np.average(df['val_scores_std'], weights=df['val_scores_mean']))**2 
            + np.var(df['val_scores_mean']))
        
        test_AUC_std = np.sqrt(np.average(df['test_scores_std'], weights=df['val_scores_mean']) ** 2 
            + np.var(df['test_scores_mean']))

        summary_list.append({'Model':model_name,'train_AUC':train_AUC,'val_AUC':val_AUC,'test_AUC':test_AUC,'val_AUC_std':val_AUC_std,'test_AUC_std':test_AUC_std})

    summary_df = pd.DataFrame(summary_list)    
    summary_df.set_index('Model',inplace=True)
    # Round to 3 decimal places
    # summary_df = summary_df*100
    summary_df = summary_df.round(3)

    # sort the index alphabetically
    summary_df.sort_index(inplace=True)

    # save to excel file
    summary_df.to_excel(os.path.join(model_output_dir,'summary.xlsx'))
    return summary_df