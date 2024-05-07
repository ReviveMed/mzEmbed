# Get the optima CPH thresholds usng Matt's code
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from lifelines import KaplanMeierFitter, CoxPHFitter
import tqdm
from models import create_compound_model_from_info, create_pytorch_model_from_info, MultiHead
import json
from misc import download_data_dir
import os
from sksurv.linear_model import CoxPHSurvivalAnalysis


# %%
# %%
# generalized function to run grid search with custom objective
def prep_surv_y(survival_column, event_column):
    """
    Function for formatting survival data as input to sksurv models.
    Args:
        survival_column: list of survival times
        event_column: list of event indicators (0 for censored, 1 for event)
    """
    surv_y = np.array([(event_column.iloc[i], survival_column.iloc[i]) for i in range(len(survival_column))], dtype=[('status', bool), ('time', float)])
    # surv_y = np.array([(event_column[i], survival_column[i]) for i in range(len(survival_column))], dtype=[('status', bool), ('time', float)])
    return surv_y

def create_data_dict(data_dir,set_name,os_col,event_col,data_dict=None):
    if data_dict is None:
        data_dict = {}
    X_path = f'{data_dir}/X_finetune_{set_name}.csv'
    y_path = f'{data_dir}/y_finetune_{set_name}.csv'
    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    not_nan = ~y[os_col].isna()
    data_dict[set_name] = {"X": X.loc[not_nan],
                           "y": prep_surv_y(y.loc[not_nan,os_col], y.loc[not_nan,event_col]),
                           'os_col':os_col,
                           'event_col':event_col,
                           'X_file':X_path,
                           'y_file':y_path}
    return data_dict


def get_best_3way_threshold(model_trained, new_data,min_class_size=20):
    """
    Identify the threshold that maximizes C-index between classes where p remains significant. 
    Note: this does NOT account for clinical covariates
    Further note: sksurv models with LOWER predicted survival are associated with LONGER survival
    Args:
        model_trained: trained model
        new_data: dict with new data (transformed and scaled) and survival data
    Returns:
        optimal thresholds as a tuple. 
          To apply these thresholds downstream, predicted survival of  "FAVORABLE" <= the LOWER threshold; "INTERMEDIATE" > LOWER and <= HIGHER; "POOR" > HIGHER
    """


    best_thresh = (np.nan, np.nan)

    tmp = pd.DataFrame(new_data["y"]).set_axis(['Event', 'Survival'], axis=1)

    predicted_surv = model_trained.predict(new_data["X"])

    thresh_list = np.sort(np.unique(predicted_surv))  # SUBSET FOR DEV
    # identify threshold maximizing c-index between classes where p remains significant
    thresh_summary = {}
    for th_lo in tqdm.tqdm(thresh_list):
        for th_hi in thresh_list[thresh_list>th_lo]:
            # predict classes
            pred_class = np.empty_like(predicted_surv, dtype=object)
            pred_class[predicted_surv <= th_lo] = 'FAVORABLE'
            pred_class[(predicted_surv > th_lo) & (predicted_surv <= th_hi)] = 'INTERMEDIATE'
            pred_class[predicted_surv > th_hi] = 'POOR'
            mod_d = tmp.copy()
            mod_d['pred'] = pd.Categorical(pred_class, categories=['INTERMEDIATE', 'POOR', 'FAVORABLE'], ordered=True)


            # don't bother unless we have >1 of each class
            if (sum(pred_class=='FAVORABLE')>min_class_size) & (sum(pred_class=='INTERMEDIATE')>min_class_size) & (sum(pred_class=='POOR')>min_class_size):
                thresh_summary[(th_lo, th_hi)] = {}
                try:
                    ct = CoxPHSurvivalAnalysis()
                    ct.fit(pd.get_dummies(mod_d['pred'], drop_first=True), new_data['y'])
                    coef_list = ct.coef_
                    if (coef_list[0]>0) & (coef_list[1]<0):
                        c_score = ct.score(pd.get_dummies(mod_d['pred'], drop_first=True), new_data['y'])
                        thresh_summary[(th_lo, th_hi)].update({'c_ind': c_score, "n_POOR": sum(pred_class=='POOR'), "n_INTERMEDIATE": sum(pred_class=='INTERMEDIATE'), "n_FAVORABLE": sum(pred_class=='FAVORABLE')})
                except:
                    continue


    thresh_df = pd.DataFrame(thresh_summary).T.dropna()

    if len(thresh_df)>0:
        best_thresh = thresh_df.dropna().sort_values('c_ind', ascending=False).index[0]

    return best_thresh, thresh_df


# %%


dropbox_url = 'https://www.dropbox.com/scl/fo/y36aiinfbsf6rymzj1qh4/ACjkhRE_Ig_IBwlrH4Tl_Ug?rlkey=4s5imazvcoktgk83lig8rypjw&dl=1'
download_data_dir(dropbox_url, save_dir='/app/finetune_data')

dropbox_url = 'https://www.dropbox.com/scl/fo/0zbnyynf5igdorbh6veqh/AE8PiJDZ5FB2Qbu_KiAF9W4?rlkey=km43xr7dpfssgdjcbev21ddb2&dl=1'
download_data_dir(dropbox_url, save_dir='/app/model_data')


input_dir = '/app/model_data'
data_dir = '/app/finetune_data'


data_dict =create_data_dict(data_dir,'trainval','OS','OS_Event')
data_dict =create_data_dict(data_dir,'test','OS','OS_Event',data_dict)


model_info = json.load(open(f'{input_dir}/Model_2925 info.json'))
model_state = torch.load(f'{input_dir}/Model_2925 state.pt')

model_trained = create_pytorch_model_from_info(model_info, model_state)

best_thresh, thresh_df = get_best_3way_threshold(model_trained, data_dict['trainval'], min_class_size=30)

print(best_thresh)

thresh_df.to_csv('~/thresholds.csv')