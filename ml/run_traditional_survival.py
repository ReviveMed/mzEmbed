import pandas as pd
import numpy as np
from sksurv.svm import FastSurvivalSVM
from sklearn.feature_selection import RFE, RFECV
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
import os
import copy
import tqdm
import itertools


import neptune
from neptune.utils import stringify_unsupported

from utils_neptune import get_latest_dataset

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

PROJECT_ID = 'revivemed/Survival-RCC'




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


def clean_param(param):
    if isinstance(param,list):
        return param[0]
    else:
        return param


def survival_grid_search(base_model, param_grid, data_dict, eval_times=None,base_model_name=None,use_neptune=True):
    """
    Function for grid search of survival models using the sklearn api. There is some hardcoding of objective function that will need revising.
    Args:
        base_model: sklearn model object
        params: dict of parameter grid
        data_dict: dictionary of data sets to use (must have at least "train")
        eval_times: list of times to evaluate (if None, uses the 20th to 80th percentile of survival data)
    Returns:
        df of full model stats
    """
    base_model = copy.deepcopy(base_model)
    if base_model_name is None:
        base_model_name = str(base_model).split("(")[0]

    if eval_times is None:
        pct_lim = np.percentile([i[1] for i in data_dict["train"]["y"]], [20, 80])
        eval_times = [i for i in np.arange(pct_lim[0], pct_lim[1], 1)]

    param_combs = list(itertools.product(*param_grid.values()))
    grid_search = [dict(zip(param_grid.keys(), values)) for values in param_combs]
    hp_summary = {}
    for i, gs in tqdm.tqdm(enumerate(grid_search)):
        model = base_model.set_params(**gs)
        if use_neptune:
            run = neptune.init_run(
                project=PROJECT_ID,
                api_token=NEPTUNE_API_TOKEN,
            ) 

            gs_clean = {k: clean_param(v) for k,v in gs.items()}
            run['parameters']=stringify_unsupported(gs_clean)
            run['model']=base_model_name
            if 'os_col' in data_dict['train']:
                run['datasets/os_col']=data_dict['train']['os_col']
            if 'event_col' in data_dict['train']:
                run['datasets/event_col']=data_dict['train']['event_col']
            for set_name in data_dict.keys():
                if 'X_file' in data_dict[set_name]:
                    run[f'datasets/{set_name}/X'].track_files(data_dict[set_name]['X_file'])
                if 'y_file' in data_dict[set_name]:
                    run[f'datasets/{set_name}/y'].track_files(data_dict[set_name]['y_file'])
                run[f'datasets/{set_name}/size'] = len(data_dict[set_name]['y'])
        try:

            model.fit(data_dict["train"]["X"], data_dict["train"]["y"])
            # get model stats
            model_stats = [
                {(k, "c_score"): model.score(v["X"], v["y"]) for k,v in data_dict.items()},
                {(k, "ibs"): get_ibrier(model, data_dict["train"]["y"], v["X"], v["y"], eval_times) for k,v in data_dict.items()},
                {(k, "dauc"): {(k, "dauc_"+str(t)): cumulative_dynamic_auc(data_dict["train"]["y"], v["y"], model.predict(v["X"]), times=[t])[1] for t in eval_times + [eval_times[0:2]]} for k,v in data_dict.items()} 
            ]

            if use_neptune:
                # run['metrics/eval_times'] = eval_times
                for k in data_dict.keys():
                    run['metrics/'+k+'/c_score'] = stringify_unsupported(model_stats[0][k, "c_score"])
                    run['metrics/'+k+'/ibs'] = stringify_unsupported(model_stats[1][k, "ibs"])
                    # duac_array = []
                    # for t in eval_times + [eval_times[0:2]]:
                    #     duac_array.append(model_stats[2][k, "dauc_"+str(t)])
                    # run['metrics/'+k+'/dauc'] = duac_array

        except:
            model_stats = [
                {(k, "c_score"): np.nan for k,v in data_dict.items()},
                {(k, "ibs"): np.nan for k,v in data_dict.items()},
                {(k, "dauc"): {(k, "dauc_"+str(t)): np.nan for t in eval_times + [eval_times[0:2]]} for k,v in data_dict.items()} 
            ]
        i_df = pd.concat(
                [pd.DataFrame(model_stats[0], [0]),
                pd.DataFrame(model_stats[1], [0]),
                pd.concat([pd.DataFrame(iv, [0]) for iv in model_stats[2].values()], axis=1)], axis=1)

        i_df['params'] = [gs]
        if hasattr(model, 'coef_'):
            i_df['n_nonzero'] = np.sum(model.coef_!=0)
            if use_neptune:
                run['n_nonzero'] = i_df['n_nonzero']
        else:
            i_df['n_nonzero'] = np.nan
        
        if use_neptune:
            run.stop()

        hp_summary[i] = i_df

    hp_df = pd.concat(hp_summary)
    # # asdf.to_excel(f"{output_dir}/{str(sp)+mn}_hp_summary.xlsx")

    return hp_df

def get_ibrier(cph_trained, input_y, apply_x, apply_y, eval_times=None):
    """
    Calculate integrated Brier score for a trained coxph model at various timepoints.
    Args:
        cph_trained: trained CoxNet model (Must be fit with fit_baseline_model=True)
        input_y: survival data for the training set
        apply_x: new data to apply the model to
        apply_y: survival data for the new data
        eval_times: list of times to evaluate
    """
    try:
        # if eval_times is None:
        #     pct_lim = np.percentile([i[1] for i in data_dict["train"]["y"]], [20, 80])
        #     eval_times = [i for i in np.arange(pct_lim[0], pct_lim[1], 1)]

        survs = cph_trained.predict_survival_function(apply_x)
        # lower, upper = np.percentile([i[1] for i in apply_y], [10, 90])
        # times = np.arange(lower, upper + 1)
        times = eval_times
        preds = np.asarray([[fn(t) for t in times] for fn in survs])
        ibs = integrated_brier_score(input_y, apply_y, preds, times)
    except:
        ibs = np.nan

    return  ibs


if __name__ == '__main__':


    # get the home directory
    home_dir = os.path.expanduser("~")
    data_dir = f'{home_dir}/DATA'
    output_dir = f'{home_dir}/OUTPUT'

    # %%
    ## Get the latest dataset
    data_dir = get_latest_dataset(data_dir=data_dir,project=PROJECT_ID)

    # %%
    # Import the data
    # os_col = 'OS'
    for os_col in ['OS','NIVO OS','EVER OS']:

        event_col = 'OS_Event'

        output_dir = f'{output_dir}/survival_results/{os_col}'
        os.makedirs(output_dir, exist_ok=True)

        data_dict = create_data_dict(data_dir,'train',os_col,event_col)
        data_dict = create_data_dict(data_dir,'val',os_col,event_col,data_dict)
        data_dict = create_data_dict(data_dir,'test',os_col,event_col,data_dict)

        # X_train = pd.read_csv(f'{data_dir}/X_finetune_train.csv', index_col=0)
        # X_val = pd.read_csv(f'{data_dir}/X_finetune_val.csv', index_col=0)
        # X_test = pd.read_csv(f'{data_dir}/X_finetune_test.csv', index_col=0)

        # y_train = pd.read_csv(f'{data_dir}/y_finetune_train.csv', index_col=0)
        # y_val = pd.read_csv(f'{data_dir}/y_finetune_val.csv', index_col=0)
        # y_test = pd.read_csv(f'{data_dir}/y_finetune_test.csv', index_col=0)

        # data_dict = {
        #     "train": {"X": X_train, "y": prep_surv_y(y_train[os_col], y_train[event_col])},
        #     "val": {"X": X_val, "y": prep_surv_y(y_val[os_col], y_val[event_col])},
        #     "test": {"X": X_test, "y": prep_surv_y(y_test[os_col], y_test[event_col])}
        # }


        # %%
        # running L1-Cox
        l1_cox_param_grid = {
            "l1_ratio": [1],
            "fit_baseline_model": [True],
            "alphas": [[i] for i in 10.0 ** np.linspace(-4, 4, 100)]
        }

        gs_model = CoxnetSurvivalAnalysis()
        grid_res_df = survival_grid_search(gs_model, l1_cox_param_grid, data_dict)
        grid_res_df.to_csv(f'{output_dir}/l1_cox_grid_search_results.csv')


        # %% running random survival forest
        random_forest_param_grid = {
                        'n_estimators': [100, 200, 300, 400],
                        'max_depth': [3, 5, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4, 8],
                        'bootstrap': [True, False],
                        'n_jobs': [-1],
                        'random_state': [8]
        }

        gs_model = RandomSurvivalForest()

        grid_res_df = survival_grid_search(gs_model, random_forest_param_grid, data_dict)
        grid_res_df.to_csv(f'{output_dir}/random_forest_grid_search_results.csv')   


        # %% running survival SVM


        svm_param_grid = {
                        "rank_ratio": [0.1, 0.2, 0.5, 1],
                        "alpha": 2.0 ** np.arange(-12, 13, 2),
                        "fit_intercept": [True, False],
                        'random_state': [8]
        }
        gs_model = FastSurvivalSVM()

        grid_res_df = survival_grid_search(gs_model, svm_param_grid, data_dict)
        grid_res_df.to_csv(f'{output_dir}/svm_grid_search_results.csv')


        # %% GBSA

        gs_model = GradientBoostingSurvivalAnalysis()

        gbsa_param_grid = {
            'learning_rate': [0.01, 0.1, 1],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'random_state': [8]
        }

        grid_res_df = survival_grid_search(gs_model, gbsa_param_grid, data_dict)
        grid_res_df.to_csv(f'{output_dir}/gbsa_grid_search_results.csv')


        # %% L2-Cox

        l2_cph_param_grid = {
            "alpha": 10.0 ** np.linspace(-4, 4, 100)
        }

        gs_model = CoxPHSurvivalAnalysis()

        grid_res_df = survival_grid_search(gs_model, l2_cph_param_grid, data_dict)
        grid_res_df.to_csv(f'{output_dir}/l2_cox_grid_search_results.csv')