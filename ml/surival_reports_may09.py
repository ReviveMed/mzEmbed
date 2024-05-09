#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neptune
from models import create_compound_model_from_info, create_pytorch_model_from_info, MultiHead
import json
import torch
import os
from utils_neptune import check_if_path_in_struc, get_sub_struc_from_path
from collections import defaultdict

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='
# data_dir= '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_finetune_optimization/April_30_Finetune_Data'
# local_dir = os.path.expanduser('~/Desktop/saved_models')

data_dir= '/app/finetune_data'
output_dir = '/app/mz_embed_engine/output'
local_dir = os.path.expanduser('~/saved_models')
run_id = 'RCC-2925'

# %%
def generate_survival_report(desc_str,head_key,pretrained=True,data_dir=None,local_dir=None):
    if data_dir is None:
        data_dir = '/app/finetune_data'
    if local_dir is None:
        local_dir = os.path.expanduser('~/saved_models')
    if ('OS' not in head_key):
        raise ValueError('head_key must be an OS')
    
    if ('finetune' in desc_str) or ('randinit' in desc_str):
        task_id = desc_str
    else:
        if pretrained:
            task_id = desc_str + '_finetune'
        else:
            task_id = desc_str + '_randinit'

    model_dir = f'{local_dir}/{run_id}/{task_id}'
    components_dir = f'{local_dir}/{run_id}/{task_id}/components'
    os.makedirs(components_dir, exist_ok=True)
    model_files = os.listdir(components_dir)
    
    if (len(model_files) < 4):

        run = neptune.init_run(project='revivemed/RCC',
            api_token= NEPTUNE_API_TOKEN,
            with_id=run_id,
            mode="read-only")   
        run_struc= run.get_structure()

        original_kwargs=run[task_id+'/original_kwargs'].fetch()
        json.dump(original_kwargs,open(f'{components_dir}/original_kwargs.json','w'),indent=4)
        substruc = get_sub_struc_from_path(run_struc,f'{task_id}/models')
        for key in substruc.keys():
            if 'info' in key:
                run[f'{task_id}/models/{key}'].download(f'{components_dir}/{key}.json')
            elif 'state' in key:
                run[f'{task_id}/models/{key}'].download(f'{components_dir}/{key}.pt')

        run.stop()
        model_files = os.listdir(components_dir)


    encoder_info = None
    head_info = None
    encoder_state = None
    head_state = None
    original_kwargs = None

    for f in model_files:
        if ('encoder' in f):
            if 'info' in f:
                encoder_info = json.load(open(f'{components_dir}/{f}'))
            elif 'state' in f:
                encoder_state = torch.load(f'{components_dir}/{f}')
        if (head_key in f):
            if 'info' in f:
                head_info = json.load(open(f'{components_dir}/{f}'))
                # head_name = f.replace('_info.json', '')
            elif 'state' in f:
                head_state = torch.load(f'{components_dir}/{f}')
        if 'original_kwargs' in f:
            original_kwargs = json.load(open(f'{components_dir}/{f}'))
                
    if (encoder_info is not None) and (head_info is not None):            
        model = create_compound_model_from_info(encoder_info=encoder_info, 
                                                head_info= head_info,
                                                encoder_state_dict=encoder_state,
                                                head_state_dict=head_state)
        
        params = {}
        params['encoder dropout_rate'] = encoder_info['dropout_rate']
        params['head name'] = head_info['name']
        params['head layers']= head_info['architecture']['num_hidden_layers']
        if (original_kwargs is not None) and ('train_kwargs' in original_kwargs):
            params['head weight'] = head_info['weight']*original_kwargs['train_kwargs']['head_weight']
            params['num auxillary heads'] = len(original_kwargs['head_kwargs_dict']) -1
            
            if 'adversarial_head_kwargs_dict' in original_kwargs:
                params['num adversarial heads'] = len(original_kwargs['adversarial_head_kwargs_dict'])
            else:
                params['num adversarial heads'] = 0
            params['adversary weight'] = original_kwargs['train_kwargs']['adversary_weight']
            params['adversarial_start_epoch'] = original_kwargs['train_kwargs']['adversarial_start_epoch']
            if params['adversary weight'] == 0 or params['num adversarial heads'] == 0:
                params['adversary weight'] = 0
                params['num adversarial heads'] = 0
                params['adversarial_start_epoch'] = 0
            
            params['encoder weight'] = original_kwargs['train_kwargs']['encoder_weight']
            params['learning rate'] = original_kwargs['train_kwargs']['learning_rate']
            params['l1_reg_weight'] = original_kwargs['train_kwargs']['l1_reg_weight']
            params['l2_reg_weight'] = original_kwargs['train_kwargs']['l2_reg_weight']
            params['noise_factor'] = original_kwargs['train_kwargs']['noise_factor']
            params['num_epochs'] = original_kwargs['train_kwargs']['num_epochs']
            params['weight_decay'] = original_kwargs['train_kwargs']['weight_decay']

        model.save_info(model_dir, f'VAE {head_key} info.json')
        model.save_state_to_path(model_dir, f'VAE {head_key} state.pt')

        skmodel = create_pytorch_model_from_info(full_model=model)
    else:
        return None


    X_data = pd.read_csv(f'{data_dir}/X_finetune_test.csv',index_col=0)
    y_data = pd.read_csv(f'{data_dir}/y_finetune_test.csv',index_col=0)


    score_dict = {}
    score_dict['both-OS'] = skmodel.score(X_data.to_numpy(),y_data[['OS','OS_Event']].to_numpy())['Concordance Index']
    score_dict['NIVO-OS'] = skmodel.score(X_data.to_numpy(),y_data[['NIVO OS','OS_Event']].to_numpy())['Concordance Index']
    score_dict['EVER-OS'] = skmodel.score(X_data.to_numpy(),y_data[['EVER OS','OS_Event']].to_numpy())['Concordance Index']
    
    res_dict = {
        'test c-index' : score_dict,
        'params' : params
    }


    return res_dict

# %%
def save_full_report(desc_str):
    report_dict = defaultdict(dict)

    os.makedirs(f'{output_dir}/surv_report',exist_ok=True)
    for head_key in ['Cox_NIVO OS', 'Cox_EVER OS', 'Cox_OS']:
        print(head_key)
        report_dict[head_key] = generate_survival_report(desc_str,head_key,pretrained=True,data_dir=data_dir,local_dir=local_dir)

    try:
        report_df = pd.DataFrame(report_dict).T

        params_df = pd.json_normalize(report_df['params'])
        score_df = pd.json_normalize(report_df['test c-index'])
        score_df.columns = [c + ' c-index' for c in score_df.columns]

        final_df = pd.concat([params_df,score_df],axis=1)
        final_df.index = report_df.index

        final_df.to_csv(f'{output_dir}/surv_report/{desc_str}_survival_report.csv')

        return final_df
    except: 
        json.dump(report_dict,open(f'{output_dir}/surv_report/{desc_str}_survival_report.json','w'),indent=4)
        return


# %%

desc_str_list = [
            'Optimized_NIVO-OS_finetune','Optimized_NIVO-OS_randinit',
            'Optimized_EVER-OS_finetune', 'Optimized_EVER-OS_randinit',
            'Optimized_both-OS_finetune', 'Optimized_both-OS_randinit',
            'Optimized_NIVO-OS ADV EVER-OS_finetune', 'Optimized_NIVO-OS ADV EVER-OS_randinit']
for desc_str in desc_str_list:
    print(desc_str)
    # save_full_report(desc_str)


# %%


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('talk')



def generate_survival_umap(task_id,data_dir=None,local_dir=None,use_full_title=True):
    if data_dir is None:
        data_dir = '/app/finetune_data'
    if local_dir is None:
        local_dir = os.path.expanduser('~/saved_models')

    os.makedirs(f'{output_dir}/plots')

    run = neptune.init_run(project='revivemed/RCC',
        api_token= NEPTUNE_API_TOKEN,
        with_id=run_id,
        mode="read-only")   
    run_struc= run.get_structure()
    substruc = get_sub_struc_from_path(run_struc,f'{task_id}')
    embeds_dir = f'{local_dir}/{run_id}/{task_id}/embed'
    os.makedirs(embeds_dir,exist_ok=True)
    
    for key in substruc.keys():
        if 'Z_embed' in key:
            run[f'{task_id}/{key}'].download(f'{embeds_dir}/{key}.csv')
    run.stop()


    for set_name in ['Train','TrainVal','Val','Test']:

        embed_data_file = f'{embeds_dir}/Z_embed_{set_name.lower()}2.csv'
        if not os.path.exists(embed_data_file):
            continue

        df_org = pd.read_csv(embed_data_file,index_col=0)

        for hue_col in ['OS','NIVO-OS','EVER-OS']:
            for remove_no_imdc in [True,False]:
                df = df_org[(df_org[hue_col].notnull())]

                fig_name = f'{hue_col} on {set_name}'
                if remove_no_imdc:
                    df = df[(df['IMDC'].isin(['FAVORABLE','INTERMEDIATE','POOR']))]
                    fig_name += ' reqIMDC'

                    sns.scatterplot(data=df,x='UMAP1',y='UMAP2',hue='OS',style='OS_Event',style_order=[1,0],hue_norm=(0,60))
                    # place the legend outside the plot
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    if use_full_title:
                        plt.title(f'{set_name} (N={len(df)})\n{task_id}')
                    else:
                        plt.title(f'{set_name} (N={len(df)})')

                    plt.savefig(f'{output_dir}/plots/{task_id}__{fig_name}.png',bbox_inches='tight')
                    plt.close()

        return

# %%

desc_str_list = [
            'Optimized_NIVO-OS_finetune','Optimized_NIVO-OS_randinit',
            'Optimized_EVER-OS_finetune', 'Optimized_EVER-OS_randinit',
            'Optimized_both-OS_finetune', 'Optimized_both-OS_randinit',
            'Optimized_NIVO-OS ADV EVER-OS_finetune', 'Optimized_NIVO-OS ADV EVER-OS_randinit']


for desc_str in desc_str_list:
    print(desc_str)
    save_full_report(desc_str)                    