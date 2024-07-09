# useful for debug purposes and general testing, makes kwargs with a VAE encoder using some default values
import os
from prep_study import make_kwargs
from setup3 import setup_neptune_run
from utils_neptune import get_latest_dataset
from sklearn.linear_model import LogisticRegression


homedir = os.path.expanduser("~")
data_dir = f'{homedir}/PRETRAIN_DATA'
os.makedirs(data_dir, exist_ok=True)
# data_dir = get_latest_dataset(data_dir=data_dir)

# encoder_kind = 'MA_Encoder_to_FF_Decoder'
encoder_kind = 'VAE'
# encoder_kind = 'metabFoundation'
# kwargs = make_kwargs(encoder_kind=encoder_kind,choose_from_distribution=False)
kwargs = make_kwargs(encoder_kind=encoder_kind,choose_from_distribution=False)

kwargs['run_evaluation'] = True
# kwargs['eval_kwargs'] = {
#     'sklearn_models': {
#         'Adversary Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
#         # 'Adversary KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'),
#     }
# }

print(kwargs)


kwargs['y_head_cols'] = ['Sex','Cohort Label v0', 'Age']
kwargs['y_adv_cols'] = ['Study ID']
kwargs['X_filename']  = 'X_Finetune'
kwargs['y_filename']  = 'y_Finetune'
kwargs['train_name'] = 'Discovery_Train'
kwargs['eval_name']  = 'Discovery_Train2'
kwargs['head_kwargs_list'] = []
kwargs['head_kwargs_dict'] = {}
kwargs['adv_kwargs_list'] = []
kwargs['adv_kwargs_dict'] = {}
kwargs['train_kwargs']['train_name'] = 'Discovery_Train'
kwargs['train_kwargs']['encoder_weight'] = 1
kwargs['train_kwargs']['head_weight'] = 1
# kwargs['head_kwargs_list'] = [
#                         {
#                         'kind': 'MultiClass',
#                         'name': 'Cohort Label',
#                         'y_idx': 1,
#                         'weight': 1.0,
#                         'hidden_size': 4,
#                         'num_hidden_layers': 1,
#                         'dropout_rate': 0,
#                         'activation': 'leakyrelu',
#                         'use_batch_norm': False,
#                         'num_classes': 4,
#                     },
#                     {
#                         'kind': 'Binary',
#                         'name': 'Sex',
#                         'y_idx': 0,
#                         'weight': 2.0,
#                         'hidden_size': 4,
#                         'num_hidden_layers': 1,
#                         'dropout_rate': 0,
#                         'activation': 'leakyrelu',
#                         'use_batch_norm': False,
#                         'num_classes': 2,
#                     },
#                     {
#                         'kind': 'Regression',
#                         'name': 'Age',
#                         'y_idx': 2,
#                         'weight': 1.0,
#                         'hidden_size': 4,
#                         'num_hidden_layers': 1,
#                         'dropout_rate': 0,
#                         'activation': 'leakyrelu',
#                         'use_batch_norm': False,
#                     }
# ]

kwargs['head_kwargs_dict'] = {
                # 'MultiClass_Cohort Label': 
                    #         {
                    #     'kind': 'MultiClass',
                    #     'name': 'Cohort Label',
                    #     'y_idx': 1,
                    #     'weight': 1.0,
                    #     'hidden_size': 4,
                    #     'num_hidden_layers': 1,
                    #     'dropout_rate': 0,
                    #     'activation': 'leakyrelu',
                    #     'use_batch_norm': False,
                    #     'num_classes': 4,
                    # },
                    'Binary_Sex':
                    {
                        'kind': 'Binary',
                        'name': 'Sex',
                        'y_idx': 0,
                        'weight': 2.0,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
                        'num_classes': 2,
                    },
                    'Regression_Age':
                    {
                        'kind': 'Regression',
                        'name': 'Age',
                        'y_idx': 2,
                        'weight': 1.0,
                        'hidden_size': 4,
                        'num_hidden_layers': 1,
                        'dropout_rate': 0,
                        'activation': 'leakyrelu',
                        'use_batch_norm': False,
                    }
}

data_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination/alignment_RCC_2024_Feb_27/July_09_Data2/formatted_data'

run_id = setup_neptune_run(data_dir,
                           setup_id='pretrain',
                           neptune_mode='async',
                        #    neptune_mode='debug',
                           yes_logging = True,
                           tags=['debug'],
                           # load_model_from_run_id=with_id,
                           **kwargs)



# setup_id = 'pretrain'
# # with_id = 'RCC-3132'
# run_id = setup_neptune_run(data_dir,
#                            setup_id=setup_id,
#                            neptune_mode='async',
#                         #    neptune_mode='debug',
#                            yes_logging = True,
#                            tags=['debug'],
#                            # load_model_from_run_id=with_id,
#                            **kwargs)
# print(run_id)


# kwargs['encoder_kwargs']['decoder_embed_dim'] = 16
# kwargs['encoder_kwargs']['latent_size'] = 32
# run_id = setup_neptune_run(data_dir,
#                            setup_id=setup_id,
#                            neptune_mode='async',
#                         #    neptune_mode='debug',
#                            yes_logging = True,
#                            tags=['debug'],
#                            # load_model_from_run_id=with_id,
#                            **kwargs)
# print(run_id)

# RCC-2893