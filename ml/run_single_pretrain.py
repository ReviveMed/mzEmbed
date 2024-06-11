# useful for debug purposes and general testing, makes kwargs with a VAE encoder using some default values
import os
from prep_study import make_kwargs
from setup3 import setup_neptune_run
from utils_neptune import get_latest_dataset
from sklearn.linear_model import LogisticRegression


homedir = os.path.expanduser("~")
data_dir = f'{homedir}/PRETRAIN_DATA'
os.makedirs(data_dir, exist_ok=True)
data_dir = get_latest_dataset(data_dir=data_dir)

encoder_kind = 'MA_Encoder_to_FF_Decoder'
# encoder_kind = 'VAE'
kwargs = make_kwargs(encoder_kind=encoder_kind,choose_from_distribution=False)
kwargs['run_evaluation'] = True
# kwargs['eval_kwargs'] = {
#     'sklearn_models': {
#         'Adversary Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
#         # 'Adversary KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'),
#     }
# }

print(kwargs)

setup_id = 'pretrain'
run_id = setup_neptune_run(data_dir,
                           setup_id=setup_id,
                           neptune_mode='async',
                        #    neptune_mode='debug',
                           yes_logging = True,
                           tags=['debug'],
                           **kwargs)
print(run_id)

# RCC-2893