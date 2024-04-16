# useful for debug purposes and general testing, makes kwargs with a VAE encoder using some default values

from prep_study import make_kwargs
from setup2 import setup_neptune_run
from utils_neptune import get_latest_dataset
from sklearn.linear_model import LogisticRegression


data_dir = get_latest_dataset()

encoder_kind = 'VAE'
kwargs = make_kwargs(encoder_kind=encoder_kind,choose_from_distribution=False)
kwargs['run_evaluation'] = True
kwargs['eval_kwargs'] = {
    'sklearn_models': {
        'Adversary Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
        # 'Adversary KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'),
    }
}

setup_id = 'pretrain'
run_id = setup_neptune_run(data_dir,
                           setup_id=setup_id,
                           neptune_mode='async',
                           yes_logging = True,
                           tags=['debug'],
                           **kwargs)
print(run_id)

# RCC-2893