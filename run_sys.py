import os
import sys
if len(sys.argv)>1:
    n_trials = int(sys.argv[1])
else:
    n_trials = 10

if len(sys.argv)>2:
    run_id = sys.argv[2]
else:
    run_id = 'RCC-2925'


# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'NIVO-OS'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'EVER-OS'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'IMDC' 1")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'MSKCC' 1")

# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'IMDC'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'MSKCC'")

if n_trials == 0:
    print('evaluate on the top performing datasets')

# y_col = 'IMDC BINARY'
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'xgboost' --n_trials {n_trials} --y_col '{y_col}'")
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'logistic_regression' --n_trials {n_trials} --y_col '{y_col}'")
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'random_forest' --n_trials {n_trials} --y_col '{y_col}'")
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'svc' --n_trials {n_trials} --y_col '{y_col}'")
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'decision_tree' --n_trials {n_trials} --y_col '{y_col}'")


# y_col = 'MSKCC BINARY'
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'xgboost' --n_trials {n_trials} --y_col '{y_col}'")
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'logistic_regression' --n_trials {n_trials} --y_col '{y_col}'")
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'random_forest' --n_trials {n_trials} --y_col '{y_col}'")
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'svc' --n_trials {n_trials} --y_col '{y_col}'")
# os.system(f"python3 ml/run_traditional_classifier.py --model_name 'decision_tree' --n_trials {n_trials} --y_col '{y_col}'")


os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS AND both-PFS AND Benefit'")

os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS AND Benefit'")


os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS AND both-PFS AND Benefit'")

os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS AND Benefit'")


os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'NIVO-OS ADV EVER-OS'")

os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'NIVO-OS AND NIVO-PFS AND EVER-OS AND EVER-PFS'")

# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'IMDC'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'MSKCC'")

# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'EVER-OS'")

os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'NIVO-OS ADV EVER-OS'")

os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS AND both-PFS'")

# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'NIVO-OS AND EVER-OS'")

os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS AND both-PFS AND MSKCC'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS AND both-PFS AND MSKCC'")

# python3 ml/run_finetune_study.py 0 'RCC-2925' 'NIVO-OS AND EVER-OS'