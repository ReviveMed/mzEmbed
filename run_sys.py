import os
import sys
if len(sys.argv)>1:
    n_trials = int(sys.argv[1])
else:
    n_trials = 10

if len(sys.argv)>2:
    run_id = sys.argv[2]
else:
    run_id = 'RCC-3011'


# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'both-OS'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'NIVO-OS'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'EVER-OS'")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'IMDC' 1")
# os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'MSKCC' 1")

os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'IMDC'")
os.system(f"python3 ml/run_finetune_study.py {n_trials} '{run_id}' 'MSKCC'")