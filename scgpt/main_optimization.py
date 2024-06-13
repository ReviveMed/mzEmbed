import optuna
import gc

from main_wrapper import train_scgpt_wrapper
import argparse

WEBAPP_DB_LOC = 'mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB'


# Create the parser
parser = argparse.ArgumentParser(description='Parse user input from the terminal')

# Add arguments
parser.add_argument('--label', type=str, help='What is the celltype label used for pretraining',default = 'Age Group')
parser.add_argument('--ntrials', type=int, help='Number of optuna trials', default=10)

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
celltype_label = args.label
num_trials = args.ntrials


study_info_dict = {
    'study_name': [f'scGPT {celltype_label} June04'],
    'directions': ['maximize','minimize','minimize'],
    'task': 'integration'
} 



def objective(trial):
    dataset_name = 'metab_v2'
    mask_ratio = trial.suggest_float('mask_ratio', 0.05, 0.25, step=0.05)
    epochs = trial.suggest_int('epochs', 20, 40, step=1)
    n_bins = trial.suggest_int('n_bins', 21, 51, step=10)
    # lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    # layer_size = trial.suggest_int('layer_size', 64, 512, step=64)
    # layer_size = trial.suggest_int('layer_size', 64, 128, step=64)
    layer_size = trial.suggest_int('layer_size', 32, 64, step=32)
    nlayers = trial.suggest_int('nlayers', 2, 8, step=2)
    # nlayers = trial.suggest_int('nlayers', 2, 12, step=1)
    nhead = trial.suggest_int('nhead', 2, 8, step=2)
    if nhead == 6:
        print('overwriting nhead=6 to nhead=4')
        nhead = 4
    dropout = trial.suggest_float('dropout', 0.05, 0.25, step=0.05)
    # max_seq_len = trial.suggest_int('max_seq_len', 1001, 2401, step=200)
    max_seq_len = trial.suggest_int('max_seq_len', 2001, 5201, step=400)


    # mask_ratio = trial.suggest_float('mask_ratio', 0.1, 0.5, step=0.05)
    # epochs = trial.suggest_int('epochs', 5, 30, step=1)
    # n_bins = trial.suggest_int('n_bins', 11, 51, step=10)
    # # lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    # # layer_size = trial.suggest_int('layer_size', 64, 512, step=64)
    # # layer_size = trial.suggest_int('layer_size', 64, 128, step=64)
    # layer_size = trial.suggest_int('layer_size', 32, 64, step=32)
    # nlayers = trial.suggest_int('nlayers', 2, 8, step=2)
    # # nlayers = trial.suggest_int('nlayers', 2, 12, step=1)
    # nhead = trial.suggest_int('nhead', 2, 8, step=2)
    # if nhead == 6:
    #     print('overwriting nhead=6 to nhead=4')
    #     nhead = 4
    # dropout = trial.suggest_float('dropout', 0.05, 0.25, step=0.05)
    # max_seq_len = trial.suggest_int('max_seq_len', 1001, 2001, step=200)

    task = study_info_dict['task']
    if task == 'integration':
        CLS = False
    elif task == 'annotation':
        CLS = True

    if celltype_label == 'Dummy':
        GEPC = False
    else:
        GEPC = True

    pretrain_config = dict(
        task =task,
        CLS = CLS,
        GEPC = GEPC,
        celltype_label = celltype_label,
        dataset_name=dataset_name,
        datasubset_label = 'pretrain_set',
        trainsubset_label = 'Train',
        valsubset_label = 'Val',
        testsubset_label = 'Test',
        load_model = None,
        mask_ratio = mask_ratio,
        epochs = epochs,
        n_bins = n_bins,
        layer_size = layer_size,
        nlayers = nlayers,
        nhead = nhead,
        dropout = dropout,
        max_seq_len = max_seq_len,
    )



    pretrain_res = train_scgpt_wrapper(**pretrain_config)
    if 'val/avg_bio' in pretrain_res:
        val_avg_bio = pretrain_res['val/avg_bio']
    else: val_avg_bio = 0

    val_gep_loss = pretrain_res['val/gep']
    if 'val/gepc' in pretrain_res:
        val_gepc_loss = pretrain_res['val/gepc']
    else: val_gepc_loss = val_gep_loss

    save_dir = pretrain_res['save_dir']
    # trial.set_user_attr('save_dir', save_dir)
    trial.set_user_attr('pretrain_wandb_name', pretrain_res['wandb_name'])

    finetune_config1 = dict(
        task = 'annotation',
        CLS= True,
        dataset_name=dataset_name,
        celltype_label = "IMDC Binary",
        datasubset_label = 'finetune_set',
        trainsubset_label = 'Finetune',
        valsubset_label = 'Validation',
        testsubset_label = 'Test',
        load_model = save_dir,
        mask_ratio = 0.05,
        epochs = 15,
    )

    finetune_config2 = {**finetune_config1}
    finetune_config2['celltype_label'] = "MSKCC Binary"


    finetune_res1 = train_scgpt_wrapper(**finetune_config1)
    finetune1_val_cls_accuracy = finetune_res1['val/cls_accuracy']

    finetune_res2 = train_scgpt_wrapper(**finetune_config2)
    finetune2_val_cls_accuracy = finetune_res2['val/cls_accuracy']

    trial.set_user_attr('IMDC_wandb_name', finetune_res1['wandb_name'])
    trial.set_user_attr('MSKCC_wandb_name', finetune_res2['wandb_name'])
    trial.set_user_attr('IMDC_val_cls_accuracy', finetune1_val_cls_accuracy)
    trial.set_user_attr('MSKCC_val_cls_accuracy', finetune2_val_cls_accuracy)
    
    # take out the garbage and clean up
    gc.collect()



    return val_avg_bio, val_gep_loss, val_gepc_loss




study = optuna.create_study(directions=study_info_dict['directions'], 
                            study_name=study_info_dict['study_name'], 
                            storage=WEBAPP_DB_LOC, 
                            load_if_exists=True)


study.optimize(objective, n_trials=num_trials)