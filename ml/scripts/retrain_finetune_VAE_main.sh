#!/bin/bash
#running code in the backgrouhn
#nohup ./retrain_finetune_VAE_main.sh > output_finetune_retrain.log 2>&1 &
#tail -f output_finetune_retrain.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/PROCESSED_DATA_finetune_OG_split"
FINETUNE_SAVE_DIR="/home/leilapirhaji/finetune_unsupervised_VAE"

PRETRAIN_MODEL_DF_FILE="/home/leilapirhaji/top_pretrain_VAE_S_8.1.1.txt"
pretrain_model_name='pretrain_VAE_L_410_490_e_400_p_25_S_8.1.1'
pretrain_trial_ID='106'


# TASK="OS"
# TASK_TYPE="cox"

TASK='IMDC BINARY'
TASK_TYPE='classification'

num_classes='2'
TASK_EVENT='OS_Event'


# Run the finetune_VAE_local_main.py script
python ../finetune/retrain_finetune_VAE_main.py \
    --input_data_location $INPUT_DATA_LOCATION \
    --finetune_save_dir $FINETUNE_SAVE_DIR \
    --pretrain_model_name $pretrain_model_name \
    --pretrain_trial_ID $pretrain_trial_ID \
    --task "$TASK" \
    --task_type $TASK_TYPE \
    --num_classes $num_classes \
    --task_event $TASK_EVENT \
    --add_post_latent_layers 'False'\
    --post_latent_layer_size '16,32' \
    --num_layers_to_retrain '1' \
    --dropout_rate 0.4 \
    --learning_rate 1e-4 \
    --l1_reg 1e-6 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 20 \
    --n_trials 3
