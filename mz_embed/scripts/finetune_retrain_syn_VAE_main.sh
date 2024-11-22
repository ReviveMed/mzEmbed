#!/bin/bash
#running code in the backgrouhn
#nohup ./retrain_syn_finetune_VAE_main.sh > output_finetune_retrain_syn.log 2>&1 &
#tail -f output_finetune_retrain_syn.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/PROCESSED_DATA_finetune_OG_split"
FINETUNE_SAVE_DIR="/home/leilapirhaji/finetune_unsupervised_VAE"
pretrain_model_name="pretrain_VAE_L_400_425_e_400_p_25_s_8.1.1"
pretrain_trial_ID="143"

TASK='NIVO OS'
SYN_TASk='EVER OS'
TASK_EVENT="OS_Event"



python ../finetune/retrain_synergistic_cox_finetune_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --task "$TASK" \
    --syn_task "$SYN_TASk" \
    --task_event "$TASK_EVENT" \
    --lambda_syn 4.0 \
    --add_post_latent_layers 'True' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "1" \
    --dropout_rate 0.4 \
    --learning_rate 1e-6 5e-6 1e-5 4e-5 4e-5 1e-4 5e-4 9e-4 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 20 \

