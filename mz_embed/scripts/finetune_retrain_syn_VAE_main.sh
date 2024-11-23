#!/bin/bash


# Define common variables
INPUT_DATA_LOCATION="~/input_data"
FINETUNE_SAVE_DIR="~/finetune_VAE"
pretrain_model_name="pretrain_VAE_L_128_256_e_400_p_25"
pretrain_trial_ID="126"

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
    --learning_rate 1e-5 4e-5 4e-5 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 20 \

