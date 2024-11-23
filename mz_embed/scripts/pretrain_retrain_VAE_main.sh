#!/bin/bash

# Define common variables
INPUT_DATA_LOCATION="~/input_data"
PRETRAIN_SAVE_DIR="~/pretrained_VAE"
pretrain_model_name="pretrain_VAE_L_128_256_e_400_p_25"
pretrain_trial_ID="126"


TASK='is Female'
TAKS_TYPE='classification'


python ../pretrain/retrain_last_layer_pretrain_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --task "$TASK" \
    --task_type "$TAKS_TYPE" \
    --add_post_latent_layers 'False' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "1" \
    --dropout_rate 0.4 \
    --learning_rate 1e-4 1e-3 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 30
