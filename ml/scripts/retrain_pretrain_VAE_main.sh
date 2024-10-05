#!/bin/bash
#running code in the backgrouhn
#nohup ./retrain_finetune_VAE_main.sh > output_finetune_retrain.log 2>&1 &
#tail -f output_finetune_retrain.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/PROCESSED_DATA_S_8.1.1"
PRETRAIN_SAVE_DIR="/home/leilapirhaji/pretrained_models"
pretrain_model_name="pretrain_VAE_L_400_425_e_400_p_25_s_8.1.1"
pretrain_trial_ID="143"

TASK='Age'


python ../pretrain/retrain_last_layer_pretrain_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --task "$TASK" \
    --add_post_latent_layers 'False' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "2" \
    --dropout_rate 0.2 \
    --learning_rate 5e-5 1e-4 4e-5 9e-4 1e-3 \
    --l1_reg 0 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 30
