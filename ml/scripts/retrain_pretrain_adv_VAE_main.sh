#!/bin/bash
#running code in the backgrouhn
#nohup ./retrain_pretrain_adv_VAE_main.sh > output_adv_pretrain_retrain.log 2>&1 &
#tail -f output_adv_pretrain_retrain.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/PROCESSED_DATA_S_8.1.1"
PRETRAIN_SAVE_DIR="/home/leilapirhaji/pretrained_models"
pretrain_model_name="pretrain_VAE_L_400_425_e_400_p_25_s_8.1.1"
pretrain_trial_ID="143"

TASK='Study ID'



python ../pretrain/retrain_last_layer_adv_pretrain_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --adv_task "$TASK" \
    --add_post_latent_layers 'False' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "1,2" \
    --dropout_rate 0.4 \
    --learning_rate 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 9e-4 1e-3 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 20