#!/bin/bash



# Define common variables
INPUT_DATA_LOCATION="~/input_data"
FINETUNE_SAVE_DIR="~/finetune_VAE"
pretrain_model_name="pretrain_VAE_L_128_256_e_400_p_25"
pretrain_trial_ID="126"
TASK_EVENT="OS_Event"
TASK="OS"
TASK_TYPE="cox"
num_classes="1"

python ../finetune/retrain_finetune_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --task "$TASK" \
    --task_type "$TASK_TYPE" \
    --num_classes "$num_classes" \
    --task_event "$TASK_EVENT" \
    --optimization_type 'grid_search' \
    --add_post_latent_layers 'False' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "1" \
    --dropout_rate 0.4 \
    --learning_rate 1e-5 3e-5 5e-5 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 20 \
    --n_trials 1
