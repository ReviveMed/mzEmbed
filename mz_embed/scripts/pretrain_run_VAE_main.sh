#!/bin/bash

# nohup ./run_pretrain_VAE_main.sh > output_pretrain.log 2>&1 &
# tail -f output_pretrain.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/pretrain_finetune_inputs"
PRETRAIN_SAVE_DIR="/home/leilapirhaji/pretrained_HILIC_pos"
TRIAL_NAME="pretrain_VAE_L_425_e_400_p_25_s_8.1.1"



python ../pretrain/run_pretrain_VAE_main.py \
  --input_data_location "$INPUT_DATA_LOCATION" \
  --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
  --latent_size 425 \
  --num_hidden_layers 2 \
  --dropout_rate 0.1 0.15 0.05 \
  --noise_factor 0.2 0.25 0.05 \
  --learning_rate 8e-5 2e-4 \
  --l1_reg 0 \
  --weight_decay 1e-6 1e-4 \
  --batch_size 64 \
  --patience 25 \
  --num_epochs 400 \
  --trial_name "$TRIAL_NAME" \
  --n_trials 50





