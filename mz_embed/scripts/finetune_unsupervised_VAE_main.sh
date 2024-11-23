#!/bin/bash


# Define common variables
INPUT_DATA_LOCATION="~/input_data"
FINETUNE_SAVE_DIR="/~/finetune_VAE"
PRETRAIN_SAVE_DIR="~/pretrained_models"
PRETRAIN_MODEL_DF_FILE="~/top_pretrain_VAEs.txt"



# # Run the finetune_VAE_local_main.py script
python ../finetune/finetune_VAE_unsupervised_main.py \
    --input_data_location $INPUT_DATA_LOCATION \
    --finetune_save_dir $FINETUNE_SAVE_DIR \
    --pretrain_save_dir $PRETRAIN_SAVE_DIR \
    --pretrain_model_list_file $PRETRAIN_MODEL_DF_FILE \
    --dropout_rate 0.1 0.4 0.05 \
    --learning_rate 1e-5 5e-5 \
    --l1_reg 1e-6 1e-3 \
    --weight_decay 1e-6 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 30 50 10 \
    --n_trials 10

