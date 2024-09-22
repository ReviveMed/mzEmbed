#!/bin/bash
#running code in the backgrouhn
#nohup ./run_finetune_VAE_eval_latent_local_main.sh > output_finetune.log 2>&1 &
#tail -f output_finetune.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/PROCESSED_DATA"
FINETUNE_SAVE_DIR="/home/leilapirhaji/finetune_VAE_models"
PRETRAIN_SAVE_DIR="/home/leilapirhaji/pretrained_models"
PRETRAIN_MODEL_DF_FILE="/home/leilapirhaji/top_pretrain_VAE_tensor_board.txt"
N_TRIALS=50



# Run the finetune_VAE_local_main.py script
python ../finetune/finetune_VAE_local_main.py \
    --input_data_location $INPUT_DATA_LOCATION \
    --finetune_save_dir $FINETUNE_SAVE_DIR \
    --pretrain_save_dir $PRETRAIN_SAVE_DIR \
    --pretrain_model_list_file $PRETRAIN_MODEL_DF_FILE \
    --n_trial $N_TRIALS


# Evaluate the latent space
python ../finetune/eval_finetune_latent_local_main.py \
    --input_data_location $INPUT_DATA_LOCATION \
    --finetune_save_dir $FINETUNE_SAVE_DIR \
    --pretrain_model_list_file $PRETRAIN_MODEL_DF_FILE