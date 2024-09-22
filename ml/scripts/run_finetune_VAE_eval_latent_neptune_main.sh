#!/bin/bash

#running code in the backgrouhn
#nohup ./run_finetune_VAE_eval_latent_neptune_main.sh > output_finetune.log 2>&1 &
#tail -f output_finetune.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/PROCESSED_DATA"
FINETUNE_SAVE_DIR="/home/leilapirhaji/finetune_VAE_models_neptune"
PRETRAIN_MODEL_LIST_FILE="/home/leilapirhaji/top_neptune_pretrained_models_to_finetune.txt"
N_TRIALS=50
RESULT_NAME="Top_neptune_models_eval"

# Run the finetune_VAE_neptune_main.py script
# python ../finetune/finetune_VAE_neptune_main.py \
#     --input_data_location $INPUT_DATA_LOCATION \
#     --finetune_save_dir $FINETUNE_SAVE_DIR \
#     --pretrain_model_list_file $PRETRAIN_MODEL_LIST_FILE \
#     --n_trial $N_TRIALS


# Evaluate the latent space
python ../finetune/eval_finetune_latent_neptune_main.py \
    --input_data_location $INPUT_DATA_LOCATION \
    --finetune_save_dir $FINETUNE_SAVE_DIR \
    --pretrain_model_list_file $PRETRAIN_MODEL_LIST_FILE \
    --result_name $RESULT_NAME
