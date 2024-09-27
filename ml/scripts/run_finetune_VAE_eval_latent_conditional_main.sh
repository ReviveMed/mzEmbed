#!/bin/bash
#running code in the backgrouhn
#nohup ./run_finetune_VAE_eval_latent_conditional_main.sh > output_finetune.log 2>&1 &
#tail -f output_finetune.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/PROCESSED_DATA_S_8.1.1"
FINETUNE_SAVE_DIR="/home/leilapirhaji/finetune_conditional_VAE"
PRETRAIN_SAVE_DIR="/home/leilapirhaji/pretrained_models"
PRETRAIN_MODEL_DF_FILE="/home/leilapirhaji/top_pretrain_VAE_L_410_490_e_400_p_25_S_8.1.1.txt"

CONDITION_LIST='OS,NIVO OS,EVER OS,OS_Event'



# Run the finetune_VAE_local_main.py script
python ../finetune/finetune_VAE_conditional_main.py \
    --input_data_location $INPUT_DATA_LOCATION \
    --finetune_save_dir $FINETUNE_SAVE_DIR \
    --pretrain_save_dir $PRETRAIN_SAVE_DIR \
    --pretrain_model_list_file $PRETRAIN_MODEL_DF_FILE \
    --condition_list "$CONDITION_LIST" \
    --dropout_rate 0.1 0.4 0.05 \
    --learning_rate 1e-5 1e-3 \
    --l1_reg 1e-6 1e-3 \
    --weight_decay 1e-6 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 30 100 20 \
    --n_trials 30


# # # Evaluate the latent space
python ../finetune/eval_finetune_latent_conditional_main.py \
    --input_data_location $INPUT_DATA_LOCATION \
    --finetune_save_dir $FINETUNE_SAVE_DIR \
    --pretrain_model_list_file $PRETRAIN_MODEL_DF_FILE \
    --condition_list "$CONDITION_LIST"