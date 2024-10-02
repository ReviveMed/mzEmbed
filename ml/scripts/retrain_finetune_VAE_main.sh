#!/bin/bash
#running code in the backgrouhn
#nohup ./retrain_finetune_VAE_main.sh > output_finetune_retrain.log 2>&1 &
#tail -f output_finetune_retrain.log

# Define common variables
INPUT_DATA_LOCATION="/home/leilapirhaji/PROCESSED_DATA_finetune_OG_split"
FINETUNE_SAVE_DIR="/home/leilapirhaji/finetune_unsupervised_VAE"
pretrain_model_name="pretrain_VAE_L_400_425_e_400_p_25_s_8.1.1"
pretrain_trial_ID="143"
TASK_EVENT="OS_Event"

# Define task lists with num_classes for classification
classification_tasks=(
    "IMDC BINARY:2"
    "MSKCC BINARY:2"
    "IMDC ORDINAL:3"
    "MSKCC ORDINAL:3"
    # Add more classification tasks in the format 'TASK_NAME:num_classes'
)

# Define survival tasks (num_classes is always 1 for Cox tasks)
survival_tasks=("OS" "NIVO OS" "EVER OS")

# Run for classification tasks
for task_info in "${classification_tasks[@]}"; do
    IFS=":" read -r TASK num_classes <<< "$task_info"  # Split task and num_classes

    TASK_TYPE="classification"

    echo "Running classification task: $TASK with num_classes: $num_classes"

    python ./finetune/retrain_finetune_VAE_main.py \
        --input_data_location "$INPUT_DATA_LOCATION" \
        --finetune_save_dir "$FINETUNE_SAVE_DIR" \
        --pretrain_model_name "$pretrain_model_name" \
        --pretrain_trial_ID "$pretrain_trial_ID" \
        --task "$TASK" \
        --task_type "$TASK_TYPE" \
        --num_classes "$num_classes" \
        --task_event "$TASK_EVENT" \
        --optimization_type 'grid_search' \
        --add_post_latent_layers 'True,False' \
        --post_latent_layer_size "32,64" \
        --num_layers_to_retrain "1,2" \
        --dropout_rate 0.4 \
        --learning_rate 1e-6 5e-6 1e-5 5e-5 1e-4 2e-4 4e-4 5e-4 \
        --l1_reg 1e-6 \
        --weight_decay 1e-3 \
        --batch_size 32 \
        --patience 0 \
        --num_epochs 20 \
        --n_trials 1
done

# Run for survival tasks
for TASK in "${survival_tasks[@]}"; do
    TASK_TYPE="cox"
    num_classes="1"

    echo "Running survival task: $TASK"

    python ./finetune/retrain_finetune_VAE_main.py \
        --input_data_location "$INPUT_DATA_LOCATION" \
        --finetune_save_dir "$FINETUNE_SAVE_DIR" \
        --pretrain_model_name "$pretrain_model_name" \
        --pretrain_trial_ID "$pretrain_trial_ID" \
        --task "$TASK" \
        --task_type "$TASK_TYPE" \
        --num_classes "$num_classes" \
        --task_event "$TASK_EVENT" \
        --optimization_type 'grid_search' \
        --add_post_latent_layers 'True,False' \
        --post_latent_layer_size "32,64" \
        --num_layers_to_retrain "1,2" \
        --dropout_rate 0.4 \
        --learning_rate 1e-6 5e-6 1e-5 5e-5 1e-4 2e-4 4e-4 5e-4 \
        --l1_reg 1e-6 \
        --weight_decay 1e-3 \
        --batch_size 32 \
        --patience 0 \
        --num_epochs 20 \
        --n_trials 1
done
