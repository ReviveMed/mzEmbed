#!/bin/bash

python ../finetune/eval_finetune_latent_main.py --input_data_location /home/leilapirhaji/PROCESSED_DATA_2 --finetune_save_dir /home/leilapirhaji/finetune_VAE_models --pretrain_model_list_file /home/leilapirhaji/pretrained_models_to_finetune.txt --result_name top_finetune_latent_w_zero