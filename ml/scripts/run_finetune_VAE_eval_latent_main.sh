#!/bin/bash

#running the finetune_VAE_local_main.py script
python ../finetune/finetune_VAE_local_main.py --input_data_location /home/leilapirhaji/PROCESSED_DATA --finetune_save_dir /home/leilapirhaji/finetune_VAE_models --pretrain_save_dir /home/leilapirhaji/pretrained_models --pretrain_model_list_file /home/leilapirhaji/top_pretrain_VAE_Latent_464.txt --n_trial 50

#evaluating the latent space
python ../finetune/eval_finetune_latent_local_main.py --input_data_location /home/leilapirhaji/PROCESSED_DATA --finetune_save_dir /home/leilapirhaji/finetune_VAE_models --pretrain_model_list_file /home/leilapirhaji/top_pretrain_VAE_Latent_464.txt