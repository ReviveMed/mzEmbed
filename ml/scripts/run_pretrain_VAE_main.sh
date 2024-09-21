#!/bin/bash


# python pretrain/run_pretrain_VAE_main.py \
#   --input_data_location /home/leilapirhaji/PROCESSED_DATA \
#   --pretrain_save_dir /home/leilapirhaji/pretrained_models \
#   --latent_size 460 \
#   --num_hidden_layers 2 \
#   --dropout_rate 0.2 \
#   --noise_factor 0.15 \
#   --learning_rate 1.7e-4 \
#   --l1_reg 0 \
#   --weight_decay 1e-4 \
#   --batch_size 94 \
#   --patience 0 \
#   --num_epochs 100 \
#   --trial_name pretrain_VAE_optuna_test2 \
#   --n_trials 3




python ../pretrain/run_pretrain_VAE_main.py \
  --input_data_location /home/leilapirhaji/PROCESSED_DATA \
  --pretrain_save_dir /home/leilapirhaji/pretrained_models \
  --latent_size 380 480 10 \
  --num_hidden_layers 2 \
  --dropout_rate 0.0 0.5 0.05 \
  --noise_factor 0.0 0.25 0.05 \
  --learning_rate 1e-4 1e-3 \
  --l1_reg 0 \
  --weight_decay 1e-6 1e-2 \
  --batch_size 32 126 32 \
  --patience 0 \
  --num_epochs 80 200 20 \
  --trial_name pretrain_VAE_L_380_480_e_80_200_KL_ann_.75 \
  --n_trials 200
