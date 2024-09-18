#!/bin/bash
#python ../pretrain/run_pretrain_VAE_main.py --input_data_location /home/leilapirhaji/PROCESSED_DATA_2 --pretrain_save_dir /home/leilapirhaji/pretrained_models --latent_size 256 --num_hidden_layers 2 --dropout_rate 0.1 --learning_rate 1e-5 --weight_decay 1e-6 --batch_size 64 --patience 10 --num_epochs 100 --trial_name test --n_trials 5


python ../pretrain/run_pretrain_VAE_main.py \
  --input_data_location /home/leilapirhaji/PROCESSED_DATA \
  --pretrain_save_dir /home/leilapirhaji/pretrained_models \
  --latent_size 256 512 32 \
  --num_hidden_layers 2 3 1 \
  --dropout_rate 0.1 0.5 0.05 \
  --learning_rate 1e-6 1e-4 \
  --weight_decay 1e-6 1e-3 \
  --batch_size 64 \
  --patience 10 30 5 \
  --num_epochs 200 \
  --trial_name pretrain_VAE_no_neptune \
  --n_trials 200
