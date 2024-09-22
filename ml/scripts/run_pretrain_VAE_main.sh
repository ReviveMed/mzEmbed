#!/bin/bash

# nohup ./run_pretrain_VAE_main.sh > output_pretrain.log 2>&1 &
# tail -f output_pretrain.log



python ../pretrain/run_pretrain_VAE_main.py \
  --input_data_location /home/leilapirhaji/PROCESSED_DATA \
  --pretrain_save_dir /home/leilapirhaji/pretrained_models \
  --latent_size 425 485 5 \
  --num_hidden_layers 2 \
  --dropout_rate 0.0 0.4 0.05 \
  --noise_factor 0.0 0.25 0.05 \
  --learning_rate 8e-5 3e-4 \
  --l1_reg 0 \
  --weight_decay 1e-6 1e-3 \
  --batch_size 64 128 32 \
  --patience 25 \
  --num_epochs 400 \
  --trial_name pretrain_VAE_L_425_485_e_400_p_25 \
  --n_trials 100


# python ../pretrain/run_pretrain_VAE_main.py \
#   --input_data_location /home/leilapirhaji/PROCESSED_DATA \
#   --pretrain_save_dir /home/leilapirhaji/pretrained_models \
#   --latent_size 470 \
#   --num_hidden_layers 2 \
#   --dropout_rate 0.05 \
#   --noise_factor 0.25 \
#   --learning_rate 1.7e-4 \
#   --l1_reg 0 \
#   --weight_decay 1e-5 \
#   --batch_size 96 \
#   --patience 25 \
#   --num_epochs 400 \
#   --trial_name pretrain_VAE_test_tensor_board \
#   --n_trials 1



#save_path=/home/leilapirhaji/pretrained_models/tensorboard_logs/pretrain_VAE_test_tensor_board/trial_0
#tensorboard --logdir $save_path
#Open the Browser: Visit http://localhost:6006/
