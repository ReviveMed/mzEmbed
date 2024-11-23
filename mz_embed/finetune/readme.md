## Fine-tuning 

Examples of the Python commands for fine-tuning use cases. 

### **1. Unsupervised Fine-Tuning**
Fine-tune pretrained VAE models in an unsupervised manner on new datasets with transfer learning from a list of pretrained models

```bash
python ../finetune/finetune_VAE_unsupervised_main.py \
    --input_data_location $INPUT_DATA_LOCATION \
    --finetune_save_dir $FINETUNE_SAVE_DIR \
    --pretrain_save_dir $PRETRAIN_SAVE_DIR \
    --pretrain_model_list_file $PRETRAIN_MODEL_DF_FILE \
    --dropout_rate 0.1 0.4 0.05 \
    --learning_rate 1e-5 1e-3 \
    --l1_reg 1e-6 1e-3 \
    --weight_decay 1e-6 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 30 50 10 \
    --n_trials 50
```

Alternatively, you can run this process using the provided script:
```
cd mzEmbed/mz_embed/scripts
./finetune_unsupervised_VAE_main.sh
```

---

### **2. Task-Specific Fine-Tuning**
Fine-tune pretrained VAE models for binary classification, multi-class classification, or survival analysis. 

```bash
python ../finetune/retrain_finetune_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --task "OS" \
    --task_type "cox" \
    --num_classes "1" \
    --task_event "OS_Event" \
    --optimization_type 'grid_search' \
    --add_post_latent_layers 'False' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "1" \
    --dropout_rate 0.4 \
    --learning_rate 1e-5 3e-5 5e-5 7e-5 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 20 \
    --n_trials 1
```

Alternatively, you can run this process using the provided script:
```
cd mzEmbed/mz_embed/scripts
./finetune_retrain_VAE_main.sh
```

---

### **3. Joint Learning for Prognostic Models**
Jointly fine-tune two VAE models to identify treatment-independent prognostic features. Grid search is used for hyper-parameter tuning.

```bash
python ../finetune/retrain_synergistic_cox_finetune_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --task 'NIVO OS' \
    --syn_task 'EVER OS' \
    --task_event "OS_Event" \
    --lambda_syn 2.0 \
    --add_post_latent_layers 'True' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "1" \
    --dropout_rate 0.4 \
    --learning_rate 1e-5 3e-5 5e-5 7e-5 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 20 \
```

Alternatively, you can run this process using the provided script:
```
cd mzEmbed/mz_embed/scripts
./finetune_retrain_syn_VAE_main.sh
```

---

### **4. Adversarial Learning for Predictive Models**
Perform adversarial fine-tuning to isolate treatment-specific predictive features. Grid search is used for hyper-parameter tuning.

```bash
python ../finetune/retrain_adverserial_cox_finetune_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --finetune_save_dir "$FINETUNE_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --task 'NIVO OS' \
    --adv_task 'EVER OS' \
    --task_event "OS_Event" \
    --lambda_adv 0.1 \
    --add_post_latent_layers 'True' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "1" \
    --dropout_rate 0.4 \
    --learning_rate 1e-5 3e-5 5e-5 7e-5 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 0 \
    --num_epochs 20 \
```

Alternatively, you can run this process using the provided script:
```
cd mzEmbed/mz_embed/scripts
./finetune_retrain_adv_VAE_main.sh
```


---

### Parameter Details
- **Range Parameters:** Arguments with three values (`min max increment`) allow grid search. Example: `--latent_size 128 256 32` searches sizes between 128 and 256 in steps of 32.
- **Single-Value Parameters:** Fixed values used directly in training (e.g., `--batch_size 64`).
- **Optimization:** Optuna is used for hyperparameter tuning where specified (e.g., learning rate, dropout rate) for training models. hyperparameter tuning for re-training models are done via grid search. 

