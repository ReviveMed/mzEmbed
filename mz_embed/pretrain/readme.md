## Pretraining 

Examples of the Python commands for pre-training use cases. 

### **1. Pretraining VAE Models**
This command pretrains a Variational Autoencoder (VAE) using large-scale metabolomics data, with hyperparameter optimization using **Optuna**:

```bash
python ../pretrain/run_pretrain_VAE_main.py \
  --input_data_location "$INPUT_DATA_LOCATION" \
  --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
  --latent_size 128 256 32 \
  --num_hidden_layers 2 3 1 \
  --dropout_rate 0.1 0.45 0.05 \
  --noise_factor 0.2 0.25 0.05 \
  --learning_rate 8e-5 2e-4 \
  --l1_reg 0 \
  --weight_decay 1e-6 1e-4 \
  --batch_size 64 \
  --patience 25 \
  --num_epochs 400 \
  --trial_name "$TRIAL_NAME" \
  --n_trials 50

```

Alternatively, you can run this process using the provided script:
```
cd mzEmbed/mz_embed/scripts
./pretrain_run_VAE_main.sh
```


---

#### **2. Retraining Pretrained VAE Models**
Retrain the last layer of pretrained VAE models for specific tasks, such as learning demographic or clinical variables:

```bash
python ../pretrain/retrain_last_layer_pretrain_VAE_main.py \
    --input_data_location "$INPUT_DATA_LOCATION" \
    --pretrain_save_dir "$PRETRAIN_SAVE_DIR" \
    --pretrain_model_name "$pretrain_model_name" \
    --pretrain_trial_ID "$pretrain_trial_ID" \
    --task "gender" \
    --task_type "classification" \
    --add_post_latent_layers 'False' \
    --post_latent_layer_size "64" \
    --num_layers_to_retrain "1" \
    --dropout_rate 0.4 \
    --learning_rate 1e-4 \
    --l1_reg 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --patience 5 \
    --num_epochs 30

```

Alternatively, you can run this process using the provided script:
```
cd mzEmbed/mz_embed/scripts
./pretrain_retrain_VAE_main.sh
```


---

### Parameter Details
- **Range Parameters:** Arguments with three values (`min max increment`) allow grid search. Example: `--latent_size 128 256 32` searches sizes between 128 and 256 in steps of 32.
- **Single-Value Parameters:** Fixed values used directly in training (e.g., `--batch_size 64`).
- **Optimization:** Optuna is used for hyperparameter tuning where specified (e.g., learning rate, dropout rate) for training models. hyperparameter tuning for re-training models are done via grid search. 

