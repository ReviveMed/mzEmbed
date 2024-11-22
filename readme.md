# mzLearn, a data-driven LC/MS signal detection algorithm, enables pre-trained generative models for untargeted metabolomics

This repository contains the codebase for **mzEmbed**, a framework for developing pre-trained generative models and fine-tuning them for specific tasks for untargeted metabolomics datasets.

**Author**:
- [Leila Pirhaji](https://www.linkedin.com/in/pirhaji/)


## Overview of mzLearn

**mzLearn** is a data-driven algorithm designed to autonomously detect metabolite signals from raw LC/MS data without requiring input parameters from the user. The algorithm processes raw LC/MS data files in the open-source `mzML` format, iteratively learning signal characteristics to ensure high-quality signal detection. 

### Key Features of mzLearn:
- **Zero-parameter design:** No prior knowledge or QC samples are required.
- **Iterative learning:** mzLearn autonomously refines signal detection, correcting for retention time (rt) and intensity drifts caused by batch effects and run order.
- **Output:** A two-dimensional table of detected features defined by median rt and m/z values, with normalized intensities across samples.
- **Scalability:** Capable of handling large-scale datasets (e.g., 2,075 files in a single run).
- **Accessibility:** mzLearn’s website for accessing the tool is available at [http://mzlearn.com/](http://mzlearn.com/).

---

## Overview of mzEmbed Codebase

**mzEmbed** extends mzLearn’s capabilities by combining outputs from multiple datasets to develop pre-trained generative models and applying them to a range of metabolomics applications.

### Key Components of mzEmbed:
1. **Pre-trained Model Development:**
   - Combines metabolomics data from multiple studies to create robust pre-trained generative models.
   - Supports Variational Autoencoders (VAEs) for unsupervised learning of metabolite representations.
   - Enables parameter optimization using grid search and Optuna for hyperparameter tuning.
   - Outputs embeddings that capture biological and demographic variability, such as age, disease state.

2. **Fine-Tuning Pre-Trained Models:**
   - Allows fine-tuning of pre-trained models on independent datasets for improved task-specific performance.
   - Supports fine-tuning for binary classification, multi-class classification, and survival analysis.

3. **Task-Specific Model Refinement:**
   - Retrains the last layer of fine-tuned models for specific tasks, such as prognostic biomarker discovery or treatment response prediction.
   - Implements synergistic and adversarial learning frameworks to train joint and predictive models.

4. **Advanced Architectures:**
   - Supports the development of joint learning models for treatment-independent, prognostic stratification of patient.
   - Implements adversarial learning to isolate treatment-specific predictive biomarkers, or predictive stratification of patient.

---

## Getting Started

### Requirements
- Python 3.9 or higher

### Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:ReviveMed/mzEmbed.git
   cd mzEmbed
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. building the package:
    ```
    cd mz_embed'
    python -m build
    pip install -e .
    ```

---


## Usage: Running Python Commands Directly

The repository supports six main use cases, including pretraining, fine-tuning, and advanced learning architectures. Below are examples of the Python commands for each use case. 


#### **1. Pretraining VAE Models**
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

#### **3. Unsupervised Fine-Tuning**
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

#### **4. Task-Specific Fine-Tuning**
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

#### **5. Joint Learning for Prognostic Models**
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

#### **6. Adversarial Learning for Predictive Models**
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



---

## License
This project is licensed under the Academic and Non-Profit Use License. See the LICENSE.txt file for details.


---

## Citation
If you use mzLearn or mzEmbed in your research, please cite:
```
[mzLearn, a data-driven LC/MS signal detection algorithm, enables pre-trained generative models for untargeted metabolomics]
[Leila Pirhaji, Jonah Eaton, Adarsh K. Jeewajee, Min Zhang, Matthew Morris, Maria Karasarides]
[Journal/Conference Name]
```

---