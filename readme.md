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
   - Retrains the last layer of fine-tuned models for specific tasks, such as clinical classifcation  and surivival analysis.
   

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
    cd mz_embed
    python -m build
    pip install -e .
    ```

---


## Usage: 

The repository supports six main use cases, including pretraining, fine-tuning, and advanced learning architectures. **pretrain** and **finetune** directories includes examples of the Python commands for each use case. 

---

## Quickstart (TL;DR)

```bash
# 1) Pretrain on public 2,736-feature matrix
python -m mz_embed.pretrain.run_pretrain_VAE_main   --data_location /path/to/pretrain_csvs   --latent_size 460 --n_layers 3 --trials 40 --max_epochs 200   --output_dir /path/to/out_pretrain
```

```python
# 2) Export encoder + latent embeddings
from mz_embed.pretrain.get_pretrain_encoder import get_pretrain_encoder
get_pretrain_encoder("/path/to/out_pretrain",
                     "/path/to/pretrain_csvs",
                     "/path/to/encoder_out")
```

```bash
# 3) Fine-tune on a new cohort (uses transfer learning if --pretrained_dir is set)
python -m mz_embed.finetune.finetune_VAE_unsupervised_main   --data_location /path/to/finetune_csvs   --pretrained_dir /path/to/out_pretrain   --output_dir /path/to/out_finetune
```

```bash
# 4) Evaluate a lightweight task head on the held-out test set
python -m mz_embed.finetune.best_finetune_model_test_eval   --model_dir /path/to/out_finetune   --data_location /path/to/finetune_csvs   --task binary
```

**Expected input files (CSV, floats; shape `[n_samples, 2736]`)**
- **Pretrain:** `X_Pretrain_Discovery_Train.csv`, `X_Pretrain_Discovery_Val.csv`, `X_Pretrain_Test.csv` (+ optional `y_*`)
- **Finetune:** `X_Finetune_Train.csv`, `X_Finetune_Val.csv`, `X_Finetune_Test.csv` (+ required `y_*` for tasks)

We assume `pandas.read_csv(..., index_col=0)` semantics (row index in the first column).

---

## API Overview (user-facing entry points)

### 1) `mz_embed.pretrain.run_pretrain_VAE_main.main(**kwargs)`
Run Optuna-driven unsupervised pretraining of the VAE on the standardized 2,736-feature matrix.

**Args**
- `data_location: str` — folder containing `X_Pretrain_*` (float; z-scored; `[n, 2736]`) and optional `y_Pretrain_*`.
- `output_dir: str` — destination for artifacts (`best_model.pt`, `study.html`, `config.json`, logs, latent plots`).
- `latent_size: int` — latent dimension (e.g., 460).
- `n_layers: int` — encoder/decoder depth `{2, 3, 4}`.
- `kl_weight: float` — target KL weight (KL anneals from 0 to `kl_weight`).
- `dropout: float` — 0–0.5.
- `learning_rate: float`
- `trials: int` — number of Optuna trials.
- `max_epochs: int` — early stopping monitors validation loss.
- `seed: int`

**Returns**  
Artifacts written to `output_dir` (no Python return object).

---

### 2) `mz_embed.pretrain.get_pretrain_encoder.get_pretrain_encoder(model_id, data_location, output_dir)`
Load a saved pretrained VAE, export the encoder and latent embeddings for each split.

**Args**
- `model_id: str` — path to directory with `best_model.pt` (or a `.pt` file).
- `data_location: str` — folder with `X_Pretrain_{All,Discovery_Train,Discovery_Val,Test}.csv` (+ optional `y_*`).
- `output_dir: str` — saves `encoder.pt` and `Z_Pretrain_*.csv`.

**Returns**
- `encoder: torch.nn.Module`
- `latents: dict[str, numpy.ndarray]` — split → latent array `[n, latent_size]`
- `labels: dict[str, pandas.DataFrame]` — if provided

**Example**
```python
from mz_embed.pretrain.get_pretrain_encoder import get_pretrain_encoder
enc, Z, Y = get_pretrain_encoder(
    model_id="/path/to/out_pretrain",
    data_location="/path/to/pretrain_csvs",
    output_dir="/path/to/encoder_out"
)
print({k: v.shape for k, v in Z.items()})
```

---

### 3) `mz_embed.finetune.finetune_VAE_unsupervised_main.main(**kwargs)`
Unsupervised fine-tuning of a VAE on a new cohort. If `pretrained_dir` is set, weights are initialized from the pretrained model (transfer learning); otherwise the model is randomly initialized.

**Args**
- `data_location: str` — folder with `X_Finetune_{Train,Val,Test}.csv` (+ `y_*`).
- `output_dir: str` — destination for `best_finetune_model.pt`, `study.html`, `config.json`.
- `pretrained_dir: str | None` — path to pretrained directory or `None`.
- Other hyperparameters as in pretraining (`latent_size`, `n_layers`, `kl_weight`, `dropout`, `learning_rate`, `trials`, `max_epochs`, `seed`).

**Returns**  
Artifacts written to `output_dir`.

---

### 4) `mz_embed.finetune.best_finetune_model_test_eval.evaluate_model_main(model_dir, data_location, task='binary', seed=42, **kwargs)`
Attach a lightweight task head to the fine-tuned encoder and evaluate on the held-out test set.

**Args**
- `model_dir: str` — folder with `best_finetune_model.pt` and the saved config.
- `data_location: str` — `X_Finetune_*`, `y_Finetune_*`.
- `task: {'binary', 'multiclass', 'cox'}` — IMDC binary (AUC), IMDC 3-class (F1), or OS (C-index).
- `seed: int`

**Returns**
- `dict` — metrics (e.g., `{'AUC': 0.93}` / `{'F1': 0.60}` / `{'C-index': 0.67}`)

**CLI example**
```bash
python -m mz_embed.finetune.best_finetune_model_test_eval   --model_dir /path/to/out_finetune   --data_location /path/to/finetune_csvs   --task binary
```

---

## Notes on data types and shapes
- **Feature matrices:** CSV of floats, shape `[n_samples, 2736]` (the robust mzLearn peak set).
- **Labels:** CSV with task-specific columns (e.g., IMDC class, OS time/event).
- **Index/columns:** first column is treated as row index (`pandas.read_csv(..., index_col=0)`).

---

## Documentation style
All user-facing functions include **NumPy-style docstrings** with Purpose, Args (names, types, shapes), Returns, and Examples. See:
- `mz_embed/pretrain/run_pretrain_VAE_main.py`
- `mz_embed/pretrain/get_pretrain_encoder.py`
- `mz_embed/finetune/finetune_VAE_unsupervised_main.py`
- `mz_embed/finetune/best_finetune_model_test_eval.py`


---

## License
This project is licensed under the Academic and Non-Profit Use License. See the LICENSE.txt file for details.


---

## Citation
If you use mzLearn or mzEmbed in your research, please cite:

mzLearn, a data-driven LC/MS signal detection algorithm, enables pre-trained generative models for untargeted metabolomics
Leila Pirhaji, Jonah Eaton, Adarsh K. Jeewajee, Min Zhang, Matthew Morris, Maria Karasarides


---
