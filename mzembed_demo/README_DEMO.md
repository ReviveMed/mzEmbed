# mzEmbed Demo: Minimal Reproducible Example (for Reviewers)

**Purpose:** Provide test data + notebook examples demonstrating the VAE workflow with the mzLearn/mzEmbed pipeline, without shipping large or proprietary datasets.

This folder includes:
- **Synthetic tiny dataset** (`data/synthetic_small/`), shaped like real inputs (CSV, floats; shape `[n_samples, 2736]`), but randomly generated — no PHI.
- Four **notebook examples** in `notebooks/`:
  1. `01_quickstart_pretrain.ipynb` — unsupervised pretraining (few trials/epochs; CPU).
  2. `02_finetune_and_evaluate.ipynb` — fine-tune and evaluate a lightweight task head (binary/multiclass/Cox).
  3. `03_adversarial_finetune.ipynb` — demonstrates adversarial learning (removing treatment/batch signal) with auto-discovery of the correct module entrypoint.
  4. `04_synergistic_learning.ipynb` — demonstrates synergistic/joint learning (prognostic/orthogonalization) with module auto-discovery.

The demo is **CPU-friendly** and designed as a **smoke test**. It demonstrates end-to-end usage and IO, not paper-level performance.

## How to run (from repo root)

### 1) Pretrain (tiny)
```bash
python -m mz_embed.pretrain.run_pretrain_VAE_main   --data_location ./mzembed_demo/data/synthetic_small   --latent_size 32 --n_layers 2 --trials 2 --max_epochs 5   --output_dir ./out_pretrain_demo
```

### 2) Export encoder + latents
```python
from mz_embed.pretrain.get_pretrain_encoder import get_pretrain_encoder
get_pretrain_encoder("./out_pretrain_demo",
                     "./mzembed_demo/data/synthetic_small",
                     "./out_encoder_demo")
```

### 3) Fine-tune (transfer learning if `--pretrained_dir` set)
```bash
python -m mz_embed.finetune.finetune_VAE_unsupervised_main   --data_location ./mzembed_demo/data/synthetic_small   --pretrained_dir ./out_pretrain_demo   --latent_size 32 --n_layers 2 --trials 2 --max_epochs 5   --output_dir ./out_finetune_demo
```

### 4) Evaluate task head (binary)
```bash
python -m mz_embed.finetune.best_finetune_model_test_eval   --model_dir ./out_finetune_demo   --data_location ./mzembed_demo/data/synthetic_small   --task binary
```

## Large data & models
For full-scale experiments, host large datasets/models **outside GitHub** (S3/Zenodo/OSF/etc.) and include a small `download_data.sh` or Python script with checksums. Keep this repo light with the synthetic demo for reproducibility.
