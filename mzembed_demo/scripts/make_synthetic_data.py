#!/usr/bin/env python3
"""Regenerate tiny synthetic mzEmbed demo CSVs (no PHI/proprietary data)."""
import os, numpy as np, pandas as pd
rng = np.random.default_rng(42)
n_features = 2736

def make_matrix(n_samples):
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype("float32")
    cols = [f"f{str(i).zfill(4)}" for i in range(1, n_features+1)]
    idx = [f"S{str(i).zfill(4)}" for i in range(1, n_samples+1)]
    return pd.DataFrame(X, index=idx, columns=cols)

def make_labels(n_samples):
    return pd.DataFrame({
        "IMDC": rng.binomial(1, 0.45, size=n_samples).astype(int),
        "OS_time": rng.integers(100, 2000, size=n_samples).astype(int),
        "OS_event": rng.binomial(1, 0.6, size=n_samples).astype(int),
    }, index=[f"S{str(i).zfill(4)}" for i in range(1, n_samples+1)])

def main(out_dir="data/synthetic_small"):
    os.makedirs(out_dir, exist_ok=True)
    # Pretrain
    for name, n in [("Discovery_Train", 100), ("Discovery_Val", 25), ("Test", 25)]:
        X = make_matrix(n)
        y = make_labels(n)
        X.to_csv(os.path.join(out_dir, f"X_Pretrain_{name}.csv"))
        y.to_csv(os.path.join(out_dir, f"y_Pretrain_{name}.csv"))
    # Finetune
    for name, n in [("Train", 80), ("Val", 20), ("Test", 20)]:
        X = make_matrix(n)
        y = make_labels(n)
        X.to_csv(os.path.join(out_dir, f"X_Finetune_{name}.csv"))
        y.to_csv(os.path.join(out_dir, f"y_Finetune_{name}.csv"))
    print(f"Wrote synthetic CSVs to {out_dir}")

if __name__ == "__main__":
    main()
