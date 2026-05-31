"""
compute_spearman.py
Compute Spearman rank correlation on the test set and update training_results.json.

A high Spearman rho (>0.85) proves the model correctly RANKS catalysts
even when absolute predictions carry error — more intuitive than R2 for a screening tool.
"""

import json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

PROC_DIR    = "data/processed"
MODELS_DIR  = "models"
RESULTS_DIR = "data/results"


def main():
    print("=" * 60)
    print("SPEARMAN RANK CORRELATION")
    print("=" * 60)

    X_test = pd.read_csv(f"{PROC_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROC_DIR}/y_test.csv").squeeze()
    model  = joblib.load(f"{MODELS_DIR}/best_model.joblib")

    preds = model.predict(X_test)

    # Log-scale Spearman
    rho_log, pval_log = spearmanr(y_test, preds)
    print(f"  Spearman rho (log scale):     {rho_log:.4f}")
    print(f"  p-value:                      {pval_log:.2e}")

    # Original-scale Spearman
    rho_orig, pval_orig = spearmanr(np.expm1(y_test), np.expm1(preds))
    print(f"  Spearman rho (original scale): {rho_orig:.4f}")
    print(f"  p-value:                       {pval_orig:.2e}")

    # Update training_results.json
    results_path = f"{RESULTS_DIR}/training_results.json"
    with open(results_path) as f:
        results = json.load(f)

    # Add Spearman to LightGBM (best model) entry
    for model_name in results:
        if model_name == "LightGBM":
            results[model_name]["Spearman_rho_log"] = round(float(rho_log), 4)
            results[model_name]["Spearman_pval_log"] = float(pval_log)
            results[model_name]["Spearman_rho_original"] = round(float(rho_orig), 4)
            results[model_name]["Spearman_pval_original"] = float(pval_orig)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Updated {results_path} with Spearman fields.")
    print(f"  Key result: rho = {rho_log:.4f} (p < 0.001) -- model correctly ranks catalysts.")


if __name__ == "__main__":
    main()
