"""
compute_spearman.py
Compute Spearman rank correlation on the test set and update training_results.json.

A high Spearman rho (>0.85) proves the model correctly RANKS catalysts
even when absolute predictions carry error — more intuitive than R2 for a screening tool.
"""

import json
import os
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

    # Update training_results.json
    results_path = f"{RESULTS_DIR}/training_results.json"
    with open(results_path) as f:
        results = json.load(f)

    # Calculate for each model
    for model_name in ["LightGBM", "XGBoost", "Ridge"]:
        joblib_name = model_name.lower() + "_model.joblib"
        model_path = f"{MODELS_DIR}/{joblib_name}"
        if not os.path.exists(model_path):
            print(f"  Warning: {model_path} not found.")
            continue

        model = joblib.load(model_path)
        preds = model.predict(X_test)

        # Log-scale Spearman
        rho_log, pval_log = spearmanr(y_test, preds)
        # Original-scale Spearman
        rho_orig, pval_orig = spearmanr(np.expm1(y_test), np.expm1(preds))

        print(f"  {model_name}:")
        print(f"    Spearman rho (log scale):     {rho_log:.4f}")
        print(f"    Spearman rho (original scale): {rho_orig:.4f}")

        if model_name in results:
            results[model_name]["Spearman_rho_log"] = round(float(rho_log), 4)
            results[model_name]["Spearman_pval_log"] = float(pval_log)
            results[model_name]["Spearman_rho_original"] = round(float(rho_orig), 4)
            results[model_name]["Spearman_pval_original"] = float(pval_orig)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Updated {results_path} with Spearman fields for all models.")


if __name__ == "__main__":
    main()

