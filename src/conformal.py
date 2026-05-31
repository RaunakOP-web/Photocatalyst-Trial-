"""
conformal.py
Split conformal prediction for guaranteed marginal coverage.

Uses a held-out calibration set to compute nonconformity scores,
then sets prediction intervals that provably contain the true
value with probability >= 1 - alpha.

Saves:
  data/results/conformal_intervals.csv  -- per-test-point intervals
  data/results/conformal_summary.json   -- coverage + q_hat summary
  models/conformal_model.joblib         -- refitted model + q_hat
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split

PROC_DIR    = "data/processed"
MODELS_DIR  = "models"
RESULTS_DIR = "data/results"
ALPHA = 0.10  # target 90% coverage


def run_conformal():
    """
    Split conformal prediction for guaranteed marginal coverage.
    Uses a held-out calibration set to compute nonconformity scores,
    then sets prediction intervals that provably contain the true
    value with probability >= 1 - alpha.
    """
    print("=" * 60)
    print("SPLIT CONFORMAL PREDICTION")
    print("=" * 60)

    X_train = pd.read_csv(f"{PROC_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{PROC_DIR}/y_train.csv").squeeze()
    X_test  = pd.read_csv(f"{PROC_DIR}/X_test.csv")
    y_test  = pd.read_csv(f"{PROC_DIR}/y_test.csv").squeeze()
    model   = joblib.load(f"{MODELS_DIR}/final_model.joblib") if os.path.exists(
        f"{MODELS_DIR}/final_model.joblib"
    ) else joblib.load(f"{MODELS_DIR}/best_model.joblib")

    print(f"  Training set:  {len(X_train)} samples")
    print(f"  Test set:      {len(X_test)} samples")
    print(f"  Target alpha:  {ALPHA} (coverage target = {1 - ALPHA})")

    # Step 1: Split training set into proper train + calibration
    X_prop, X_cal, y_prop, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=12
    )
    print(f"  Proper train:  {len(X_prop)} samples")
    print(f"  Calibration:   {len(X_cal)} samples")

    # Step 2: Refit model on proper training set only
    print("  Refitting model on proper training set...")
    model.fit(X_prop, y_prop)

    # Step 3: Compute nonconformity scores on calibration set
    cal_preds = model.predict(X_cal)
    scores    = np.abs(y_cal.values - cal_preds)

    # Step 4: Compute conformal quantile
    n = len(scores)
    level = np.ceil((n + 1) * (1 - ALPHA)) / n
    level = min(level, 1.0)
    q_hat = np.quantile(scores, level)
    print(f"  Conformal quantile q_hat = {q_hat:.4f}")

    # Step 5: Apply to test set
    test_preds = model.predict(X_test)
    lower = test_preds - q_hat
    upper = test_preds + q_hat

    # Step 6: Compute empirical coverage on test set
    covered  = ((y_test.values >= lower) & (y_test.values <= upper))
    coverage = covered.mean()
    print(f"  Empirical coverage on test set: {coverage:.4f}")
    print(f"  Target coverage:                {1 - ALPHA:.4f}")
    print(f"  Mean interval width (log):      {(upper - lower).mean():.4f}")
    print(f"  Mean interval width (original): "
          f"{(np.expm1(upper) - np.expm1(lower)).mean():,.0f} umol/g/h")

    # Save conformal results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = pd.DataFrame({
        "y_true_log": y_test.values,
        "y_pred_log": test_preds,
        "lower_log":  lower,
        "upper_log":  upper,
        "y_true_her": np.expm1(y_test.values),
        "y_pred_her": np.expm1(test_preds),
        "lower_her":  np.expm1(lower),
        "upper_her":  np.expm1(upper),
        "covered":    covered,
    })
    results.to_csv(f"{RESULTS_DIR}/conformal_intervals.csv", index=False)

    summary = {
        "method":               "split_conformal",
        "alpha":                ALPHA,
        "target_coverage":      1 - ALPHA,
        "empirical_coverage":   round(float(coverage), 4),
        "q_hat":                round(float(q_hat), 4),
        "mean_width_log":       round(float((upper - lower).mean()), 4),
        "mean_width_her_umol":  round(float((np.expm1(upper) - np.expm1(lower)).mean()), 0),
        "n_calibration":        n,
        "n_test":               len(y_test),
        "coverage_guarantee":   "marginal, finite-sample",
    }
    with open(f"{RESULTS_DIR}/conformal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save q_hat for use in virtual screening
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"q_hat": q_hat, "model": model},
                f"{MODELS_DIR}/conformal_model.joblib")

    print("\n  Conformal calibration complete.")
    print(f"  Results saved to {RESULTS_DIR}/conformal_intervals.csv")
    print(f"  Summary saved to {RESULTS_DIR}/conformal_summary.json")
    print(f"  Model + q_hat saved to {MODELS_DIR}/conformal_model.joblib")
    return summary


if __name__ == "__main__":
    run_conformal()
