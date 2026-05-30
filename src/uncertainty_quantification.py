"""
uncertainty_quantification.py
Phase 5 — Advanced Ensemble Uncertainty Quantification

Implements THREE uncertainty methods:
  1. Bootstrap ensemble (BaggingRegressor over LightGBM/XGBoost)
  2. Conformal prediction intervals (split conformal, coverage-guaranteed)
  3. Model disagreement (spread across XGBoost, LightGBM, Ridge)

Saves:
  data/results/uncertainty_report.csv  — row-level CI table for the test set
  data/results/calibration_curve.png   — empirical vs nominal coverage
  data/results/uncertainty_distribution.png — CI width histogram by material
"""

import os
import yaml
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths = CFG["paths"]
data_cfg = CFG["data"]
PROC = paths["proc_dir"]
RES = paths["results_dir"]
MOD = paths["models_dir"]


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ensemble(
    X_train, y_train, X_test,
    n_estimators: int = 100,
    alpha: float = 0.10,
    random_state: int = 42
):
    """
    Train n_estimators bootstrap resampled LightGBM models.
    Returns (p50, p_lo, p_hi) in log-HER space.
    """
    print(f"  Training {n_estimators}-member bootstrap ensemble...")
    lgb_params = json.load(
        open(os.path.join(RES, "best_params_LightGBM.json"))
    )
    lgb_params["verbose"] = -1
    lgb_params["random_state"] = random_state

    rng = np.random.default_rng(random_state)
    preds_matrix = np.zeros((n_estimators, len(X_test)))

    for i in range(n_estimators):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        X_b = X_train.iloc[idx]
        y_b = y_train.iloc[idx]
        m = LGBMRegressor(**lgb_params)
        m.fit(X_b, y_b)
        preds_matrix[i] = m.predict(X_test)

    lo = np.percentile(preds_matrix, 100 * alpha / 2, axis=0)
    hi = np.percentile(preds_matrix, 100 * (1 - alpha / 2), axis=0)
    median = np.median(preds_matrix, axis=0)
    return median, lo, hi, preds_matrix


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def conformal_prediction(best_model, X_train, y_train, X_test, alpha: float = 0.10):
    """
    Split-conformal prediction interval.
    Uses a 20% calibration hold-out from training data.
    Returns (y_hat, lo, hi) for the test set.
    """
    print("  Computing conformal prediction intervals...")
    X_cal_fit, X_cal, y_cal_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42
    )
    # Conformity scores on calibration set
    cal_preds = best_model.predict(X_cal)
    nonconformity = np.abs(y_cal.values - cal_preds)
    # q-level threshold
    q = np.ceil((1 - alpha) * (len(nonconformity) + 1)) / len(nonconformity)
    q = min(q, 1.0)
    threshold = np.quantile(nonconformity, q)

    y_hat = best_model.predict(X_test)
    lo = y_hat - threshold
    hi = y_hat + threshold
    return y_hat, lo, hi, threshold


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DISAGREEMENT
# ─────────────────────────────────────────────────────────────────────────────

def model_disagreement(X_test, models_dir):
    """
    Spread of predictions across XGBoost, LightGBM, Ridge.
    Returns std of predictions as an uncertainty proxy.
    """
    print("  Computing model disagreement uncertainty...")
    preds = {}
    for name in ["xgboost", "lightgbm", "ridge"]:
        path = os.path.join(models_dir, f"{name}_model.joblib")
        if os.path.exists(path):
            m = joblib.load(path)
            preds[name] = m.predict(X_test)
    if not preds:
        return np.zeros(len(X_test))
    arr = np.stack(list(preds.values()), axis=0)
    return arr.std(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION CURVE
# ─────────────────────────────────────────────────────────────────────────────

def calibration_curve(preds_matrix, y_test, results_dir):
    """
    Plot empirical vs nominal coverage for the bootstrap ensemble.
    A well-calibrated model should lie near the diagonal.
    """
    alphas = np.linspace(0.05, 0.95, 19)
    empirical = []
    for a in alphas:
        lo = np.percentile(preds_matrix, 100 * a / 2, axis=0)
        hi = np.percentile(preds_matrix, 100 * (1 - a / 2), axis=0)
        covered = ((y_test.values >= lo) & (y_test.values <= hi)).mean()
        empirical.append(covered)

    nominal = 1 - alphas
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(nominal, empirical, "o-", color="#2e86ab", lw=2, ms=6, label="Bootstrap ensemble")
    ax.set_xlabel("Nominal coverage (1 - alpha)", fontsize=12)
    ax.set_ylabel("Empirical coverage", fontsize=12)
    ax.set_title("Uncertainty Calibration Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "calibration_curve.png"), dpi=300)
    plt.close()
    print(f"  Saved calibration_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# CI WIDTH DISTRIBUTION BY MATERIAL
# ─────────────────────────────────────────────────────────────────────────────

def uncertainty_distribution(ci_width, host_material, results_dir):
    df_plot = pd.DataFrame({"ci_width_log": ci_width, "material": host_material})
    order = df_plot.groupby("material")["ci_width_log"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, max(4, len(order) * 0.55)))
    palette = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(order)))

    for i, mat in enumerate(order):
        vals = df_plot[df_plot["material"] == mat]["ci_width_log"]
        ax.barh(i, vals.median(), xerr=vals.std(), color=palette[i],
                capsize=4, alpha=0.85, edgecolor="k", linewidth=0.5)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("90% CI Width (log(HER+1) units)", fontsize=11)
    ax.set_title("Prediction Uncertainty by Host Material (Test Set)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "uncertainty_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved uncertainty_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RES, exist_ok=True)

    print("Phase 5 - Uncertainty Quantification")

    # Load data
    X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv")).squeeze()

    df_clean = pd.read_csv(os.path.join(PROC, "df_clean.csv"), index_col=0)
    strat_bins = pd.qcut(df_clean["log_HER"], 10, labels=False, duplicates="drop")
    _, _, _, y_test_al = train_test_split(
        df_clean[[]], df_clean["log_HER"],
        test_size=data_cfg["test_size"],
        stratify=strat_bins,
        random_state=data_cfg["random_state"]
    )
    test_idx = y_test_al.index
    train_idx = df_clean.index.difference(test_idx)
    X_test.index = test_idx
    y_test.index = test_idx
    X_train.index = train_idx
    y_train.index = train_idx

    host_test = df_clean.loc[test_idx, "host_material"].fillna("unknown")

    best_model = joblib.load(os.path.join(MOD, "best_model.joblib"))

    # ── Method 1: Bootstrap ──────────────────────────────────────────────────
    boot_median, boot_lo, boot_hi, preds_matrix = bootstrap_ensemble(
        X_train, y_train, X_test, n_estimators=200, alpha=0.10
    )
    boot_ci_width = boot_hi - boot_lo

    # ── Method 2: Conformal ──────────────────────────────────────────────────
    conf_preds, conf_lo, conf_hi, conf_threshold = conformal_prediction(
        best_model, X_train, y_train, X_test, alpha=0.10
    )
    conf_ci_width = conf_hi - conf_lo

    # ── Method 3: Model Disagreement ─────────────────────────────────────────
    disagreement_std = model_disagreement(X_test, MOD)

    # ── Assemble report ──────────────────────────────────────────────────────
    report = pd.DataFrame({
        "y_true_log":        y_test.values,
        "y_true_umol":       np.expm1(y_test.values),
        "boot_median_log":   boot_median,
        "boot_p05_log":      boot_lo,
        "boot_p95_log":      boot_hi,
        "boot_ci_width_log": boot_ci_width,
        "conf_pred_log":     conf_preds,
        "conf_lo_log":       conf_lo,
        "conf_hi_log":       conf_hi,
        "conf_ci_width_log": conf_ci_width,
        "model_disagree_std": disagreement_std,
        "host_material":     host_test.values,
        # Coverage flags
        "boot_covered":  ((y_test.values >= boot_lo) & (y_test.values <= boot_hi)).astype(int),
        "conf_covered":  ((y_test.values >= conf_lo) & (y_test.values <= conf_hi)).astype(int),
    })
    report.to_csv(os.path.join(RES, "uncertainty_report.csv"), index=False)

    # ── Summary stats ────────────────────────────────────────────────────────
    boot_coverage = report["boot_covered"].mean()
    conf_coverage = report["conf_covered"].mean()
    print(f"\n  Bootstrap 90% CI empirical coverage : {boot_coverage:.3f} (target >= 0.90)")
    print(f"  Conformal 90% CI empirical coverage  : {conf_coverage:.3f} (target >= 0.90)")
    print(f"  Conformal threshold                  : {conf_threshold:.4f} log units")
    print(f"  Mean bootstrap CI width              : {boot_ci_width.mean():.4f} log units")
    print(f"  Mean model disagreement (std)        : {disagreement_std.mean():.4f} log units")

    # Save summary
    uq_summary = {
        "bootstrap_n_estimators": 200,
        "bootstrap_alpha": 0.10,
        "bootstrap_empirical_coverage": round(float(boot_coverage), 4),
        "bootstrap_mean_ci_width_log": round(float(boot_ci_width.mean()), 4),
        "conformal_alpha": 0.10,
        "conformal_threshold_log": round(float(conf_threshold), 4),
        "conformal_empirical_coverage": round(float(conf_coverage), 4),
        "mean_model_disagreement_std": round(float(disagreement_std.mean()), 4),
    }
    with open(os.path.join(RES, "uncertainty_summary.json"), "w") as fh:
        json.dump(uq_summary, fh, indent=2)

    # ── Plots ────────────────────────────────────────────────────────────────
    calibration_curve(preds_matrix, y_test, RES)
    uncertainty_distribution(boot_ci_width, host_test.values, RES)

    print("\nPhase 5 complete - uncertainty outputs saved to data/results/")


if __name__ == "__main__":
    main()
