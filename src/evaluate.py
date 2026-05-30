"""
evaluate.py
Generates SHAP feature importance plots, actual vs predicted plots,
and a full evaluation report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import json
import os

PROC_DIR    = "data/processed"
MODELS_DIR  = "models"
RESULTS_DIR = "data/results"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_test = pd.read_csv(f"{PROC_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROC_DIR}/y_test.csv").squeeze()
    model  = joblib.load(f"{MODELS_DIR}/best_model.joblib")

    preds    = model.predict(X_test)
    her_pred = np.expm1(preds)
    her_true = np.expm1(y_test)

    # ── Plot 1: Actual vs Predicted (log scale) ──────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.5, edgecolors="k", linewidths=0.3, s=40)
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual log(HER+1)")
    ax.set_ylabel("Predicted log(HER+1)")
    ax.set_title("Actual vs Predicted HER (log scale)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/actual_vs_predicted.png", dpi=150)
    plt.close()
    print("Saved actual_vs_predicted.png")

    # ── Plot 2: Residuals ────────────────────────────────
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(preds, residuals, alpha=0.4, s=30, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted log(HER+1)")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/residuals.png", dpi=150)
    plt.close()
    print("Saved residuals.png")

    # ── Plot 3: SHAP feature importance ─────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_importance.png")

    # ── Plot 4: SHAP beeswarm ────────────────────────────
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_beeswarm.png")

    print(f"\nAll evaluation plots saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
