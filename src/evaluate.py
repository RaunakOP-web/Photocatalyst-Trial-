"""
evaluate.py
Generates comparison reports and all 10 visual artifacts: actual vs predicted,
residuals, SHAP importance, SHAP beeswarm, HER distribution, learning curves,
SHAP dependence plots, and per-material error analysis.
"""

import os
import yaml
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load config
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths = CFG["paths"]
data_cfg = CFG["data"]

def main():
    proc_dir = paths["proc_dir"]
    results_dir = paths["results_dir"]
    models_dir = paths["models_dir"]
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset splits
    X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()
    
    # Align X_test index with df_clean using same split parameters
    df_clean = pd.read_csv(os.path.join(proc_dir, "df_clean.csv"), index_col=0)
    strat_bins = pd.qcut(df_clean["log_HER"], 10, labels=False, duplicates="drop")
    _, _, _, y_test_aligned = train_test_split(
        df_clean[[]], df_clean["log_HER"],
        test_size=data_cfg["test_size"],
        stratify=strat_bins,
        random_state=data_cfg["random_state"]
    )
    X_test.index = y_test_aligned.index
    y_test.index = y_test_aligned.index
    
    # Load best model name and best model
    with open(os.path.join(models_dir, "best_model_name.txt"), "r") as f:
        best_name = f.read().strip()
        
    best_model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    
    # Load training results metrics
    with open(os.path.join(results_dir, "training_results.json"), "r") as f:
        metrics = json.load(f)
        
    # Predict with best model
    preds = best_model.predict(X_test)
    residuals = y_test - preds
    
    # ── 4a. Model Comparison ──
    print("\n--- 4a. Model Comparison Table ---")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=True)
    print(metrics_df.to_string())
    
    # ── Keep 1: Actual vs Predicted (log scale) ──
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.5, edgecolors="k", linewidths=0.3, s=40)
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual log(HER+1)")
    ax.set_ylabel("Predicted log(HER+1)")
    ax.set_title(f"Actual vs Predicted HER (log scale) - {best_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close()
    
    # ── Keep 2: Residuals ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(preds, residuals, alpha=0.4, s=30, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted log(HER+1)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual Plot - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "residuals.png"), dpi=150)
    plt.close()
    
    # ── Keep 3: SHAP feature importance ──
    # Ridge model doesn't support TreeExplainer. Explanations should come from best model.
    # Note: If best model is Ridge, we might fall back to LinearExplainer or XGBoost TreeExplainer.
    # But usually best model is XGBoost or LightGBM. Let's make it robust:
    if best_name in ["XGBoost", "LightGBM"]:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        
        # Plot 3: Importance Bar
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "shap_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
        # Plot 4: Beeswarm
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        # Fallback explanation if Ridge is best (using coef_ as proxy or loading XGBoost to explain)
        # We know XGBoost exists, let's load it just to generate SHAP plots
        alt_model = joblib.load(os.path.join(models_dir, "xgboost_model.joblib"))
        explainer = shap.TreeExplainer(alt_model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "shap_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ── 4b. HER Distribution before/after log transform ──
    raw_her = df_clean[data_cfg["target"]].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw log scale
    ax1.hist(raw_her, bins=30, edgecolor="k", alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("HER (umol/g/h) [log scale]")
    ax1.set_ylabel("Count")
    ax1.set_title("Raw HER (Log X-axis)")
    
    # Log1p scale
    ax2.hist(np.log1p(raw_her), bins=30, edgecolor="k", color="teal", alpha=0.7)
    ax2.set_xlabel("log1p(HER)")
    ax2.set_ylabel("Count")
    ax2.set_title("Log1p-Transformed HER")
    
    fig.suptitle("HER Distribution Before and After Log Transform", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "her_distribution.png"), dpi=150)
    plt.close()
    
    # ── 4c. Learning Curve ──
    print("Generating learning curve...")
    from sklearn.base import clone
    model_for_lc = clone(best_model)
    if hasattr(model_for_lc, "early_stopping_rounds"):
        model_for_lc.set_params(early_stopping_rounds=None)
        
    train_sizes, train_scores, val_scores = learning_curve(
        model_for_lc, X_test, y_test,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5, scoring="r2", n_jobs=-1
    )
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="red", label="Train R²")
    ax.plot(train_sizes, np.mean(val_scores, axis=1), "o-", color="blue", label="Validation R²")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R² Score")
    ax.set_title(f"Learning Curve ({best_name})")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "learning_curve.png"), dpi=150)
    plt.close()
    
    # ── 4d. SHAP dependence plots for top 5 features ──
    if best_name in ["XGBoost", "LightGBM"]:
        # Compute mean absolute shap values to get top features
        vals = np.abs(shap_values).mean(0)
        feature_names = X_test.columns
        top_indices = np.argsort(vals)[::-1][:5]
        
        for i, idx in enumerate(top_indices):
            feature_name = feature_names[idx]
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(feature_name, shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"shap_dependence_{feature_name}.png"), dpi=150, bbox_inches="tight")
            plt.close()
    else:
        # Load XGBoost to generate SHAP dependence if Ridge is best
        alt_model = joblib.load(os.path.join(models_dir, "xgboost_model.joblib"))
        explainer = shap.TreeExplainer(alt_model)
        shap_values_alt = explainer.shap_values(X_test)
        vals = np.abs(shap_values_alt).mean(0)
        feature_names = X_test.columns
        top_indices = np.argsort(vals)[::-1][:5]
        for idx in top_indices:
            feature_name = feature_names[idx]
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(feature_name, shap_values_alt, X_test, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"shap_dependence_{feature_name}.png"), dpi=150, bbox_inches="tight")
            plt.close()
            
    # ── 4e. Per Material Error Analysis ──
    # Join test set predictions with host_material from df_clean
    test_hosts = df_clean.loc[X_test.index, "host_material"].fillna("unknown")
    
    eval_df = pd.DataFrame({
        "host_material": test_hosts,
        "y_true_orig": np.expm1(y_test),
        "y_pred_orig": np.clip(np.expm1(preds), 0, None),
        "y_true_log": y_test,
        "y_pred_log": preds
    })
    
    material_metrics = []
    material_counts = eval_df["host_material"].value_counts()
    valid_materials = material_counts[material_counts >= 5].index.tolist()
    
    for mat in valid_materials:
        mat_sub = eval_df[eval_df["host_material"] == mat]
        mae_val = mean_absolute_error(mat_sub["y_true_orig"], mat_sub["y_pred_orig"])
        r2_val = r2_score(mat_sub["y_true_log"], mat_sub["y_pred_log"])
        material_metrics.append({
            "host_material": mat,
            "Count": len(mat_sub),
            "MAE_umol_g_h": round(mae_val, 2),
            "R2_log": round(r2_val, 4)
        })
        
    mat_metrics_df = pd.DataFrame(material_metrics)
    if not mat_metrics_df.empty:
        mat_metrics_df.to_csv(os.path.join(results_dir, "per_material_metrics.csv"), index=False)
        
        # Plot as horizontal bar chart sorted by MAE descending
        mat_metrics_df_sorted = mat_metrics_df.sort_values(by="MAE_umol_g_h", ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(4, len(mat_metrics_df_sorted) * 0.5)))
        sns.barplot(
            x="MAE_umol_g_h", y="host_material", data=mat_metrics_df_sorted,
            ax=ax, palette="Reds_r", hue="host_material", legend=False
        )
        ax.set_xlabel("MAE (umol/g/h)")
        ax.set_ylabel("Host Material")
        ax.set_title("Prediction MAE by Host Material (Test Set)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "per_material_error.png"), dpi=150)
        plt.close()
    else:
        # Save empty placeholder CSV if no material has >= 5 samples in test set
        pd.DataFrame(columns=["host_material", "Count", "MAE_umol_g_h", "R2_log"]).to_csv(
            os.path.join(results_dir, "per_material_metrics.csv"), index=False
        )
        
    # ── 4f. Residual Normality ──
    stat, p_val = stats.normaltest(residuals)
    print(f"Residual normality test: statistic={stat:.4f}, p-value={p_val:.4e}")

if __name__ == "__main__":
    main()
