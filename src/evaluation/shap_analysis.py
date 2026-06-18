import os
import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.utils.logging import setup_logger
from src.utils.config import get_train_config

logger = setup_logger(__name__)
CFG = get_train_config()

def generate_shap_plots(best_model, X_test, y_test, host_materials_test, results_dir, feature_names):
    """Generates all SHAP-related publication figures."""
    logger.info("Computing SHAP values...")
    
    # Extract tree-based base model
    model_to_explain = best_model
    if hasattr(best_model, "models") and best_model.models is not None:
        model_to_explain = best_model.models.get("CatBoost")
    elif hasattr(best_model, "tio2_models") and best_model.tio2_models is not None:
        model_to_explain = best_model.tio2_models.get("CatBoost")
    elif hasattr(best_model, "generalist_blend") and best_model.generalist_blend is not None:
        if hasattr(best_model.generalist_blend, "base_models"):
            model_to_explain = best_model.generalist_blend.base_models.get("CatBoost")
    elif hasattr(best_model, "base_models"):
        model_to_explain = best_model.base_models.get("XGBoost", list(best_model.base_models.values())[0])
        
    explainer = shap.TreeExplainer(model_to_explain)
    shap_values = explainer.shap_values(X_test)

    # Save summary data
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    # Print drivers
    logger.info("Top drivers computed successfully.")
    
    # Beeswarm & Bar plot
    fig = plt.figure(figsize=(8.5, 7.0))
    gs = gridspec.GridSpec(2, 1, hspace=0.35, height_ratios=[1, 1.6])
    
    ax_a = fig.add_subplot(gs[0])
    top_n = min(15, len(sorted_idx))
    top_features = [feature_names[i] for i in sorted_idx[:top_n]]
    top_vals = mean_abs_shap[sorted_idx[:top_n]]
    ax_a.barh(range(top_n), top_vals[::-1], edgecolor="k", linewidth=0.3, alpha=0.9)
    ax_a.set_yticks(range(top_n))
    ax_a.set_yticklabels(top_features[::-1], fontsize=8)
    ax_a.set_xlabel("Mean |SHAP|")
    ax_a.set_title("Global SHAP Feature Importance")
    
    ax_b = fig.add_subplot(gs[1])
    shap.summary_plot(shap_values, X_test, max_display=top_n, show=False, alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
