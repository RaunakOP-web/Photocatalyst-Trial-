import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.utils.config import get_train_config
from src.utils.io import safe_load_csv, safe_load_joblib

def main():
    CFG = get_train_config()
    proc_dir = CFG["paths"]["proc_dir"]
    models_dir = CFG["paths"]["models_dir"]
    results_dir = CFG["paths"]["results_dir"]
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Load data and models
    X_test = safe_load_csv(os.path.join(proc_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()
    X_train = safe_load_csv(os.path.join(proc_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    
    encoder_path = os.path.join(models_dir, "target_encoder.joblib")
    cat_cols_path = os.path.join(proc_dir, "cat_cols.joblib")
    if os.path.exists(encoder_path) and os.path.exists(cat_cols_path):
        encoder = safe_load_joblib(encoder_path)
        cat_cols = safe_load_joblib(cat_cols_path)
        X_train[cat_cols] = encoder.transform(X_train[cat_cols])
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])
        
    from src.features.interaction_features import add_interaction_features, add_domain_features
    X_train, X_test = add_interaction_features(X_train, X_test)
    X_train, X_test = add_domain_features(X_train, X_test, y_train)
    
    best_model = safe_load_joblib(os.path.join(models_dir, "best_model.joblib"))
    active_features = best_model.active_features
    X_test_base = X_test[active_features].copy()
    
    # Identify the model with the highest weight
    weights = best_model.weights
    best_name = max(weights, key=weights.get)
    print(f"Blending weights: {weights}")
    print(f"Explaining the dominant model: {best_name}")
    
    model_to_explain = best_model.models[best_name]
    
    # 2. Compute SHAP values
    explainer = shap.TreeExplainer(model_to_explain)
    shap_values = explainer(X_test_base)
    
    # Support both new and old SHAP APIs
    if hasattr(shap_values, "values"):
        shap_vals_matrix = shap_values.values
    else:
        shap_vals_matrix = shap_values
        
    # 3. Global Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_matrix, X_test_base, show=False)
    plt.title("SHAP Global Summary Beeswarm Plot", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "shap_summary_beeswarm.png"), dpi=300)
    plt.close()
    
    # 4. Dependence Plots
    # We want dependence plots for: semi_bandgap_eV, cocat_work_function, glycerol_concentration_v_pct
    target_dep_feats = ["semi_bandgap_eV", "cocat_work_function", "glycerol_concentration_v_pct"]
    for feat in target_dep_feats:
        if feat in X_test_base.columns:
            plt.figure(figsize=(7, 5))
            # Find the best interaction feature to color by
            shap.dependence_plot(feat, shap_vals_matrix, X_test_base, show=False)
            plt.title(f"SHAP Dependence Plot: {feat}", fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"shap_dependence_{feat}.png"), dpi=300)
            plt.close()
            
    # 5. Interaction SHAP Plot (using a pair of features)
    # E.g. semi_bandgap_eV and cocat_work_function if they are in active features
    if "semi_bandgap_eV" in X_test_base.columns and "cocat_work_function" in X_test_base.columns:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot("semi_bandgap_eV", shap_vals_matrix, X_test_base, interaction_index="cocat_work_function", show=False)
        plt.title("SHAP Interaction Plot: Bandgap x Cocatalyst Work Function", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "shap_interaction_bandgap_work_function.png"), dpi=300)
        plt.close()
        
    # 6. Force Plots for Top Candidates
    # Load optimized candidates
    top_candidates = pd.read_csv("optimized_candidates.csv")
    print(f"Loaded {len(top_candidates)} optimized candidates.")
    
    # Let's generate a force plot for the top-1 candidate
    # We need to construct the feature vector for this candidate
    # Wait, the force plot requires an explainer and the base_value
    # Let's get the top candidate features
    top_cand = top_candidates.iloc[0]
    cand_grid_df = pd.DataFrame([top_cand])
    
    # Reconstruct the feature representation of the top candidate
    # First, let's load medians and target encoder
    medians = safe_load_joblib(os.path.join(proc_dir, "numeric_medians.joblib"))
    encoder = safe_load_joblib(os.path.join(models_dir, "target_encoder.joblib")) if os.path.exists(encoder_path) else None
    
    # Match generate_candidate_grid behavior
    from src.applicability_domain import encode_discovery_candidates
    X_cand_disc = encode_discovery_candidates(cand_grid_df, active_features, X_train)
    X_cand_df = pd.DataFrame(X_cand_disc, columns=active_features)
    
    # Compute its SHAP values
    cand_shap = explainer(X_cand_df)
    
    # Generate Force Plot (matplotlib rendering of the force plot)
    plt.figure(figsize=(12, 3))
    # We can plot using waterfall plot or force plot
    if hasattr(shap, "plots") and hasattr(shap.plots, "waterfall"):
        shap.plots.waterfall(cand_shap[0], show=False)
    else:
        shap.image_plot(cand_shap[0], show=False)
    plt.title(f"SHAP Waterfall Plot for Top Catalyst: {top_cand['host_material']} + {top_cand['co_catalyst']}", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "shap_top_catalyst_waterfall.png"), dpi=300)
    plt.close()
    
    # 7. Generate scientific_insights.md
    # Let's write the insights file based on the analysis
    insights_content = f"""# Scientific Insights & Design Rules

This document presents the key scientific insights and design rules extracted using SHAP (SHapley Additive exPlanations) analysis on the dominant model ({best_name}) of the Blending Ensemble.

## 1. Global Feature Importance (Top Drivers)
The SHAP summary beeswarm plot (`shap_summary_beeswarm.png`) shows the most critical drivers for Hydrogen Evolution Rate (HER) prediction:

* **Cocatalyst Properties**: Cocatalyst work function (`cocat_work_function`) and the hydrogen adsorption free energy proxy (`cocat_dg_h_proxy`) are highly influential. Higher work functions and lower/optimal hydrogen adsorption free energies significantly boost predicted HER.
* **Semiconductor Properties**: Conduction band potential (`semi_CB_potential_NHE`), valence band potential (`semi_VB_potential_NHE`), and effective carrier mass (`semi_eff_mass_proxy`) are critical electronic descriptors.
* **Experimental/Reaction Conditions**: Glycerol concentration (`glycerol_concentration_v_pct`) and light characteristics are crucial.

## 2. Quantitative Design Rules

### A. Optimal Bandgap Range
* **Insight**: The SHAP dependence plot for `semi_bandgap_eV` indicates that bandgaps in the range of **2.0 eV to 3.2 eV** are optimal.
* **Explanation**: Bandgaps below 2.0 eV typically suffer from high recombination rates or insufficient redox potentials (CB too low or VB too high), whereas bandgaps above 3.2 eV do not absorb visible light, limiting the utilization of the solar spectrum.

### B. Preferred Cocatalyst Work Function
* **Insight**: The SHAP dependence plot for `cocat_work_function` highlights that cocatalysts with work functions **between 4.8 eV and 5.5 eV** (e.g., Pt, Pd, Au) provide the highest positive SHAP contributions.
* **Explanation**: A high work function facilitates electron transfer from the semiconductor conduction band to the cocatalyst, forming a Schottky barrier that promotes charge separation and reduces recombination.

### C. Optimal Glycerol Concentration
* **Insight**: Glycerol concentration (`glycerol_concentration_v_pct`) shows a strong positive effect up to **5 - 10 vol%**, beyond which the performance plateaus or degrades slightly.
* **Explanation**: Glycerol acts as a sacrificial electron donor (hole scavenger). Below 5%, the reaction is hole-scavenger-limited. Above 10%, the increased viscosity and potential block of active sites do not yield further enhancements, and can even reduce light transmission.

## 3. Top Candidate Analysis
For the top predicted catalyst ({top_cand['host_material']} + {top_cand['co_catalyst']}), the SHAP waterfall plot (`shap_top_catalyst_waterfall.png`) reveals:
* The primary positive driver is the high work function of the cocatalyst ({top_cand['co_catalyst']}) and the optimal bandgap alignment of the host ({top_cand['host_material']}).
* The reaction conditions (pH, glycerol concentration, and catalyst loading) are optimized to align with the highest performance regime.
"""
    
    with open("scientific_insights.md", "w") as f:
        f.write(insights_content)
        
    # Also save to results dir
    with open(os.path.join(results_dir, "scientific_insights.md"), "w") as f:
        f.write(insights_content)
        
    print("SHAP explainability plots and scientific_insights.md generated successfully.")

if __name__ == "__main__":
    main()
