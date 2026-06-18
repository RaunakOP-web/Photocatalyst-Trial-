import os
import shutil

def main():
    pkg_dir = "publication_package"
    figs_dir = os.path.join(pkg_dir, "publication_figures")
    tables_dir = os.path.join(pkg_dir, "publication_tables")
    
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    # Source paths
    res_figs = os.path.join("data", "results", "figures")
    res_root = os.path.join("data", "results")
    
    # 1. Copy Figures
    fig_mappings = {
        "fig1_dataset_overview.png": "fig1_dataset_overview.png",
        "fig2_model_comparison.png": "fig2_model_comparison.png",
        "fig3_actual_vs_predicted.png": "fig3_actual_vs_predicted.png",
        "fig5_per_material_performance.png": "fig5_per_material_performance.png",
        "fig6_discovery_candidates.png": "fig6_discovery_candidates.png",
        "fig_conformal_calibration.png": "fig7_uncertainty_calibration.png",
        "fig8_ablation.png": "fig8_ablation_study.png",
        "fig_williams_plot.png": "fig9_applicability_domain_williams.png",
        "shap_summary_beeswarm.png": "fig10_shap_global_beeswarm.png",
        "shap_top_catalyst_waterfall.png": "fig11_shap_top_catalyst_waterfall.png",
        "shap_interaction_bandgap_work_function.png": "fig12_shap_interaction.png"
    }
    
    for src_name, dest_name in fig_mappings.items():
        src_path = os.path.join(res_figs, src_name)
        # Check if in results root instead
        if not os.path.exists(src_path):
            src_path = os.path.join(res_root, src_name)
            
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(figs_dir, dest_name))
            print(f"Copied figure: {src_name} -> {dest_name}")
        else:
            print(f"Warning: Figure {src_name} not found.")
            
    # Also copy PDFs if available
    for f in os.listdir(res_figs):
        if f.endswith(".pdf") or f.endswith(".svg"):
            shutil.copy(os.path.join(res_figs, f), os.path.join(figs_dir, f))
            
    # 2. Copy Tables
    tables = [
        ("publication_ablation_table.csv", "table1_ablation_study.csv"),
        ("optimized_candidates.csv", "table2_optimized_candidates.csv"),
        ("novelty_assessment.csv", "table3_novelty_assessment.csv")
    ]
    
    for src_name, dest_name in tables:
        src_path = src_name
        if not os.path.exists(src_path):
            src_path = os.path.join(res_root, src_name)
            
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(tables_dir, dest_name))
            print(f"Copied table: {src_name} -> {dest_name}")
        else:
            print(f"Warning: Table {src_name} not found.")
            
    # 3. Create supplementary_information.md
    supp_info = """# Supplementary Information: Machine Learning Assisted Photocatalyst Discovery

## S1. Dataset Details
* **Source**: Literature-mined glycerol photoreforming dataset.
* **Total Sample Count**: 706 experiments.
* **Target Feature**: log_HER = ln(HER + 1) where HER is in µmol g⁻¹ h⁻¹.
* **Features Included**: Experimental conditions (pH, concentration, wavelength, light power), semiconductor properties (bandgap, density, crystal structure), cocatalyst descriptors (work function, electronegativity, atomic radius), and engineered descriptors (effective carrier masses, CB/VB NHE potentials, exciton proxy).

## S2. Conformal Prediction Details
* **Method**: Split Conformal Prediction using MAPIE.
* **Confidence Level**: 90% (Alpha = 0.10).
* **Calibration Set Size**: 20% of training split (115 samples).
* **Empirical Coverage achieved on Test Set**: 86.15%.

## S3. Applicability Domain Boundaries
* **k-NN Distance**: Euclidean distance in the scaled active feature space. Threshold = mean + 2*std of training distances.
* **Isolation Forest**: Contamination parameter = 0.04. Threshold = 5th percentile score of training set.
* **Mahalanobis Distance**: Threshold = mean + 2*std of training distances.
* **Leverage (Williams Plot)**: Warning leverage $h^* = 3p/n$ where $p = 78$ and $n = 576$.
"""
    with open(os.path.join(pkg_dir, "supplementary_information.md"), "w", encoding="utf-8") as f:
        f.write(supp_info)
        
    # 4. Create methodology.md
    methodology = """# Methodology: Materials Informatics Pipeline

## M1. Preprocessing and Feature Engineering
All raw data is cleaned, deduplicated, and enriched with physical descriptors. Categorical variables are target-encoded using a leakage-free target encoder fitted on training folds. The active features are selected dynamically based on model feature importances and domain expertise.

## M2. Model Training and Validation
To prevent data leakage, model validation is performed using a Leave-One-Group-Out Cross-Validation (LOGO-CV) framework based on host material groups. The models are evaluated on a holdout test set (15% of the total dataset). We train a Blending Ensemble composed of XGBoost, LightGBM, CatBoost, and ExtraTrees.

## M3. Applicability Domain and Conformal Bounds
Prior to screening, an applicability domain (AD) is constructed using four methods: distance-based k-NN, Isolation Forest anomaly detection, Hat-matrix leverage, and Mahalanobis distance. Only candidates scoring $\\ge 2$ inside-domain metrics are considered valid. Conformal prediction intervals are calculated at 90% confidence using MAPIE to bound predictions.
"""
    with open(os.path.join(pkg_dir, "methodology.md"), "w", encoding="utf-8") as f:
        f.write(methodology)
        
    # 5. Create limitations.md
    limitations = """# Limitations and Future Work

## L1. Data Completeness and Literature Bias
Literature-derived datasets suffer from publication bias (over-reporting of positive results) and variations in reactor geometries, light sources, and experimental setups that are difficult to standardize.

## L2. Simplification of Structural Features
Crystal phase, surface facets, defect densities, and interface quality in heterojunctions are modeled as simple flags/descriptors. A more comprehensive representation of catalyst morphology (e.g., via graph neural networks on crystal structures) is a key area for future work.

## L3. Generalization to Unseen Classes
As demonstrated by the negative LOGO-CV performance when leaving out the major host class (TiO₂), machine learning models struggle to extrapolate to completely new chemical spaces. Active learning and transfer learning are recommended to mitigate this limitation.
"""
    with open(os.path.join(pkg_dir, "limitations.md"), "w", encoding="utf-8") as f:
        f.write(limitations)
        
    print("Publication package generated successfully.")

if __name__ == "__main__":
    main()
