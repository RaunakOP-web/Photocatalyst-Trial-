import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr, pearsonr
from catboost import CatBoostRegressor, Pool
from src.utils.config import get_train_config
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def run_descriptor_validation():
    """Compute coverage, variance, correlations, mutual information, and SHAP importance
    for the newly added physics‑informed descriptors. Generates a markdown report
    `descriptor_validation_report.md` in the processed data directory.
    """
    # Load processed data
    cfg_train = get_train_config()
    proc_dir = cfg_train["paths"]["proc_dir"]
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    # Ensure alignment by position; assume X_train rows correspond to y_train order

    # Descriptors introduced by helper scripts
    new_descriptors = [
        "cb_potential_nhe",
        "vb_potential_nhe",
        "cb_overpotential_h2",
        "vb_overpotential_glycerol",
        "cocat_dg_h_proxy",
        "cocat_h_binding_strength",
        "cocat_sabatier_distance",
        "electron_effective_mass",
        "hole_effective_mass",
        "mobility_ratio",
        "carrier_transport_score",
        "carrier_mobility_uncertainty",
        "exciton_binding_proxy",
        "charge_separation_score",
        "exciton_binding_uncertainty",
        "surface_area_normalized_HER",
        "loading_surface_ratio",
        "active_site_density_proxy",
        "surface_reactivity_score",
    ]
    new_descriptors = [c for c in new_descriptors if c in X_train.columns]

    report_rows = []
    for col in new_descriptors:
        series = X_train[col]
        coverage = series.notna().mean() * 100
        variance = series.var()
        std_dev = series.std()
        mask = ~np.isnan(series.values)
        if mask.sum() == 0:
            pearson_corr = np.nan
            spearman_corr = np.nan
        else:
            pearson_corr, _ = pearsonr(series.values[mask], y_train.values[mask])
            spearman_corr, _ = spearmanr(series.values[mask], y_train.values[mask])
        mi = mutual_info_regression(series.values.reshape(-1, 1), y_train.values, discrete_features=False, random_state=0)[0]
        report_rows.append({
            "Descriptor": col,
            "Coverage%": f"{coverage:.1f}",
            "Variance": f"{variance:.4f}",
            "StdDev": f"{std_dev:.4f}",
            "Pearson": f"{pearson_corr:.3f}" if not np.isnan(pearson_corr) else "nan",
            "Spearman": f"{spearman_corr:.3f}" if not np.isnan(spearman_corr) else "nan",
            "MI": f"{mi:.4f}",
        })

    # Quick CatBoost model for SHAP importance
    # Replace placeholder strings with NaN and convert to numeric
    X_train = X_train.replace('missing', np.nan)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_train = X_train.fillna(X_train.median())
    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )
    model.fit(X_train, y_train)
    shap_vals = model.get_feature_importance(Pool(X_train, y_train), type="ShapValues")
    shap_vals = shap_vals[:, :-1]
    shap_importance = np.mean(np.abs(shap_vals), axis=0)
    shap_dict = dict(zip(X_train.columns, shap_importance))

    for row in report_rows:
        row["SHAP_Importance"] = f"{shap_dict.get(row['Descriptor'], 0.0):.4f}"
        cov = float(row["Coverage%"])
        var = float(row["Variance"])
        if cov < 40.0:
            comment = "Low coverage – consider dropping"
        elif var == 0.0:
            comment = "Zero variance – drop"
        else:
            comment = "OK"
        row["Comments"] = comment

    md_lines = [
        "# Descriptor Validation Report",
        "",
        "| Descriptor | Coverage% | Variance | StdDev | Pearson | Spearman | MI | SHAP_Importance | Comments |",
        "|------------|----------|----------|--------|---------|----------|----|----------------|----------|",
    ]
    for r in report_rows:
        md_lines.append(
            f"| {r['Descriptor']} | {r['Coverage%']} | {r['Variance']} | {r['StdDev']} | {r['Pearson']} | {r['Spearman']} | {r['MI']} | {r['SHAP_Importance']} | {r['Comments']} |"
        )
    md_content = "\n".join(md_lines)
    report_path = os.path.join(proc_dir, "descriptor_validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Descriptor validation report saved to {report_path}")
    # Identify descriptors to drop based on comments
    drop_descriptors = [row['Descriptor'] for row in report_rows if row['Comments'] != "OK"]
    if drop_descriptors:
        logger.info(f"Dropping descriptors: {drop_descriptors}")
        X_train_filtered = X_train.drop(columns=drop_descriptors, errors='ignore')
        # Load X_test similarly
        X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
        X_test_filtered = X_test.drop(columns=drop_descriptors, errors='ignore')
        # Save filtered datasets
        X_train_filtered.to_csv(os.path.join(proc_dir, "X_train_filtered.csv"), index=False)
        X_test_filtered.to_csv(os.path.join(proc_dir, "X_test.csv"), index=False)  # overwrite test with filtered version
        logger.info("Filtered datasets saved: X_train_filtered.csv and X_test.csv")
    else:
        logger.info("No descriptors dropped; all passed criteria.")

if __name__ == "__main__":
    run_descriptor_validation()
