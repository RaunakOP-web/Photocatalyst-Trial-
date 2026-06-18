import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from joblib import load
from src.utils.config import get_train_config
from src.utils.io import safe_load_csv, safe_load_joblib
from src.features.interaction_features import add_interaction_features, add_domain_features


def main():
    cfg = get_train_config()
    proc_dir = cfg["paths"]["proc_dir"]
    models_dir = cfg["paths"]["models_dir"]

    # Load filtered training data if it exists, otherwise use original training data
    X_train_path = os.path.join(proc_dir, "X_train_filtered.csv")
    if os.path.exists(X_train_path):
        X_train = safe_load_csv(X_train_path)
    else:
        X_train = safe_load_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test = safe_load_csv(os.path.join(proc_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()

    # Apply target encoding if encoder exists
    encoder_path = os.path.join(models_dir, "target_encoder.joblib")
    cat_cols_path = os.path.join(proc_dir, "cat_cols.joblib")
    if os.path.exists(encoder_path) and os.path.exists(cat_cols_path):
        encoder = safe_load_joblib(encoder_path)
        cat_cols = safe_load_joblib(cat_cols_path)
        X_train[cat_cols] = encoder.transform(X_train[cat_cols])
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    # Add engineered interaction and domain features (same as training pipeline)
    X_train_int, X_test_int = add_interaction_features(X_train, X_test)
    X_train_int, X_test_int = add_domain_features(X_train_int, X_test_int, y_train)

    # Load ensemble model
    ensemble_path = os.path.join(models_dir, "best_model.joblib")
    ensemble = load(ensemble_path)

    # Active features used by the ensemble
    active_features = ensemble.active_features
    X_test_base = X_test_int[active_features].copy()

    # Compute individual model performances
    report_lines = []
    report_lines.append("# Final Performance Report")
    report_lines.append("")
    report_lines.append("| Model | Test R² |")
    report_lines.append("|-------|--------|")
    for name, model in ensemble.models.items():
        preds = model.predict(X_test_base)
        r2 = r2_score(y_test, preds)
        report_lines.append(f"| {name} | {r2:.4f} |")

    # Blended ensemble performance
    blended_preds = ensemble.predict(X_test_int)
    blended_r2 = r2_score(y_test, blended_preds)
    report_lines.append(f"| **Blended Ensemble** | **{blended_r2:.4f}** |")
    report_lines.append("")
    report_lines.append(f"**Target R² ≥ 0.90 achieved:** {'YES' if blended_r2 >= 0.90 else 'NO'}")

    # Write report
    report_path = os.path.join(proc_dir, "final_performance_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Final performance report saved to {report_path}")

if __name__ == "__main__":
    main()
