import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

from src.utils.config import get_train_config
from src.utils.logging import setup_logger
from src.utils.io import safe_load_csv, safe_load_joblib, safe_save_json
from src.features.interaction_features import add_interaction_features, add_domain_features
from src.evaluation.metrics import calculate_metrics, calculate_original_scale_metrics
from src.evaluation.diagnostics import evaluate_residual_normality
from src.evaluation.shap_analysis import generate_shap_plots

logger = setup_logger(__name__)
CFG = get_train_config()

def main():
    proc_dir = CFG["paths"]["proc_dir"]
    results_dir = CFG["paths"]["results_dir"]
    models_dir = CFG["paths"]["models_dir"]
    
    # Load dataset splits
    X_test = safe_load_csv(os.path.join(proc_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()
    X_train = safe_load_csv(os.path.join(proc_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    
    # Target encode categories using the saved training phase encoder
    encoder_path = os.path.join(models_dir, "target_encoder.joblib")
    cat_cols_path = os.path.join(proc_dir, "cat_cols.joblib")
    if os.path.exists(encoder_path) and os.path.exists(cat_cols_path):
        encoder = safe_load_joblib(encoder_path)
        cat_cols = safe_load_joblib(cat_cols_path)
        X_train[cat_cols] = encoder.transform(X_train[cat_cols])
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    
    # Add interaction features
    X_train, X_test = add_interaction_features(X_train, X_test)
    X_train, X_test = add_domain_features(X_train, X_test, y_train)
    
    # Load best model
    best_model = safe_load_joblib(os.path.join(models_dir, "best_model.joblib"))
    active_features = best_model.active_features

    
    X_test_base = X_test[active_features].copy()
    
    # Predict
    preds_log = best_model.predict(X_test_base)
    
    # Calculate log & original scale metrics
    log_metrics = calculate_metrics(y_test, preds_log)
    orig_metrics = calculate_original_scale_metrics(y_test, preds_log)
    
    logger.info(f"Test R2 (Log Scale): {log_metrics['R2']:.4f}")
    logger.info(f"Test R2 (Orig Scale): {orig_metrics['R2_orig']:.4f}")
    
    # Residual diagnostic
    evaluate_residual_normality(y_test, preds_log, results_dir)
    
    # Save training results json
    report = {
        "Blending_Ensemble": {
            "Test_R2_log": log_metrics["R2"],
            "Test_R2_original": orig_metrics["R2_orig"],
            "Test_MAE_log": log_metrics["MAE"],
            "Test_MAE_umol_g_h": orig_metrics["MAE_orig"],
            "Test_RMSE_umol_g_h": orig_metrics["RMSE_orig"],
            "CV_R2_mean": 0.0,
            "CV_R2_std": 0.0,
            "LOMO_CV_R2_mean": 0.0,
            "LOMO_CV_R2_std": 0.0,
            "composite_selection_score": log_metrics["R2"]
        }
    }
    safe_save_json(report, os.path.join(results_dir, "training_results.json"))
    
    # Extract material group mappings for SHAP
    df_clean = pd.read_csv(os.path.join(proc_dir, "df_clean.csv"), index_col=0)
    strat_bins = pd.qcut(df_clean["log_HER"], 10, labels=False, duplicates="drop")
    _, _, _, y_test_aligned = train_test_split(
        df_clean[[]], df_clean["log_HER"],
        test_size=CFG["data"]["test_size"],
        stratify=strat_bins,
        random_state=CFG["data"]["random_state"]
    )
    host_materials_test = df_clean.loc[y_test_aligned.index, "host_material"].fillna("unknown").values
    
    # SHAP
    generate_shap_plots(
        best_model=best_model,
        X_test=X_test_base,
        y_test=y_test,
        host_materials_test=host_materials_test,
        results_dir=results_dir,
        feature_names=active_features
    )
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
