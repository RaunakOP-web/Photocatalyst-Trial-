"""
predict.py
Loads candidate catalysts, validates features schema, applies target encoder
and imputation, predicts HER values, and estimates bootstrap confidence intervals.
"""

import os
import yaml
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.material_features import add_physical_features

# Load config
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths = CFG["paths"]
data_cfg = CFG["data"]

def predict(input_path, output_path=None, bootstrap_n=100):
    proc_dir = paths["proc_dir"]
    models_dir = paths["models_dir"]
    results_dir = paths["results_dir"]
    
    # 1. Load pipelines artifacts
    feature_list = joblib.load(os.path.join(proc_dir, "feature_list.joblib"))
    best_model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    
    with open(os.path.join(models_dir, "best_model_name.txt"), "r") as f:
        best_name = f.read().strip()
        
    df_raw = pd.read_csv(input_path)
    df = df_raw.copy()
    
    # Drop any reported-string columns that slipped through — these are
    # provenance fields, not physical features.
    reported_cols = [c for c in df.columns if c.endswith("_reported") and c != "HER_reported"]
    if reported_cols:
        print(f"Dropping provenance _reported columns from input: {reported_cols}")
        df.drop(columns=reported_cols, inplace=True)
        
    # Add physical features before checking schema / feature list compatibility
    df = add_physical_features(df)
    
    # 5c. Schema validation
    missing_cols = [col for col in feature_list if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following required features are missing from the input CSV: {missing_cols}")
        
        # Load medians
        medians = joblib.load(os.path.join(proc_dir, "numeric_medians.joblib"))
        
        for col in missing_cols:
            if col in medians.index:
                df[col] = medians[col]
            else:
                df[col] = "missing"
                
    # Normalize and encode categoricals
    encoder_path = os.path.join(proc_dir, "target_encoder.joblib")
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
        cat_cols_present = [c for c in encoder.feature_names_in_ if c in df.columns]
        if cat_cols_present:
            for col in cat_cols_present:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "null": np.nan})
                df[col] = df[col].fillna("missing")
            # Transform
            df[cat_cols_present] = encoder.transform(df[cat_cols_present])
            
    # Impute numeric columns
    medians_path = os.path.join(proc_dir, "numeric_medians.joblib")
    if os.path.exists(medians_path):
        medians = joblib.load(medians_path)
        numeric_cols_present = [c for c in medians.index if c in df.columns]
        df[numeric_cols_present] = df[numeric_cols_present].fillna(medians)
        
    # Get feature vector
    X_cand = df[feature_list].copy()
    
    # Point estimate predictions
    log_preds = best_model.predict(X_cand)
    her_preds = np.clip(np.expm1(log_preds), 0, None)
    
    # Build output dataframe
    output_df = pd.DataFrame()
    if "full_catalyst" in df_raw.columns:
        output_df["full_catalyst"] = df_raw["full_catalyst"]
        
    # Keep input features (original ones, not encoded/imputed)
    for col in feature_list:
        if col in df_raw.columns:
            output_df[col] = df_raw[col]
        else:
            output_df[col] = df[col] # filled value
            
    output_df["predicted_HER_umol_g_h"] = her_preds
    
    # 5a. Bootstrap CI
    if bootstrap_n > 0:
        print(f"Step 5a: Performing bootstrap uncertainty estimation ({bootstrap_n} iterations)...")
        # Load training splits and weights
        X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
        sample_weights = pd.read_csv(os.path.join(proc_dir, "sample_weights_train.csv"), header=None).squeeze()
        
        # Load best params
        with open(os.path.join(results_dir, f"best_params_{best_name}.json"), "r") as f:
            best_params = json.load(f)
            
        # Recreate base estimator
        if best_name == "XGBoost":
            from xgboost import XGBRegressor
            xgb_params = {**best_params, "tree_method": "hist", "verbosity": 0, "random_state": None}
            base_estimator = XGBRegressor(**xgb_params)
        elif best_name == "LightGBM":
            from lightgbm import LGBMRegressor
            lgb_params = {**best_params, "verbose": -1, "random_state": None}
            base_estimator = LGBMRegressor(**lgb_params)
        else:
            from sklearn.linear_model import Ridge
            base_estimator = Ridge(alpha=1.0)
            
        all_preds = []
        n_samples = len(X_train)
        
        for i in range(bootstrap_n):
            # Sample indices with replacement
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train.iloc[boot_idx]
            y_boot = y_train.iloc[boot_idx]
            w_boot = sample_weights.iloc[boot_idx]
            
            # Fit clone
            est = clone(base_estimator)
            est.fit(X_boot, y_boot, sample_weight=w_boot)
            
            # Predict
            preds_log = est.predict(X_cand)
            all_preds.append(preds_log)
            
        all_preds = np.array(all_preds) # shape: (bootstrap_n, n_candidates)
        
        # Calculate quantiles
        p10 = np.expm1(np.percentile(all_preds, 10, axis=0))
        p50 = np.expm1(np.percentile(all_preds, 50, axis=0))
        p90 = np.expm1(np.percentile(all_preds, 90, axis=0))
        
        # Clip negatives to zero
        p10 = np.clip(p10, 0, None)
        p50 = np.clip(p50, 0, None)
        p90 = np.clip(p90, 0, None)
        
        output_df["predicted_HER_p10"] = p10
        output_df["predicted_HER_p50"] = p50
        output_df["predicted_HER_p90"] = p90
        output_df["CI_width_umol_g_h"] = p90 - p10
        
        # Sort by p50 descending
        output_df = output_df.sort_values(by="predicted_HER_p50", ascending=False)
    else:
        # Sort by point estimate descending if bootstrap is skipped
        output_df = output_df.sort_values(by="predicted_HER_umol_g_h", ascending=False)
        
    if output_path is None:
        output_path = input_path.replace(".csv", "_predictions.csv")
        
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print("\nTop 5 predicted catalysts:")
    cols_to_print = ["predicted_HER_umol_g_h"]
    if "full_catalyst" in output_df.columns:
        cols_to_print = ["full_catalyst"] + cols_to_print
    elif "host_material" in output_df.columns:
        cols_to_print = ["host_material"] + cols_to_print
        
    if bootstrap_n > 0:
        cols_to_print += ["predicted_HER_p10", "predicted_HER_p50", "predicted_HER_p90", "CI_width_umol_g_h"]
        
    print(output_df[cols_to_print].head(5).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input candidates CSV")
    parser.add_argument("--output", default=None, help="Path to output predictions CSV")
    parser.add_argument("--bootstrap_n", type=int, default=100, help="Number of bootstrap iterations for CI")
    args = parser.parse_args()
    
    predict(args.input, args.output, args.bootstrap_n)
