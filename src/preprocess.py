"""
preprocess.py
Loads the raw photocatalyst dataset, cleans it, normalizes, selects features,
splits (stratified), imputes, target-encodes categoricals, computes sample weights,
and saves processed splits and transformers.
"""

import os
import yaml
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

# Load config
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths = CFG["paths"]
data_cfg = CFG["data"]

def load_dataset(raw_dir):
    print("Step 2a: Loading dataset...")
    # Sort os.listdir before iterating
    for fname in sorted(os.listdir(raw_dir)):
        fpath = os.path.join(raw_dir, fname)
        # Skip gitkeep and directories
        if fname.startswith(".") or os.path.isdir(fpath):
            continue
        
        if fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"  Loaded JSON: {fname} — Shape: {df.shape}")
            return df
        elif fname.endswith(".csv"):
            df = pd.read_csv(fpath)
            print(f"  Loaded CSV: {fname} — Shape: {df.shape}")
            return df
        elif fname.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(fpath, engine="openpyxl")
            print(f"  Loaded Excel: {fname} — Shape: {df.shape}")
            return df
            
    raise FileNotFoundError(f"No suitable dataset file found in {raw_dir}")

def main():
    os.makedirs(paths["proc_dir"], exist_ok=True)
    
    # 2a. Load
    df = load_dataset(paths["raw_dir"])
    
    # 2b. Year sanity check
    if "year" in df.columns:
        before = len(df)
        df = df[df["year"] >= 2000].copy()
        print(f"Step 2b: Dropped {before - len(df)} rows with year < 2000")
    else:
        print("Step 2b: Column 'year' not found, skipping sanity check")
        
    # 2c. Quality filtering
    before_q = len(df)
    if "data_quality_flag" in df.columns:
        df = df[df["data_quality_flag"] != "LIKELY_ERROR"].copy()
        print(f"Step 2c (i): Dropped {before_q - len(df)} rows with LIKELY_ERROR")
    else:
        print("Step 2c (i): Column 'data_quality_flag' not found, skipping")
        
    before_her = len(df)
    target_col = data_cfg["target"]
    df = df[df[target_col].notna() & (df[target_col] > 0)].copy()
    print(f"Step 2c (ii): Dropped {before_her - len(df)} rows with null/non-positive target HER")
    
    # 2d. Deduplication
    hash_col = data_cfg["hash_col"]
    if hash_col in df.columns:
        before_dedup = len(df)
        # Sort by metadata completeness descending
        df = df.sort_values(by="metadata_completeness_score", ascending=False)
        df = df.drop_duplicates(subset=hash_col, keep="first").copy()
        print(f"Step 2d: Deduplicated on '{hash_col}' keeping highest completeness. Dropped {before_dedup - len(df)} rows")
    else:
        print(f"Step 2d: Column '{hash_col}' not found, skipping deduplication")
        
    # 2e. Drop near-empty and constant columns
    print("Step 2e: Identifying columns to drop due to missingness or zero variance...")
    missing_thresh = data_cfg["missing_threshold"]
    dropped_cols = []
    
    for col in df.columns:
        # Don't drop target or essential identifier/provenance fields
        if col in [target_col, "metadata_completeness_score"] or col in data_cfg["confidence_weight_cols"]:
            continue
            
        # Check null fraction
        null_frac = df[col].isna().mean()
        if null_frac >= missing_thresh:
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)
            print(f"  Dropped '{col}': null fraction = {null_frac:.3f} >= {missing_thresh}")
            continue
            
        # Check unique non-null values
        unique_vals = df[col].dropna().nunique()
        if unique_vals <= 1:
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)
            print(f"  Dropped '{col}': constant value (unique non-null count = {unique_vals})")
            
    print(f"Step 2e complete. Total dropped columns: {len(dropped_cols)}")
    
    # 2f. Case normalization
    print("Step 2f: Normalizing categoricals to lowercase...")
    # Find all string/object columns dynamically
    str_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    all_cat_cols = list(set(data_cfg["cat_cols"] + str_cols))
    for col in all_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            # Restore actual NaNs from string representation
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "null": np.nan})
            
    # 2g. Dynamic feature selection
    print("Step 2g: Selecting features dynamically...")
    feature_cols = []
    for col in df.columns:
        # Exclude leakage
        if col in data_cfg["leakage_cols"]:
            continue
        # Exclude provenance
        if col in data_cfg["provenance_cols"]:
            continue
        # Exclude confidence
        if col.startswith("confidence_") or col in data_cfg["confidence_weight_cols"]:
            continue
        # Exclude target
        if col == target_col or col == "log_HER":
            continue
            
        feature_cols.append(col)
        
    print(f"Step 2g complete. Selected {len(feature_cols)} features: {feature_cols}")
    
    # 2h. Log-transform target
    print("Step 2h: Log-transforming target...")
    df["log_HER"] = np.log1p(df[target_col])
    
    # 2i. Stratified split
    print("Step 2i: Creating stratified split on log_HER...")
    # Quantile bins for stratification (duplicates dropped to ensure valid bins)
    strat_bins = pd.qcut(df["log_HER"], 10, labels=False, duplicates="drop")
    
    X = df[feature_cols].copy()
    y = df["log_HER"].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        stratify=strat_bins,
        random_state=data_cfg["random_state"]
    )
    print(f"  Split shapes — Train: {X_train.shape} | Test: {X_test.shape}")
    
    # 2j. Numeric imputation (fit on train only)
    print("Step 2j: Performing numeric imputation...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    medians = X_train[numeric_cols].median()
    
    # Apply imputation
    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)
    
    # Save medians
    joblib.dump(medians, os.path.join(paths["proc_dir"], "numeric_medians.joblib"))
    print("  Saved numeric medians.")
    
    # 2k. Categorical encoding
    print("Step 2k: Fitting TargetEncoder on categorical features...")
    # Dynamic categorical columns: those listed in config or having object/string dtype
    cat_cols_present = [
        col for col in X_train.columns
        if col in data_cfg["cat_cols"] or X_train[col].dtype == object or isinstance(X_train[col].dtype, pd.StringDtype)
    ]
    
    if cat_cols_present:
        # TargetEncoder requires category features as objects/categories
        X_train[cat_cols_present] = X_train[cat_cols_present].fillna("missing")
        X_test[cat_cols_present] = X_test[cat_cols_present].fillna("missing")
        
        encoder = TargetEncoder(random_state=data_cfg["random_state"], cv=5)
        # Fit on train categoricals and target
        X_train[cat_cols_present] = encoder.fit_transform(X_train[cat_cols_present], y_train)
        X_test[cat_cols_present] = encoder.transform(X_test[cat_cols_present])
        
        # Save encoder
        joblib.dump(encoder, os.path.join(paths["proc_dir"], "target_encoder.joblib"))
        print(f"  Encoded categories: {cat_cols_present}")
    else:
        print("  No categorical features present. Skipping encoding.")
        
    # 2l. Sample weights
    print("Step 2l: Computing sample weights...")
    # Retrieve metadata completeness score aligning to training indices
    train_indices = X_train.index
    completeness = df.loc[train_indices, "metadata_completeness_score"].fillna(0.5)
    
    w = completeness.copy()
    
    # Iterate over confidence weight columns
    weight_map = data_cfg["confidence_weight_map"]
    for col in data_cfg["confidence_weight_cols"]:
        if col in df.columns:
            mapped_w = df.loc[train_indices, col].map(weight_map).fillna(0.5)
            w *= mapped_w
            
    # Clip weights to [0.1, 1.0]
    w = np.clip(w, 0.1, 1.0)
    
    # Save weights
    w.to_csv(os.path.join(paths["proc_dir"], "sample_weights_train.csv"), index=False, header=False)
    print("  Saved training sample weights.")
    
    # 2m. Save outputs
    print("Step 2m: Saving preprocessed datasets and feature list...")
    X_train.to_csv(os.path.join(paths["proc_dir"], "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(paths["proc_dir"], "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(paths["proc_dir"], "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(paths["proc_dir"], "y_test.csv"), index=False)
    
    joblib.dump(feature_cols, os.path.join(paths["proc_dir"], "feature_list.joblib"))
    df.to_csv(os.path.join(paths["proc_dir"], "df_clean.csv"), index=True)
    
    print("\nPreprocessing workflow completed successfully!")

if __name__ == "__main__":
    main()
