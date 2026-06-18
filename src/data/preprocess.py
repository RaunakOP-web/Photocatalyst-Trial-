import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from src.utils.logging import setup_logger
from src.utils.config import get_train_config, get_features_config
from src.utils.io import safe_save_csv, safe_save_joblib
from src.data.load import load_raw_dataset
from src.features.material_descriptors import add_physical_features
from src.data.validation import generate_stratified_split
from src.data.outlier_removal import remove_training_outliers
# New descriptor helper imports
from src.data.get_band_edges import add_band_edges
from src.data.calc_h_adsorption import add_h_adsorption
from src.data.calc_carrier_mobility import add_carrier_mobility
from src.data.calc_exciton_binding import add_exciton_binding
from src.data.calc_surface_descriptors import add_surface_descriptors

logger = setup_logger(__name__)
CFG_TRAIN = get_train_config()
CFG_FEAT = get_features_config()


def preprocess_pipeline():
    logger.info("Executing Preprocessing Pipeline...")
    raw_dir = CFG_TRAIN["paths"]["raw_dir"]
    proc_dir = CFG_TRAIN["paths"]["proc_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    # Load raw dataset and enrich with physical descriptors
    df = load_raw_dataset(raw_dir)
    df = add_physical_features(df)

    # Target and quality checks
    target_col = CFG_TRAIN["data"]["target"]
    df = df[df[target_col].notna() & (df[target_col] > 0)].copy()

    # Compute log_HER for stratified splitting (must exist before split)
    df["log_HER"] = np.log1p(df[target_col])

    # Deduplicate
    hash_col = CFG_TRAIN["data"]["hash_col"]
    if hash_col in df.columns:
        df = df.sort_values(by="metadata_completeness_score", ascending=False)
        df = df.drop_duplicates(subset=hash_col, keep="first").copy()

    # Drop empty and constant columns
    missing_thresh = CFG_TRAIN["data"]["missing_threshold"]
    for col in list(df.columns):
        if col in [target_col, "metadata_completeness_score"] or col in CFG_FEAT["confidence_weight_cols"]:
            continue
        if df[col].isna().mean() >= missing_thresh or df[col].dropna().nunique() <= 1:
            df.drop(columns=[col], inplace=True)

    # Lowercase string categories
    str_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    all_cat_cols = list(set(CFG_FEAT["cat_cols"] + str_cols))
    for col in all_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "null": np.nan})

    # Split train and test index (leakage‑free)
    train_idx, test_idx = generate_stratified_split(
        df, "log_HER", CFG_TRAIN["data"]["test_size"], CFG_TRAIN["data"]["random_state"]
    )
    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    # Remove outliers ONLY on training set (Leakage‑free!)
    df_train = remove_training_outliers(df_train, contamination=0.04)

    # NEW: Apply physics‑informed descriptor helpers post‑split
    for helper in [add_band_edges, add_h_adsorption, add_carrier_mobility, add_exciton_binding, add_surface_descriptors]:
        df_train = helper(df_train)
        df_test = helper(df_test)

    # Re‑compute feature column list after new descriptors have been added
    feature_cols = []
    for col in df.columns:
        if col in CFG_FEAT["leakage_cols"] or col in CFG_FEAT["provenance_cols"]:
            continue
        if col.startswith("confidence_") or col in CFG_FEAT["confidence_weight_cols"]:
            continue
        if col in [target_col, "log_HER", "host_material", "co_catalyst"]:
            continue
        feature_cols.append(col)

    X_train = df_train[feature_cols].copy()
    y_train = df_train["log_HER"].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test["log_HER"].copy()

    # Save groups for GroupKFold
    groups_train = df_train["host_material"].fillna("unknown")
    groups_test = df_test["host_material"].fillna("unknown")
    groups_train.to_csv(os.path.join(proc_dir, "groups_train.csv"), index=False, header=False)
    groups_test.to_csv(os.path.join(proc_dir, "groups_test.csv"), index=False, header=False)

    # Impute numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)
    safe_save_joblib(medians, os.path.join(proc_dir, "numeric_medians.joblib"))

    # Handle categoricals (fill missing values, keep raw categories)
    cat_cols_present = [
        col for col in X_train.columns
        if col in CFG_FEAT["cat_cols"] or X_train[col].dtype == object
    ]
    if cat_cols_present:
        X_train[cat_cols_present] = X_train[cat_cols_present].fillna("missing")
        X_test[cat_cols_present] = X_test[cat_cols_present].fillna("missing")
        safe_save_joblib(cat_cols_present, os.path.join(proc_dir, "cat_cols.joblib"))

    # Sample Weights for training
    completeness = df_train["metadata_completeness_score"].fillna(0.5)
    w = completeness.copy()
    weight_map = CFG_FEAT["confidence_weight_map"]
    for col in CFG_FEAT["confidence_weight_cols"]:
        if col in df_train.columns:
            mapped_w = df_train[col].map(weight_map).fillna(0.5)
            w *= mapped_w
    w = np.clip(w, 0.1, 1.0)
    w.to_csv(os.path.join(proc_dir, "sample_weights_train.csv"), index=False, header=False)

    # Save outputs
    safe_save_csv(X_train, os.path.join(proc_dir, "X_train.csv"))
    safe_save_csv(X_test, os.path.join(proc_dir, "X_test.csv"))
    y_train.to_csv(os.path.join(proc_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(proc_dir, "y_test.csv"), index=False)
    safe_save_joblib(feature_cols, os.path.join(proc_dir, "feature_list.joblib"))
    df_train.to_csv(os.path.join(proc_dir, "df_clean.csv"), index=True)

    logger.info("Preprocessing Pipeline finished successfully.")
