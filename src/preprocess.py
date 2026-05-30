"""
preprocess.py
Loads the glycerol photocatalyst master dataset, applies feature
engineering, handles missing values, encodes categoricals, and
saves train/test splits to data/processed/.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import joblib

# ── CONFIG ──────────────────────────────────────────────
RAW_DIR   = "data/raw"
PROC_DIR  = "data/processed"
RANDOM_STATE = 42

TARGET = "HER_std_umol_g_h"

FEATURES = [
    "host_material",
    "co_catalyst",
    "co_catalyst_wt_pct",
    "semiconductor_2",
    "glycerol_concentration_std",
    "catalyst_loading_mg",
    "reaction_volume_mL",
    "temperature_C",
    "pH",
    "light_power_W",
    "wavelength_cutoff_nm",
    "is_xe_lamp",
    "is_hg_lamp",
    "is_led",
    "is_uv",
    "is_visible_light",
    "is_solar_simulator",
]

# Columns that would cause data leakage — derived from HER
LEAKAGE_COLS = ["AQY_pct", "AQE_pct", "STH_pct", "HER_reported"]

CAT_COLS = ["host_material", "co_catalyst", "semiconductor_2"]

# ── LOAD ────────────────────────────────────────────────
def load_dataset(raw_dir):
    """Try JSON first, then CSV, then XLSX."""
    for fname in os.listdir(raw_dir):
        fpath = os.path.join(raw_dir, fname)
        if fname.endswith(".json"):
            with open(fpath) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"Loaded JSON: {fname} — {df.shape}")
            return df
        elif fname.endswith(".csv"):
            df = pd.read_csv(fpath)
            print(f"Loaded CSV: {fname} — {df.shape}")
            return df
        elif fname.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(fpath, engine="openpyxl")
            print(f"Loaded Excel: {fname} — {df.shape}")
            return df
    raise FileNotFoundError(
        f"No dataset file found in {raw_dir}. "
        "Drop your JSON, CSV, or XLSX file there and re-run."
    )

# ── CLEAN ───────────────────────────────────────────────
def clean(df):
    print(f"\nStarting shape: {df.shape}")

    # Remove flagged error rows
    if "data_quality_flag" in df.columns:
        before = len(df)
        df = df[df["data_quality_flag"] != "LIKELY_ERROR"].copy()
        print(f"Removed {before - len(df)} LIKELY_ERROR rows")

    # Remove zero or null HER
    before = len(df)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    print(f"Removed {before - len(df)} zero/null HER rows")

    # Drop leakage columns if present
    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    print(f"Dropped leakage columns: {drop_cols}")

    # Fill categorical nulls
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Log-transform target
    df["log_HER"] = np.log1p(df[TARGET])

    print(f"Clean shape: {df.shape}")
    return df

# ── FEATURE ENGINEERING ─────────────────────────────────
def engineer_features(df):
    # Only keep features that exist in this dataset
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"Warning: these features not found and will be skipped: {missing}")

    X = df[available].copy()
    y = df["log_HER"].copy()
    return X, y, available

# ── ENCODE ──────────────────────────────────────────────
def encode(X, cat_cols, fit=True, encoder=None):
    present_cats = [c for c in cat_cols if c in X.columns]
    if fit:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        X[present_cats] = encoder.fit_transform(X[present_cats])
        return X, encoder
    else:
        X[present_cats] = encoder.transform(X[present_cats])
        return X, encoder

# ── MAIN ────────────────────────────────────────────────
def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    df_raw = load_dataset(RAW_DIR)
    df     = clean(df_raw)
    X, y, used_features = engineer_features(df)
    X, encoder = encode(X, CAT_COLS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )

    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

    # Save splits
    X_train.to_csv(f"{PROC_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROC_DIR}/X_test.csv",  index=False)
    y_train.to_csv(f"{PROC_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROC_DIR}/y_test.csv",  index=False)

    # Save encoder and feature list
    joblib.dump(encoder,       f"{PROC_DIR}/encoder.joblib")
    joblib.dump(used_features, f"{PROC_DIR}/feature_list.joblib")

    print(f"\nAll outputs saved to {PROC_DIR}/")
    print(f"Features used: {used_features}")
    print("\nPreprocessing complete. Run src/train.py next.")

if __name__ == "__main__":
    main()
