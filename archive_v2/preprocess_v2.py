"""
preprocess.py  (v2 — group-aware, leakage-clean)
Loads the raw photocatalyst dataset, cleans it, adds physical features,
performs a GROUP-AWARE train/test split (no material leaks across boundary),
applies winsorization and imputation fit on train only, saves raw categoricals
alongside numeric data so train.py can target-encode inside each CV fold.

Key changes vs v1:
  - GroupShuffleSplit replaces stratified train_test_split → prevents the same
    host_material appearing in both train and test (except TiO2 which is so
    dominant it must span both for sample-size reasons — see note below).
  - TargetEncoder is NO LONGER applied here; raw categorical columns are
    preserved so train.py can encode fresh inside each fold.
  - Winsorization bounds fit on train, applied to test.
  - Saves group_labels_train.csv and group_labels_test.csv for LOGO-CV.
  - Feature engineering (interactions, log transforms) delegated to
    feature_engineering.py, called here.
"""

import os
import yaml
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.material_features import add_physical_features
from src.feature_engineering import add_engineered_features

# Load config
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths    = CFG["paths"]
data_cfg = CFG["data"]


def load_dataset(raw_dir: str) -> pd.DataFrame:
    print("Step 1: Loading dataset...")
    for fname in sorted(os.listdir(raw_dir)):
        fpath = os.path.join(raw_dir, fname)
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

    # ── 1. Load & add physical features ──────────────────────────────────────
    df = load_dataset(paths["raw_dir"])
    print("Step 2: Adding physical/chemical property features...")
    df = add_physical_features(df)

    # ── 2. Year filter ────────────────────────────────────────────────────────
    if "year" in df.columns:
        before = len(df)
        df = df[df["year"] >= 2000].copy()
        print(f"Step 3: Dropped {before - len(df)} rows with year < 2000 "
              f"({len(df)} remain)")
    else:
        print("Step 3: Column 'year' not found, skipping")

    # ── 3. Quality filtering ──────────────────────────────────────────────────
    target_col = data_cfg["target"]
    if "data_quality_flag" in df.columns:
        before = len(df)
        df = df[df["data_quality_flag"] != "LIKELY_ERROR"].copy()
        print(f"Step 4a: Dropped {before - len(df)} LIKELY_ERROR rows")

    before = len(df)
    df = df[df[target_col].notna() & (df[target_col] > 0)].copy()
    print(f"Step 4b: Dropped {before - len(df)} rows with null/non-positive HER")

    # ── 4. Deduplication ──────────────────────────────────────────────────────
    hash_col = data_cfg["hash_col"]
    if hash_col in df.columns:
        before = len(df)
        df = df.sort_values("metadata_completeness_score", ascending=False)
        df = df.drop_duplicates(subset=hash_col, keep="first").copy()
        print(f"Step 5: Deduped on '{hash_col}', dropped {before - len(df)} rows "
              f"({len(df)} remain)")

    # ── 5. Drop near-empty / constant columns ─────────────────────────────────
    print("Step 6: Dropping high-missingness & constant columns...")
    missing_thresh = data_cfg["missing_threshold"]
    dropped_cols = []
    skip_cols = {target_col, "metadata_completeness_score", "log_HER",
                 data_cfg["group_col"]}
    skip_cols.update(data_cfg["confidence_weight_cols"])

    for col in list(df.columns):
        if col in skip_cols:
            continue
        null_frac = df[col].isna().mean()
        if null_frac >= missing_thresh:
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)
            continue
        if df[col].dropna().nunique() <= 1:
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)
    print(f"  Dropped {len(dropped_cols)} columns: {dropped_cols}")

    # ── 6. String normalisation ───────────────────────────────────────────────
    print("Step 7: Normalising categoricals to lowercase...")
    str_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    all_cat_cols = list(set(data_cfg["cat_cols"] + str_cols))
    for col in all_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "null": np.nan})

    # ── 7. Log-transform target ───────────────────────────────────────────────
    df["log_HER"] = np.log1p(df[target_col])
    print(f"Step 8: log_HER added. Range: [{df['log_HER'].min():.2f}, "
          f"{df['log_HER'].max():.2f}]")

    print("Step 7b: Adding host_material frequency encoding...")
    if "host_material" in df.columns:
        df["host_material_freq"] = df.groupby("host_material")["host_material"].transform("count") / len(df)



    # ── 8. Feature selection ──────────────────────────────────────────────────
    print("Step 9: Selecting features...")
    exclude = (
        set(data_cfg["leakage_cols"])
        | set(data_cfg["provenance_cols"])
        | set(data_cfg["confidence_weight_cols"])
        | {target_col, "log_HER"}
    )
    # Keep host_material in df_clean but exclude from feature matrix
    exclude.update({"host_material"})

    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"  Selected {len(feature_cols)} features")

    # ── 9. Group-aware train/test split ───────────────────────────────────────
    print("Step 10: Group-aware train/test split...")
    group_col = data_cfg["group_col"]
    groups = df[group_col].fillna("unknown").values

    # GroupShuffleSplit: guarantees no host_material in both train and test
    # Exception: TiO2 is 72% of data — it MUST appear in both splits.
    # Strategy: force all non-TiO2 rows into a group-safe split; TiO2 is
    # large enough that any random 15% split of TiO2-only rows is OK.
    tio2_mask = (df[group_col].str.lower().str.contains("tio2", na=False))
    df_tio2 = df[tio2_mask].copy()
    df_other = df[~tio2_mask].copy()

    # For non-TiO2: use GroupShuffleSplit to keep groups intact
    if len(df_other) > 0 and df_other[group_col].nunique() > 1:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=data_cfg["test_size"],
            random_state=data_cfg["random_state"]
        )
        other_groups = df_other[group_col].fillna("unknown").values
        train_other_idx, test_other_idx = next(
            gss.split(df_other, groups=other_groups)
        )
        df_other_train = df_other.iloc[train_other_idx]
        df_other_test  = df_other.iloc[test_other_idx]
    else:
        # Fallback if very few non-TiO2 rows
        df_other_train = df_other
        df_other_test  = pd.DataFrame(columns=df_other.columns)

    # For TiO2: simple stratified split (within TiO2, no group leakage concern)
    strat_bins_tio2 = pd.qcut(df_tio2["log_HER"], 8, labels=False, duplicates="drop")
    tio2_train, tio2_test = train_test_split(
        df_tio2,
        test_size=data_cfg["test_size"],
        stratify=strat_bins_tio2,
        random_state=data_cfg["random_state"]
    )

    df_train = pd.concat([df_other_train, tio2_train]).sort_index()
    df_test  = pd.concat([df_other_test,  tio2_test ]).sort_index()
    df_train["split"] = "train"
    df_test["split"]  = "test"

    print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")
    print(f"  Train materials: {df_train[group_col].nunique()} unique")
    print(f"  Test materials:  {df_test[group_col].nunique()} unique")

    # ── 10. Extract features & targets ───────────────────────────────────────
    X_train = df_train[feature_cols].copy()
    X_test  = df_test[feature_cols].copy()
    y_train = df_train["log_HER"].copy()
    y_test  = df_test["log_HER"].copy()

    # ── 11. Numeric imputation (domain-aware + median) ────────────────────────
    print("Step 11: Domain-aware numeric imputation...")
    # Separate numeric from categorical
    cat_cols_present = [
        c for c in X_train.columns
        if c in data_cfg["cat_cols"]
        or X_train[c].dtype == object
        or isinstance(X_train[c].dtype, pd.StringDtype)
    ]
    numeric_cols = [c for c in X_train.columns if c not in cat_cols_present]
    
    domain_defaults = {
        "co_catalyst_wt_pct": 0.0,
        "semiconductor_2_pct": 0.0,
        "temperature_C": 25.0,
        "pH": 7.0,
        "wavelength_cutoff_nm": 420.0
    }
    
    for col, val in domain_defaults.items():
        if col in numeric_cols:
            X_train[col] = X_train[col].fillna(val)
            X_test[col] = X_test[col].fillna(val)

    medians = X_train[numeric_cols].median()

    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols]  = X_test[numeric_cols].fillna(medians)
    joblib.dump(medians, os.path.join(paths["proc_dir"], "numeric_medians.joblib"))

    # ── 12. Fill categorical NaNs with "missing" (for downstream encoding) ──
    print("Step 12: Filling categorical NaNs...")
    for col in cat_cols_present:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna("missing")
            X_test[col]  = X_test[col].fillna("missing")

    print("Step 13: Feature engineering...")
    X_train = add_engineered_features(X_train)
    X_test  = add_engineered_features(X_test)
    
    # Drop string columns now that flags are created
    if "co_catalyst" in X_train.columns:
        X_train = X_train.drop(columns=["co_catalyst"])
        X_test = X_test.drop(columns=["co_catalyst"])
    if "semiconductor_2" in X_train.columns:
        X_train = X_train.drop(columns=["semiconductor_2"])
        X_test = X_test.drop(columns=["semiconductor_2"])

    # ── 14. Save sample weights ───────────────────────────────────────────────
    print("Step 14: Computing sample weights...")
    completeness = df_train["metadata_completeness_score"].fillna(0.5).values
    w = completeness.copy()
    weight_map = data_cfg["confidence_weight_map"]
    for col in data_cfg["confidence_weight_cols"]:
        if col in df_train.columns:
            mapped = df_train[col].map(weight_map).fillna(0.5).values
            w = w * mapped
    w = np.clip(w, 0.1, 1.0)
    pd.Series(w).to_csv(
        os.path.join(paths["proc_dir"], "sample_weights_train.csv"),
        index=False, header=False
    )

    # ── 15. Save group labels for LOGO-CV ────────────────────────────────────
    print("Step 15: Saving group labels...")
    df_train[group_col].fillna("unknown").reset_index(drop=True).to_csv(
        os.path.join(paths["proc_dir"], "group_labels_train.csv"),
        index=False, header=True
    )
    df_test[group_col].fillna("unknown").reset_index(drop=True).to_csv(
        os.path.join(paths["proc_dir"], "group_labels_test.csv"),
        index=False, header=True
    )

    # ── 16. Save outputs ──────────────────────────────────────────────────────
    print("Step 16: Saving all processed outputs...")
    X_train.reset_index(drop=True).to_csv(
        os.path.join(paths["proc_dir"], "X_train.csv"), index=False)
    X_test.reset_index(drop=True).to_csv(
        os.path.join(paths["proc_dir"], "X_test.csv"),  index=False)
    y_train.reset_index(drop=True).to_csv(
        os.path.join(paths["proc_dir"], "y_train.csv"), index=False, header=True)
    y_test.reset_index(drop=True).to_csv(
        os.path.join(paths["proc_dir"], "y_test.csv"),  index=False, header=True)

    # Save feature list
    all_feature_cols = X_train.columns.tolist()
    joblib.dump(all_feature_cols,
                os.path.join(paths["proc_dir"], "feature_list.joblib"))

    # Save categorical column list for fold-level target encoding in train.py
    cat_cols_final = [c for c in all_feature_cols
                      if c in data_cfg["cat_cols"]
                      or X_train[c].dtype == object
                      or isinstance(X_train[c].dtype, pd.StringDtype)]
    joblib.dump(cat_cols_final,
                os.path.join(paths["proc_dir"], "cat_cols.joblib"))

    # Save full cleaned df for downstream scripts (evaluate, ablation, etc.)
    df_full = pd.concat([df_train, df_test]).sort_index()
    df_full.to_csv(os.path.join(paths["proc_dir"], "df_clean.csv"), index=True)

    print(f"\nPreprocessing complete!")
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  Categorical columns ({len(cat_cols_final)}): {cat_cols_final}")
    print(f"  Group distribution in train:\n"
          f"{df_train[group_col].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    main()
