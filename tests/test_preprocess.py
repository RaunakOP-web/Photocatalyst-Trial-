import pytest
import pandas as pd
import numpy as np
from src.data.outlier_removal import remove_training_outliers
from src.data.validation import generate_stratified_split, get_logo_cv_splits, preprocess_and_engineer_folds

def test_outlier_removal():
    df = pd.DataFrame({
        "log_HER": [1.0, 1.2, 1.1, 1.3, 100.0, 1.0, 1.1]
    })
    cleaned_df = remove_training_outliers(df, contamination=0.1)
    assert len(cleaned_df) < len(df)
    assert 100.0 not in cleaned_df["log_HER"].values

def test_stratified_split():
    df = pd.DataFrame({
        "log_HER": np.random.normal(5.0, 1.0, 100)
    })
    train_idx, test_idx = generate_stratified_split(df, "log_HER", test_size=0.2, random_state=42)
    assert len(train_idx) == 80
    assert len(test_idx) == 20

def test_logo_cv_splits():
    df = pd.DataFrame({
        "host_material": ["tio2", "tio2", "zno", "zno", "cds"]
    })
    splits = get_logo_cv_splits(df, "host_material")
    assert len(splits) == 3  # 3 groups

def test_preprocess_and_engineer_folds():
    # Setup dummy columns (needs at least 6 samples for TargetEncoder cv=5)
    X_tr = pd.DataFrame({
        "structure": ["rutile", "anatase", "rutile", "rutile", "anatase", "rutile"],
        "preparation_photocatalyst": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        "calcination_temp_semiconductor_C": [400.0, 500.0, 450.0, 400.0, 500.0, 450.0],
        "co_catalyst_wt_pct": [1.0, 2.0, 1.5, 1.0, 2.0, 1.5],
        "light_power_W": [300.0, 310.0, 320.0, 300.0, 310.0, 320.0],
        "catalyst_loading_mg": [10.0, 11.0, 12.0, 10.0, 11.0, 12.0],
        "glycerol_concentration_v_pct": [10.0, 11.0, 12.0, 10.0, 11.0, 12.0],
        "reaction_volume_mL": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "bandgap_eV": [3.2, 2.4, 3.2, 3.2, 2.4, 3.2],
        "catalyst_loading_g_L": [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
        "pH": [7.0, 6.0, 8.0, 7.0, 6.0, 8.0]
    })
    X_val = pd.DataFrame({
        "structure": ["rutile"],
        "preparation_photocatalyst": [1.0],
        "calcination_temp_semiconductor_C": [400.0],
        "co_catalyst_wt_pct": [1.0],
        "light_power_W": [300.0],
        "catalyst_loading_mg": [10.0],
        "glycerol_concentration_v_pct": [10.0],
        "reaction_volume_mL": [100.0],
        "bandgap_eV": [3.2],
        "catalyst_loading_g_L": [1.0],
        "pH": [7.0]
    })
    y_tr = pd.Series([2.5, 3.0, 2.8, 2.5, 3.0, 2.8])
    
    X_tr_proc, X_val_proc = preprocess_and_engineer_folds(X_tr, X_val, y_tr, cat_cols=["structure"])
    assert "structure_x_calc_temp" in X_tr_proc.columns
    assert "bandgap_visible_overlap" in X_tr_proc.columns
    assert X_tr_proc.shape[0] == 6
    assert X_val_proc.shape[0] == 1
