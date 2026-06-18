import pytest
import pandas as pd
import numpy as np
from src.features.material_descriptors import add_physical_features, encode_semiconductor, encode_cocatalyst
from src.features.interaction_features import add_interaction_features, add_domain_features
from src.features.feature_selection import select_features_by_shap

def test_material_descriptors():
    df = pd.DataFrame({
        "host_material": ["tio2", "srtio3", "unknown_semic"],
        "co_catalyst": ["pt", "none", "nan"]
    })
    enriched = add_physical_features(df)
    assert "semi_bandgap_eV" in enriched.columns
    assert enriched.loc[0, "semi_bandgap_eV"] == 3.20
    assert "cocat_work_function" in enriched.columns
    assert enriched.loc[0, "cocat_work_function"] == 5.65

def test_feature_engineering():
    X_train = pd.DataFrame({
        "structure": [1.0, 2.0],
        "preparation_photocatalyst": [1.0, 2.0],
        "calcination_temp_semiconductor_C": [400.0, 500.0],
        "co_catalyst_wt_pct": [1.0, 2.0],
        "light_power_W": [300.0, 310.0],
        "catalyst_loading_mg": [10.0, 11.0],
        "glycerol_concentration_v_pct": [10.0, 11.0],
        "reaction_volume_mL": [100.0, 100.0],
        "bandgap_eV": [3.2, 2.4],
        "catalyst_loading_g_L": [1.0, 1.1],
        "pH": [7.0, 6.0]
    })
    X_test = X_train.copy()
    y_train = pd.Series([1.0, 2.0])
    
    X_train_int, X_test_int = add_interaction_features(X_train, X_test)
    assert "structure_x_calc_temp" in X_train_int.columns
    
    X_train_dom, X_test_dom = add_domain_features(X_train_int, X_test_int, y_train)
    assert "bandgap_visible_overlap" in X_train_dom.columns

def test_shap_feature_selection():
    X_train = pd.DataFrame({
        "feat_1": np.random.normal(0, 1, 50),
        "feat_2": np.random.normal(0, 1, 50),
        "is_extreme_target": [0] * 50
    })
    # Make feat_1 highly correlated with target
    y_train = pd.Series(X_train["feat_1"] * 5.0 + np.random.normal(0, 0.1, 50))
    weights = pd.Series([1.0] * 50)
    
    active = select_features_by_shap(X_train, y_train, weights, threshold=0.01)
    assert "feat_1" in active
