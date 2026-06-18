import pytest
import pandas as pd
import numpy as np
from src.models.xgboost_model import tune_xgboost
from src.models.lightgbm_model import tune_lightgbm
from src.models.catboost_model import tune_catboost
from src.models.extratrees_model import tune_extratrees

@pytest.fixture
def dummy_dataset():
    np.random.seed(42)
    X_train = pd.DataFrame({
        "structure": ["rutile", "anatase"] * 10,
        "preparation_photocatalyst": [1.0, 2.0] * 10,
        "calcination_temp_semiconductor_C": np.random.uniform(300, 600, 20),
        "co_catalyst_wt_pct": np.random.uniform(0.1, 5.0, 20),
        "light_power_W": [300.0] * 20,
        "catalyst_loading_mg": [10.0] * 20,
        "glycerol_concentration_v_pct": [10.0] * 20,
        "reaction_volume_mL": [100.0] * 20,
        "bandgap_eV": [3.2] * 20,
        "catalyst_loading_g_L": [1.0] * 20,
        "pH": [7.0] * 20
    })
    y_train = pd.Series(np.random.normal(3.0, 0.5, 20))
    sample_weights = pd.Series([1.0] * 20)
    groups = pd.Series(["tio2"] * 10 + ["zno"] * 10)
    cat_cols = ["structure"]
    return X_train, y_train, sample_weights, groups, cat_cols

def test_tune_xgboost(dummy_dataset):
    X, y, w, g, c = dummy_dataset
    params, score = tune_xgboost(X, y, w, g, c, cv_folds=2, n_trials=1)
    assert isinstance(params, dict)
    assert isinstance(score, float)

def test_tune_lightgbm(dummy_dataset):
    X, y, w, g, c = dummy_dataset
    params, score = tune_lightgbm(X, y, w, g, c, cv_folds=2, n_trials=1)
    assert isinstance(params, dict)
    assert isinstance(score, float)

def test_tune_catboost(dummy_dataset):
    X, y, w, g, c = dummy_dataset
    params, score = tune_catboost(X, y, w, g, c, cv_folds=2, n_trials=1)
    assert isinstance(params, dict)
    assert isinstance(score, float)

def test_tune_extratrees(dummy_dataset):
    X, y, w, g, c = dummy_dataset
    params, score = tune_extratrees(X, y, w, g, c, cv_folds=2, n_trials=1)
    assert isinstance(params, dict)
    assert isinstance(score, float)
