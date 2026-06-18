import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.preprocessing import TargetEncoder

from src.utils.config import get_train_config
from src.utils.logging import setup_logger
from src.utils.io import safe_load_csv, safe_save_joblib, safe_save_json, safe_load_json, safe_load_joblib
from src.features.interaction_features import add_interaction_features, add_domain_features
from src.features.feature_selection import select_features_by_shap

from src.models.xgboost_model import tune_xgboost
from src.models.lightgbm_model import tune_lightgbm
from src.models.catboost_model import tune_catboost
from src.models.extratrees_model import tune_extratrees
from src.ensemble.blending import BlendingEnsemble

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

logger = setup_logger(__name__)
CFG = get_train_config()

def optimize_weights(oof_preds, y_true):
    """Finds optimal blending weights using SLSQP constraint optimization."""
    names = list(oof_preds.keys())
    X_meta = np.column_stack([oof_preds[n] for n in names])
    
    def loss(weights):
        pred = X_meta @ weights
        return -r2_score(y_true, pred)
        
    bounds = [(0.0, 1.0) for _ in range(len(names))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_weights = np.ones(len(names)) / len(names)
    
    res = minimize(loss, init_weights, bounds=bounds, constraints=constraints, method="SLSQP")
    return dict(zip(names, np.clip(res.x, 0.0, 1.0)))

def run_hpo_and_fitting(X_train, y_train, sample_weights, groups, cat_cols, cv_folds, res_dir):
    """Executes Optuna searches for base models or loads cached parameters."""
    best_params = {}
    
    # XGBoost
    xgb_path = os.path.join(res_dir, "best_params_XGBoost.json")
    if os.path.exists(xgb_path):
        best_params["XGBoost"] = safe_load_json(xgb_path)
    else:
        logger.info("Tuning XGBoost...")
        best_params["XGBoost"], _ = tune_xgboost(X_train, y_train, sample_weights, groups, cat_cols, cv_folds, n_trials=30)
        safe_save_json(best_params["XGBoost"], xgb_path)
        
    # LightGBM
    lgb_path = os.path.join(res_dir, "best_params_LightGBM.json")
    if os.path.exists(lgb_path):
        best_params["LightGBM"] = safe_load_json(lgb_path)
    else:
        logger.info("Tuning LightGBM...")
        best_params["LightGBM"], _ = tune_lightgbm(X_train, y_train, sample_weights, groups, cat_cols, cv_folds, n_trials=30)
        safe_save_json(best_params["LightGBM"], lgb_path)

    # CatBoost
    cb_path = os.path.join(res_dir, "best_params_CatBoost.json")
    if os.path.exists(cb_path):
        best_params["CatBoost"] = safe_load_json(cb_path)
    else:
        logger.info("Tuning CatBoost...")
        best_params["CatBoost"], _ = tune_catboost(X_train, y_train, sample_weights, groups, cat_cols, cv_folds, n_trials=30)
        safe_save_json(best_params["CatBoost"], cb_path)
        
    # ExtraTrees
    et_path = os.path.join(res_dir, "best_params_ExtraTrees.json")
    if os.path.exists(et_path):
        best_params["ExtraTrees"] = safe_load_json(et_path)
    else:
        logger.info("Tuning ExtraTrees...")
        best_params["ExtraTrees"], _ = tune_extratrees(X_train, y_train, sample_weights, groups, cat_cols, cv_folds, n_trials=30)
        safe_save_json(best_params["ExtraTrees"], et_path)

    return best_params

def main():
    proc_dir = CFG["paths"]["proc_dir"]
    models_dir = CFG["paths"]["models_dir"]
    res_dir = CFG["paths"]["results_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    # Load raw dataset splits and groups
    # Load filtered training data if available
    filtered_path = os.path.join(proc_dir, "X_train_filtered.csv")
    if os.path.exists(filtered_path):
        X_train = safe_load_csv(filtered_path)
    else:
        X_train = safe_load_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test = safe_load_csv(os.path.join(proc_dir, "X_test.csv"))  # test already filtered by descriptor validation

    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()
    sample_weights = pd.read_csv(os.path.join(proc_dir, "sample_weights_train.csv"), header=None).squeeze()
    groups_train = pd.read_csv(os.path.join(proc_dir, "groups_train.csv"), header=None).squeeze()
    cat_cols = safe_load_joblib(os.path.join(proc_dir, "cat_cols.joblib")) if os.path.exists(os.path.join(proc_dir, "cat_cols.joblib")) else []
    
    # Tune base models using raw categories under GroupKFold (Leakage-free HPO!)
    cv_folds = CFG["tuning"]["cv_folds"]
    best_params = run_hpo_and_fitting(X_train, y_train, sample_weights, groups_train, cat_cols, cv_folds, res_dir)
    
    # For final model fitting and evaluation/discovery, target-encode categoricals using y_train
    if cat_cols:
        encoder = TargetEncoder(random_state=CFG["data"]["random_state"], cv=5)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])
        safe_save_joblib(encoder, os.path.join(models_dir, "target_encoder.joblib"))
        
    # Add engineered features for final training
    X_train, X_test = add_interaction_features(X_train, X_test)
    X_train, X_test = add_domain_features(X_train, X_test, y_train)
    
    # Perform SHAP feature selection
    active_features = select_features_by_shap(X_train, y_train, sample_weights)
    X_train_base = X_train[active_features].copy()
    
    # Train final base models
    models = {
        "XGBoost": XGBRegressor(**best_params["XGBoost"]),
        "LightGBM": LGBMRegressor(**best_params["LightGBM"]),
        "CatBoost": CatBoostRegressor(**best_params["CatBoost"]),
        "ExtraTrees": ExtraTreesRegressor(**best_params["ExtraTrees"])
    }
    
    # Out-of-fold prediction simulation for blending
    oof_preds = {}
    for name, model in models.items():
        logger.info(f"Fitting final {name} model...")
        model.fit(X_train_base, y_train, sample_weight=sample_weights)
        oof_preds[name] = model.predict(X_train_base)
        
    # Optimize blend weights
    weights = optimize_weights(oof_preds, y_train)
    logger.info(f"Optimized Blending Weights: {weights}")
    
    # Save ensemble
    ensemble = BlendingEnsemble(models, weights, active_features)
    safe_save_joblib(ensemble, os.path.join(models_dir, "best_model.joblib"))
    
    with open(os.path.join(models_dir, "best_model_name.txt"), "w") as f:
        f.write("Blending_Ensemble")
        
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()

