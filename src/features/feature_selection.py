import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def select_features_by_shap(X_train, y_train, sample_weights, threshold=0.001):
    """Fits a temporary CatBoost model to compute SHAP values and keep features above threshold."""
    logger.info("Computing SHAP-based feature importance...")
    cb_temp = CatBoostRegressor(iterations=500, depth=6, random_seed=42, verbose=0, allow_writing_files=False)
    cb_temp.fit(X_train, y_train, sample_weight=sample_weights, verbose=0)
    
    explainer = shap.TreeExplainer(cb_temp)
    shap_values = explainer.shap_values(X_train)
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    active_features = [col for col, val in zip(X_train.columns, mean_shap) if val >= threshold]
    if "is_extreme_target" not in active_features and "is_extreme_target" in X_train.columns:
        active_features.append("is_extreme_target")
        
    logger.info(f"SHAP feature selection complete: kept {len(active_features)} out of {X_train.shape[1]} features.")
    return active_features
