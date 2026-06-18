import numpy as np
import pandas as pd
from sklearn.base import clone

def run_bootstrap_prediction(X_train, y_train, sample_weights, X_cand, base_model, n_boot=50):
    """Generates bootstrap predictions by resampling training data and fitting base_model copies."""
    all_preds = []
    n_samples = len(X_train)
    
    # Exclude string columns for numeric model fitting
    cols_numeric = [c for c in X_train.columns if X_train[c].dtype != object]
    X_train_num = X_train[cols_numeric].copy()
    X_cand_num = X_cand[cols_numeric].copy()
    
    for i in range(n_boot):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train_num.iloc[boot_idx]
        y_boot = y_train.iloc[boot_idx]
        w_boot = sample_weights.iloc[boot_idx]
        
        est = clone(base_model)
        est.fit(X_boot, y_boot, sample_weight=w_boot)
        all_preds.append(est.predict(X_cand_num))
        
    return np.array(all_preds)
