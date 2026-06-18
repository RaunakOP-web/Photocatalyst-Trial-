import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def generate_stratified_split(df, target_col, test_size, random_state):
    """Generates a stratified split using quantiles of the target variable."""
    logger.info("Generating stratified train/test splits...")
    strat_bins = pd.qcut(df[target_col], 10, labels=False, duplicates="drop")
    
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=strat_bins,
        random_state=random_state
    )
    return train_idx, test_idx

def get_logo_cv_splits(df, group_col):
    """Helper to generate LeaveOneGroupOut CV splits."""
    logo = LeaveOneGroupOut()
    groups = df[group_col].fillna("unknown")
    return list(logo.split(df, groups=groups))

def preprocess_and_engineer_folds(X_tr, X_val, y_tr, cat_cols=None):
    """Target encodes, imputes, and engineers features for a CV fold to avoid leakage."""
    import numpy as np
    from sklearn.preprocessing import TargetEncoder
    from src.features.interaction_features import add_interaction_features, add_domain_features
    
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    
    # Target Encode categoricals inside CV loop
    if cat_cols:
        te = TargetEncoder(random_state=42, cv=5)
        X_tr[cat_cols] = te.fit_transform(X_tr[cat_cols], y_tr)
        X_val[cat_cols] = te.transform(X_val[cat_cols])
        
    # Fill NAs
    numeric_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    medians = X_tr[numeric_cols].median()
    X_tr[numeric_cols] = X_tr[numeric_cols].fillna(medians).fillna(0.0)
    X_val[numeric_cols] = X_val[numeric_cols].fillna(medians).fillna(0.0)
    
    # Interaction Features
    X_tr, X_val = add_interaction_features(X_tr, X_val)
    
    # Domain Features (handles variance filtering inside)
    X_tr, X_val = add_domain_features(X_tr, X_val, y_tr)
    
    return X_tr, X_val

