import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def remove_training_outliers(df: pd.DataFrame, contamination: float = 0.04) -> pd.DataFrame:
    """Fits an IsolationForest on the log_HER column to identify and remove outliers."""
    if "log_HER" not in df.columns:
        logger.warning("log_HER column not found in DataFrame. Skipping outlier removal.")
        return df

    X_target = df[["log_HER"]].values
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X_target)
    
    is_outlier = (labels == -1)
    outliers_df = df[is_outlier]
    
    logger.info(f"IsolationForest flagged {len(outliers_df)} outlier rows out of {len(df)} total rows.")
    return df[~is_outlier].copy()
