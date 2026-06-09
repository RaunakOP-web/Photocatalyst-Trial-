import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def remove_training_outliers(df: pd.DataFrame, contamination: float = 0.04) -> pd.DataFrame:
    """
    Fits an IsolationForest on the log_HER column to identify and remove outliers.
    """
    if "log_HER" not in df.columns:
        print("Warning: log_HER column not found in DataFrame. Skipping outlier removal.")
        return df

    # Prepare data for IsolationForest
    X_target = df[["log_HER"]].values
    
    # Fit IsolationForest
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X_target)
    
    # Identify outliers (labels == -1)
    is_outlier = (labels == -1)
    outliers_df = df[is_outlier]
    
    print(f"\n--- Outlier Detection (IsolationForest, contamination={contamination}) ---")
    print(f"Flagged {len(outliers_df)} outlier rows out of {len(df)} total rows.")
    print("Flagged log_HER values:")
    print(sorted(outliers_df["log_HER"].tolist()))
    print("--------------------------------------------------------------------\n")
    
    # Return cleaned DataFrame
    df_clean = df[~is_outlier].copy()
    return df_clean
