import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def calculate_metrics(y_true, y_pred):
    """Computes R2, MAE, and RMSE in log scale."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse)
    }

def calculate_original_scale_metrics(y_true_log, y_pred_log):
    """Back-transforms log predictions and calculates original scale metrics."""
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    return {
        "R2_orig": float(r2),
        "MAE_orig": float(mae),
        "RMSE_orig": float(rmse)
    }
