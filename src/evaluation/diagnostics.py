import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def evaluate_residual_normality(y_true, y_pred, results_dir):
    """Performs Shapiro-Wilk test on residuals and saves analysis."""
    residuals = y_true - y_pred
    stat, p = stats.shapiro(residuals)
    logger.info(f"Residual normality test: statistic={stat:.4f}, p-value={p:.4e}")
    
    # Save residual plot
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    ax.set_title("Normal Q-Q Plot of Residuals")
    fig.savefig(os.path.join(results_dir, "residuals_qq.png"), dpi=150)
    plt.close()
    return stat, p
