import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def rank_candidates(cand_df, records, predictions, uq_intervals, X_train, kappa=1.0):
    """Combines predictions, uncertainty intervals, and novelty scores to rank candidates."""
    preds_med = predictions
    pred_lo, pred_hi = uq_intervals
    
    ucb = preds_med + kappa * (pred_hi - pred_lo) / 2.0
    
    # Calculate novelty score (distance to nearest neighbors in training set)
    X_tr_val = X_train.select_dtypes(include=[np.number]).values
    cand_val = cand_df.select_dtypes(include=[np.number]).values
    
    # Sample training points for speed
    n_sample = min(500, len(X_tr_val))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_tr_val), size=n_sample, replace=False)
    X_tr_sample = X_tr_val[sample_idx]
    
    # Compute nearest neighbor distance
    dists = euclidean_distances(cand_val, X_tr_sample)
    novelty = dists.min(axis=1)
    
    # Scale novelty to [0, 1]
    if novelty.max() > novelty.min():
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min())
    else:
        novelty = np.zeros(len(novelty))
        
    result_df = pd.DataFrame(records)
    result_df["pred_median_log"] = preds_med
    result_df["pred_p05_log"] = pred_lo
    result_df["pred_p95_log"] = pred_hi
    result_df["ucb_log"] = ucb
    result_df["novelty_score"] = novelty
    result_df["pred_her_umol_g_h"] = np.expm1(preds_med)
    result_df["ucb_her_umol_g_h"] = np.expm1(ucb)
    
    # Sort by UCB descending
    result_df = result_df.sort_values("ucb_log", ascending=False).reset_index(drop=True)
    return result_df
