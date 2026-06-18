import os
import numpy as np
import pandas as pd
import joblib

from src.utils.config import get_train_config
from src.utils.logging import setup_logger
from src.utils.io import safe_load_csv, safe_load_joblib, safe_save_csv
from src.features.interaction_features import add_interaction_features, add_domain_features
from src.discovery.candidate_generation import generate_candidate_grid
from src.discovery.ranking import rank_candidates
from src.applicability_domain import encode_discovery_candidates, compute_knn_distance, compute_mahalanobis_distance, compute_leverage, K

logger = setup_logger(__name__)
CFG = get_train_config()

def main():
    proc_dir = CFG["paths"]["proc_dir"]
    results_dir = CFG["paths"]["results_dir"]
    models_dir = CFG["paths"]["models_dir"]
    
    # 1. Load splits and files
    X_train = safe_load_csv(os.path.join(proc_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    
    feature_cols = safe_load_joblib(os.path.join(proc_dir, "feature_list.joblib"))
    medians = safe_load_joblib(os.path.join(proc_dir, "numeric_medians.joblib"))
    
    encoder_path = os.path.join(models_dir, "target_encoder.joblib")
    encoder = safe_load_joblib(encoder_path) if os.path.exists(encoder_path) else None
    
    # Load conformal model
    import sys
    from src.conformal import MAPIEWrapper
    sys.modules['__main__'].MAPIEWrapper = MAPIEWrapper
    
    conformal_model_path = os.path.join(models_dir, "conformal_model.joblib")
    if not os.path.exists(conformal_model_path):
        raise FileNotFoundError(f"Conformal model not found at {conformal_model_path}. Please run conformal.py first.")
    mapie = safe_load_joblib(conformal_model_path)
    best_model = mapie._estimator.model # extract the BlendingEnsemble inside MAPIEWrapper
    
    # 2. Generate candidate grid
    print("Generating combinatorial candidate grid (43,200 catalysts)...")
    cand_df, records = generate_candidate_grid(feature_cols, medians, encoder)
    
    # Predict Log-HER and intervals using conformal model
    X_cand = cand_df.copy()
    X_cand, _ = add_interaction_features(X_cand, X_cand)
    X_cand, _ = add_domain_features(X_cand, X_cand)
    X_cand_base = X_cand[best_model.active_features].copy()
    
    print("Predicting point estimates and conformal intervals...")
    pred_med = mapie.predict(X_cand_base)
    lower_arr, upper_arr = mapie.predict_interval(X_cand_base)
    pred_lo = upper_arr[:, 0, 0]
    pred_hi = upper_arr[:, 1, 0]
    
    # UCB Acquisition Function
    kappa = 1.0
    ucb = pred_med + kappa * (pred_hi - pred_lo) / 2.0
    
    # 3. Applicability Domain Check
    print("Evaluating applicability domain of candidates...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
    
    # Prep train base features
    X_train_encoded = X_train.copy()
    if encoder is not None:
        cat_cols = safe_load_joblib(os.path.join(proc_dir, "cat_cols.joblib"))
        X_train_encoded[cat_cols] = encoder.transform(X_train[cat_cols])
    X_train_int, _ = add_interaction_features(X_train_encoded, X_train_encoded)
    X_train_int, _ = add_domain_features(X_train_int, X_train_int, y_train)
    X_train_base = X_train_int[best_model.active_features].copy()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_base.values)
    
    X_disc_scaled = scaler.transform(X_cand_base.values)
    
    # Compute AD metrics
    nn = NearestNeighbors(n_neighbors=K+1, metric="euclidean")
    nn.fit(X_train_scaled)
    train_knn_dists, _ = nn.kneighbors(X_train_scaled)
    train_knn_scores = train_knn_dists[:, 1:].mean(axis=1)
    knn_threshold = train_knn_scores.mean() + 2.0 * train_knn_scores.std()
    disc_knn = compute_knn_distance(X_train_scaled, X_disc_scaled)
    
    iso = IsolationForest(contamination=0.04, random_state=42)
    iso.fit(X_train_scaled)
    disc_iso = iso.score_samples(X_disc_scaled)
    iso_threshold = np.percentile(iso.score_samples(X_train_scaled), 5)
    
    p = X_train_scaled.shape[1] + 1
    n = len(X_train_scaled)
    leverage_threshold = 3.0 * p / n
    disc_lev = compute_leverage(X_train_scaled, X_disc_scaled)
    
    train_mahal = compute_mahalanobis_distance(X_train_scaled, X_train_scaled)
    mahal_threshold = train_mahal.mean() + 2.0 * train_mahal.std()
    disc_mahal = compute_mahalanobis_distance(X_train_scaled, X_disc_scaled)
    
    ins_knn = disc_knn <= knn_threshold
    ins_iso = disc_iso >= iso_threshold
    ins_lev = disc_lev <= leverage_threshold
    ins_mahal = disc_mahal <= mahal_threshold
    
    inside_sum = ins_knn.astype(int) + ins_iso.astype(int) + ins_lev.astype(int) + ins_mahal.astype(int)
    
    def get_ad_label(score):
        if score == 4:
            return "Reliable"
        elif score >= 2:
            return "Moderate Confidence"
        else:
            return "Outside Domain"
            
    disc_labels = [get_ad_label(s) for s in inside_sum]
    
    # 4. Rank and Filter Candidates
    result_df = pd.DataFrame(records)
    result_df["pred_median_log"] = pred_med
    result_df["pred_p05_log"] = pred_lo
    result_df["pred_p95_log"] = pred_hi
    result_df["ucb_log"] = ucb
    result_df["ad_score"] = inside_sum
    result_df["ad_label"] = disc_labels
    result_df["pred_her_umol_g_h"] = np.expm1(pred_med)
    result_df["pred_her_lower_umol_g_h"] = np.expm1(pred_lo)
    result_df["pred_her_upper_umol_g_h"] = np.expm1(pred_hi)
    result_df["ucb_her_umol_g_h"] = np.expm1(ucb)
    
    # Calculate Novelty (nearest-neighbor distance)
    # Scale novelty to [0, 1]
    if disc_knn.max() > disc_knn.min():
        novelty = (disc_knn - disc_knn.min()) / (disc_knn.max() - disc_knn.min())
    else:
        novelty = np.zeros(len(disc_knn))
    result_df["novelty_score"] = novelty
    
    # Save raw virtual library screening results
    os.makedirs(results_dir, exist_ok=True)
    safe_save_csv(result_df, os.path.join(results_dir, "discovery_candidates.csv"))
    
    # Filter to only keep candidates INSIDE the applicability domain
    filtered_df = result_df[result_df["ad_label"] != "Outside Domain"].copy()
    
    # Sort by UCB descending and keep top 50
    top50 = filtered_df.sort_values("ucb_log", ascending=False).drop_duplicates(
        subset=["host_material", "co_catalyst"], keep="first"
    ).head(50)
    
    # Save outputs
    safe_save_csv(top50, os.path.join(results_dir, "optimized_candidates.csv"))
    safe_save_csv(top50, "optimized_candidates.csv")
    
    logger.info(f"Bayesian-style UCB screening complete. Top 50 domain-validated catalysts saved to optimized_candidates.csv")
    print(top50[["host_material", "co_catalyst", "pred_her_umol_g_h", "pred_her_lower_umol_g_h", "pred_her_upper_umol_g_h", "ad_label"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
