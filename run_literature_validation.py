import os
import pandas as pd
import numpy as np
from src.utils.config import get_train_config
from src.utils.io import safe_load_csv, safe_save_csv

def main():
    CFG = get_train_config()
    proc_dir = CFG["paths"]["proc_dir"]
    results_dir = CFG["paths"]["results_dir"]
    
    # 1. Load data
    df_clean = pd.read_csv(os.path.join(proc_dir, "df_clean.csv"))
    top_candidates = pd.read_csv("optimized_candidates.csv")
    
    # Set of existing combinations
    existing_combos = set(zip(df_clean["host_material"].str.lower(), df_clean["co_catalyst"].str.lower()))
    existing_hosts = set(df_clean["host_material"].str.lower())
    existing_cocats = set(df_clean["co_catalyst"].str.lower())
    
    classification = []
    novelty_scores = []
    reasons = []
    
    for idx, row in top_candidates.iterrows():
        host = str(row["host_material"]).lower()
        cocat = str(row["co_catalyst"]).lower()
        
        # Check for exact match
        if (host, cocat) in existing_combos:
            cls = "Previously Reported"
            score = 0.05
            reason = "Exact host-cocatalyst pair exists in dataset"
        elif host in existing_hosts and cocat in existing_cocats:
            cls = "Similar to Reported"
            score = 0.40
            reason = f"Host '{host}' and cocatalyst '{cocat}' both exist in dataset but not in this specific combination"
        elif host in existing_hosts:
            cls = "Potentially Novel"
            score = 0.75
            reason = f"Host '{host}' exists, but cocatalyst '{cocat}' is novel for this system"
        else:
            cls = "Potentially Novel"
            score = 0.95
            reason = f"Host '{host}' is not in the training dataset"
            
        classification.append(cls)
        # Combine distance-based novelty with heuristic novelty
        # Adjust score slightly using the computed novelty_score if present
        if "novelty_score" in row:
            final_score = 0.7 * score + 0.3 * row["novelty_score"]
        else:
            final_score = score
            
        novelty_scores.append(round(float(final_score), 3))
        reasons.append(reason)
        
    assessment_df = pd.DataFrame({
        "host_material": top_candidates["host_material"],
        "co_catalyst": top_candidates["co_catalyst"],
        "pred_her_umol_g_h": top_candidates["pred_her_umol_g_h"],
        "pred_her_lower_umol_g_h": top_candidates["pred_her_lower_umol_g_h"],
        "pred_her_upper_umol_g_h": top_candidates["pred_her_upper_umol_g_h"],
        "ad_label": top_candidates["ad_label"],
        "novelty_class": classification,
        "novelty_score": novelty_scores,
        "reasoning": reasons
    })
    
    # Save files
    safe_save_csv(assessment_df, "novelty_assessment.csv")
    safe_save_csv(assessment_df, os.path.join(results_dir, "novelty_assessment.csv"))
    
    print("Literature validation and novelty assessment complete. Saved novelty_assessment.csv")
    print(assessment_df[["host_material", "co_catalyst", "novelty_class", "novelty_score"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
