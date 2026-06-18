import random
import numpy as np
import pandas as pd
from itertools import product
from src.features.material_descriptors import add_physical_features

def generate_candidate_grid(feature_cols, medians, encoder):
    """Generates the combinatorial screening library of catalysts."""
    hosts = ["g-c3n4", "bivo4", "bi2wo6", "fe2o3", "wo3", "srtio3", "cu2o", "ga2o3", "in2o3", "zns", "moo3", "v2o5"]
    cocatalysts = ["pt", "ni", "cu", "co", "mos2", "ni2p", "rgo", "pd", "au", "ru", "rh", "ir", "wc", "cos2", "fes2"]
    
    grid = {
        "host_material": hosts,
        "co_catalyst": cocatalysts,
        "co_catalyst_wt_pct": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
        "glycerol_concentration_v_pct": [5.0, 10.0, 20.0, 30.0, 50.0],
        "light_type": ["visible", "uv"],
        "catalyst_loading_g_L": [0.5, 1.0, 1.5, 2.0],
    }

    keys = list(grid.keys())
    vals = list(grid.values())
    records = [dict(zip(keys, combo)) for combo in product(*vals)]
    cand_df = pd.DataFrame(records)
    
    # Map physical features
    cand_df = add_physical_features(cand_df)
    
    # Fill remaining columns with training medians or defaults
    for col in feature_cols:
        if col not in cand_df.columns:
            if col in medians.index:
                cand_df[col] = float(medians[col])
            else:
                cand_df[col] = "unknown"
                
    cand_df = cand_df.reindex(columns=feature_cols, fill_value=0.0)
    
    # Apply target encoding if present
    if encoder is not None:
        enc_cols = [c for c in encoder.feature_names_in_ if c in cand_df.columns]
        if enc_cols:
            cand_df[enc_cols] = encoder.transform(cand_df[enc_cols])
            
    cand_df = cand_df.fillna(0.0)
    return cand_df, records
