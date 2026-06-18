import pandas as pd
import numpy as np
from src.features.material_descriptors import add_physical_features

# Lookup table for ΔG_H* (eV) and related proxies for common cocatalysts
# Values taken from literature (e.g., Nørskov et al., 2005; doi:10.1021/jp050236c)
H_ADSORPTION_TABLE = {
    "pt": -0.09,
    "pd": -0.06,
    "au": 0.38,
    "ag": 0.40,
    "cu": 0.12,
    "ni": -0.15,
    "co": -0.25,
    "fe": -0.35,
    "rh": -0.12,
    "moS2": 0.15,
    "none": 0.0,
    "unknown": 0.0,
}

def add_h_adsorption(df: pd.DataFrame) -> pd.DataFrame:
    """Attach hydrogen adsorption free‑energy proxy features.

    Columns added:
        cocat_dg_h_proxy   – ΔG_H* (eV) from literature lookup.
        cocat_h_binding_strength – absolute value of ΔG_H* (proxy for binding strength).
        cocat_sabatier_distance   – |ΔG_H* + 0.24| (distance from Sabatier optimum).
    """
    # Ensure cocatalyst column exists
    if "co_catalyst" not in df.columns:
        df["co_catalyst"] = "none"
    # Normalise names
    cat_norm = df["co_catalyst"].fillna("none").astype(str).str.strip().str.lower()
    dg = cat_norm.map(H_ADSORPTION_TABLE).fillna(0.0)
    df["cocat_dg_h_proxy"] = dg
    df["cocat_h_binding_strength"] = dg.abs()
    df["cocat_sabatier_distance"] = (dg + 0.24).abs()
    # Metadata source (hard‑coded reference)
    df["cocat_dg_h_source"] = "doi:10.1021/jp050236c"
    return df
