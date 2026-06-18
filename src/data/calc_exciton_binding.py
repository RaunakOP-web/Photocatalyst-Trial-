import pandas as pd
import numpy as np
# This script expects that the dataframe already contains
#   - electron_effective_mass
#   - hole_effective_mass
#   - semi_dielectric
# which are added by `calc_carrier_mobility` and `add_physical_features`.

def add_exciton_binding(df: pd.DataFrame) -> pd.DataFrame:
    """Compute an exciton binding energy proxy and charge‑separation score.

    Reduced effective mass μ = (m_e * m_h) / (m_e + m_h)
    Exciton binding proxy E_b ∝ μ / ε²
    Charge‑separation score = 1 / E_b (higher is better).
    An uncertainty flag is set when dielectric constant is missing.
    """
    required = {"electron_effective_mass", "hole_effective_mass", "semi_dielectric"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for exciton binding calculation: {missing}")

    # Reduced effective mass μ
    df["reduced_effective_mass"] = (
        df["electron_effective_mass"] * df["hole_effective_mass"]
    ) / (df["electron_effective_mass"] + df["hole_effective_mass"])

    # Exciton binding energy proxy (arbitrary scaling factor omitted)
    df["exciton_binding_proxy"] = df["reduced_effective_mass"] / (df["semi_dielectric"] ** 2 + 1e-6)
    df["charge_separation_score"] = 1.0 / (df["exciton_binding_proxy"] + 1e-6)

    # Uncertainty flag if dielectric constant is NaN or zero
    df["exciton_binding_uncertainty"] = df["semi_dielectric"].isna() | (df["semi_dielectric"] == 0)
    df["exciton_binding_source"] = "derived from effective masses and dielectric constant"
    return df
