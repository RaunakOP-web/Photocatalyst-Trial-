import pandas as pd
import numpy as np
from src.features.material_descriptors import add_physical_features

# Effective mass lookup (electron & hole) – values from literature (placeholder values)
# Units: electron rest mass (m0). In real implementation replace with accurate data source.
EFFECTIVE_MASS_TABLE = {
    "tio2": (0.5, 0.8),
    "zno": (0.3, 0.6),
    "cds": (0.4, 0.7),
    "g-c3n4": (1.0, 1.2),
    "wo3": (0.6, 0.9),
    "fe2o3": (0.9, 1.0),
    "unknown": (1.0, 1.0),
}

def add_carrier_mobility(df: pd.DataFrame) -> pd.DataFrame:
    """Add charge transport proxy descriptors.

    Columns added:
        electron_effective_mass
        hole_effective_mass
        mobility_ratio
        carrier_transport_score
        carrier_mobility_uncertainty (bool)
    """
    # Ensure semiconductor physical features are present
    if not {"semi_electron_affinity_eV", "semi_bandgap_eV"}.issubset(df.columns):
        df = add_physical_features(df)
    # Normalise host material name
    host_norm = df["host_material"].fillna("unknown").astype(str).str.strip().str.lower()
    eff_masses = host_norm.map(EFFECTIVE_MASS_TABLE).apply(lambda x: x if isinstance(x, tuple) else (1.0, 1.0))
    df["electron_effective_mass"] = eff_masses.apply(lambda t: t[0])
    df["hole_effective_mass"] = eff_masses.apply(lambda t: t[1])
    df["mobility_ratio"] = df["electron_effective_mass"] / df["hole_effective_mass"]
    # Simple transport score – higher when both masses are low
    df["carrier_transport_score"] = 1.0 / (df["electron_effective_mass"] * df["hole_effective_mass"])
    # Uncertainty flag when we used fallback values (i.e., unknown material)
    df["carrier_mobility_uncertainty"] = host_norm.apply(lambda x: x not in EFFECTIVE_MASS_TABLE)
    df["carrier_mobility_source"] = "literature placeholder; see EFFECTIVE_MASS_TABLE"
    return df
