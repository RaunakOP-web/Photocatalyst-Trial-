import pandas as pd
import numpy as np

# Placeholder surface area estimation based on morphology
# This simple heuristic uses a predefined mapping of morphology types to a
# representative surface area (m²/g). In a real implementation, one would
# compute the surface area from particle size distributions or retrieve
# literature values.
MORPHOLOGY_SURFACE_MAP = {
    "nanorod": 50,
    "nanowire": 45,
    "sheet": 30,
    "particle": 10,
    "nanoparticle": 20,
    "nanoflake": 25,
}


def add_surface_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add surface‑area based descriptors.

    * ``surface_area_est`` – estimated surface area (m²/g) based on the
      ``morphology`` column.
    * ``surface_area_normalized_HER`` – HER divided by the estimated
      surface area (if ``HER`` column exists).
    """
    if "morphology" in df.columns:
        df["surface_area_est"] = df["morphology"].map(MORPHOLOGY_SURFACE_MAP).astype(float)
    else:
        df["surface_area_est"] = np.nan

    # Normalized HER (if HER column present)
    if "HER" in df.columns:
        # Avoid division by zero
        df["surface_area_normalized_HER"] = df["HER"] / (df["surface_area_est"] + 1e-6)
    else:
        df["surface_area_normalized_HER"] = np.nan

    # Provenance columns
    df["surface_area_source"] = "derived from morphology heuristic"
    df["surface_area_uncertainty"] = df["surface_area_est"].isna()
    return df
