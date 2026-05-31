"""
material_features.py
Encodes semiconductors and co-catalysts using physical/chemical
properties rather than target encoding. Works for ANY material
including ones not seen during training.
"""

import pandas as pd
import numpy as np

# ── SEMICONDUCTOR PHYSICAL PROPERTIES ───────────────────────────
# Source: well-established values from semiconductor literature
# bandgap_eV, electron_affinity_eV, dielectric_constant,
# crystal_structure (encoded: 1=anatase, 2=rutile, 3=wurtzite,
# 4=cubic, 5=monoclinic, 6=layered, 7=perovskite, 8=scheelite,
# 9=bismuth-based), density_g_cm3

SEMICONDUCTOR_PROPS = {
    "tio2":              {"bandgap_eV": 3.20, "electron_affinity_eV": 4.21,
                          "dielectric": 86.0, "crystal": 1, "density": 3.89},
    "tio2_p25":          {"bandgap_eV": 3.15, "electron_affinity_eV": 4.21,
                          "dielectric": 80.0, "crystal": 1, "density": 3.90},
    "zno":               {"bandgap_eV": 3.37, "electron_affinity_eV": 4.35,
                          "dielectric": 8.5,  "crystal": 3, "density": 5.61},
    "cds":               {"bandgap_eV": 2.42, "electron_affinity_eV": 4.20,
                          "dielectric": 8.9,  "crystal": 3, "density": 4.82},
    "g-c3n4":            {"bandgap_eV": 2.70, "electron_affinity_eV": 4.30,
                          "dielectric": 3.0,  "crystal": 6, "density": 1.60},
    "ceo2":              {"bandgap_eV": 3.10, "electron_affinity_eV": 4.40,
                          "dielectric": 26.0, "crystal": 4, "density": 7.22},
    "wo3":               {"bandgap_eV": 2.70, "electron_affinity_eV": 5.24,
                          "dielectric": 42.0, "crystal": 5, "density": 7.16},
    "bivo4":             {"bandgap_eV": 2.40, "electron_affinity_eV": 3.70,
                          "dielectric": 68.0, "crystal": 8, "density": 6.83},
    "fe2o3":             {"bandgap_eV": 2.10, "electron_affinity_eV": 5.88,
                          "dielectric": 14.2, "crystal": 4, "density": 5.24},
    "srtio3":            {"bandgap_eV": 3.20, "electron_affinity_eV": 3.90,
                          "dielectric": 300.0,"crystal": 7, "density": 5.12},
    "bi2wo6":            {"bandgap_eV": 2.80, "electron_affinity_eV": 3.85,
                          "dielectric": 32.0, "crystal": 6, "density": 9.00},
    "la0.2na0.98tao3":   {"bandgap_eV": 4.00, "electron_affinity_eV": 3.50,
                          "dielectric": 45.0, "crystal": 7, "density": 7.80},
    "pbtio3":            {"bandgap_eV": 3.40, "electron_affinity_eV": 3.50,
                          "dielectric": 150.0,"crystal": 7, "density": 7.99},
    "zns":               {"bandgap_eV": 3.54, "electron_affinity_eV": 3.90,
                          "dielectric": 8.3,  "crystal": 3, "density": 4.09},
    "cuo":               {"bandgap_eV": 1.70, "electron_affinity_eV": 4.07,
                          "dielectric": 18.1, "crystal": 5, "density": 6.31},
    "sral2o4":           {"bandgap_eV": 5.80, "electron_affinity_eV": 1.50,
                          "dielectric": 9.8,  "crystal": 5, "density": 3.50},
    "moo3":              {"bandgap_eV": 2.90, "electron_affinity_eV": 6.70,
                          "dielectric": 12.5, "crystal": 6, "density": 4.69},
    "in2o3":             {"bandgap_eV": 2.90, "electron_affinity_eV": 5.00,
                          "dielectric": 9.0,  "crystal": 4, "density": 7.18},
    "nb2o5":             {"bandgap_eV": 3.40, "electron_affinity_eV": 4.30,
                          "dielectric": 41.0, "crystal": 5, "density": 4.55},
    "sno2":              {"bandgap_eV": 3.60, "electron_affinity_eV": 4.90,
                          "dielectric": 14.0, "crystal": 4, "density": 6.99},
    "znin2s4":           {"bandgap_eV": 2.34, "electron_affinity_eV": 4.10,
                          "dielectric": 9.2,  "crystal": 6, "density": 3.85},
    "unknown":           {"bandgap_eV": 3.00, "electron_affinity_eV": 4.20,
                          "dielectric": 20.0, "crystal": 1, "density": 5.00},
}

# ── CO-CATALYST PHYSICAL PROPERTIES ─────────────────────────────
# work_function_eV, d_band_center_eV (vs vacuum), atomic_radius_pm,
# electronegativity (Pauling), price_index (1=cheap, 5=expensive)

COCATALYST_PROPS = {
    "pt":     {"work_function": 5.65, "d_band_center": -2.25,
                "atomic_radius": 139, "electronegativity": 2.28, "price": 5},
    "pd":     {"work_function": 5.22, "d_band_center": -1.83,
                "atomic_radius": 137, "electronegativity": 2.20, "price": 5},
    "au":     {"work_function": 5.10, "d_band_center": -3.56,
                "atomic_radius": 144, "electronegativity": 2.54, "price": 5},
    "ag":     {"work_function": 4.26, "d_band_center": -4.30,
                "atomic_radius": 144, "electronegativity": 1.93, "price": 4},
    "cu":     {"work_function": 4.65, "d_band_center": -2.67,
                "atomic_radius": 128, "electronegativity": 1.90, "price": 2},
    "ni":     {"work_function": 5.15, "d_band_center": -1.29,
                "atomic_radius": 124, "electronegativity": 1.91, "price": 2},
    "co":     {"work_function": 5.00, "d_band_center": -1.17,
                "atomic_radius": 125, "electronegativity": 1.88, "price": 2},
    "fe":     {"work_function": 4.67, "d_band_center": -0.92,
                "atomic_radius": 126, "electronegativity": 1.83, "price": 1},
    "rh":     {"work_function": 4.98, "d_band_center": -1.73,
                "atomic_radius": 134, "electronegativity": 2.28, "price": 5},
    "ir":     {"work_function": 5.27, "d_band_center": -2.11,
                "atomic_radius": 136, "electronegativity": 2.20, "price": 5},
    "ru":     {"work_function": 4.71, "d_band_center": -1.41,
                "atomic_radius": 134, "electronegativity": 2.20, "price": 4},
    "nise2":  {"work_function": 4.70, "d_band_center": -1.40,
                "atomic_radius": 124, "electronegativity": 1.91, "price": 2},
    "nio":    {"work_function": 4.90, "d_band_center": -1.50,
                "atomic_radius": 124, "electronegativity": 1.91, "price": 1},
    "cuo":    {"work_function": 4.60, "d_band_center": -2.50,
                "atomic_radius": 128, "electronegativity": 1.90, "price": 1},
    "mos2":   {"work_function": 5.00, "d_band_center": -1.00,
                "atomic_radius": 139, "electronegativity": 2.16, "price": 2},
    "none":   {"work_function": 0.00, "d_band_center": 0.00,
                "atomic_radius": 0,   "electronegativity": 0.00, "price": 0},
    "unknown":{"work_function": 4.90, "d_band_center": -2.00,
                "atomic_radius": 130, "electronegativity": 2.00, "price": 3},
}

def encode_semiconductor(name: str) -> dict:
    """Return physical property dict for a semiconductor."""
    name_clean = str(name).strip().lower()
    props = SEMICONDUCTOR_PROPS.get(name_clean,
                                     SEMICONDUCTOR_PROPS["unknown"])
    return {f"semi_{k}": v for k, v in props.items()}

def encode_cocatalyst(name: str) -> dict:
    """Return physical property dict for a co-catalyst."""
    name_clean = str(name).strip().lower() if (name and str(name).strip().lower() != "nan") else "none"
    props = COCATALYST_PROPS.get(name_clean,
                                  COCATALYST_PROPS["unknown"])
    return {f"cocat_{k}": v for k, v in props.items()}

def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace target-encoded category columns with physical property columns.
    Input df must have 'host_material' and 'co_catalyst' columns.
    Returns df with new physics columns added.
    """
    # Create copies or work with temporary Series to avoid index/alignment issues
    semi_feats  = df["host_material"].apply(
        lambda x: pd.Series(encode_semiconductor(x))
    )
    cocat_feats = df["co_catalyst"].fillna("none").apply(
        lambda x: pd.Series(encode_cocatalyst(x))
    )
    
    # Concatenate the new physical properties
    df_new = pd.concat([df, semi_feats, cocat_feats], axis=1)
    return df_new

if __name__ == "__main__":
    test_materials = ["TiO2", "BiVO4", "Fe2O3", "SrTiO3", "WO3", "unknown"]
    for m in test_materials:
        props = encode_semiconductor(m)
        print(f"{m}: bandgap={props['semi_bandgap_eV']}eV "
              f"EA={props['semi_electron_affinity_eV']}eV")
    print("\nTest co-catalysts:")
    for c in ["Pt", "Pd", "Cu", "none", "unknown"]:
        props = encode_cocatalyst(c)
        print(f"{c}: WF={props['cocat_work_function']}eV "
              f"d-band={props['cocat_d_band_center']}eV")
