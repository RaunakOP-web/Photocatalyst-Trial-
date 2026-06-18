import os
import pandas as pd
import numpy as np
import requests
from src.features.material_descriptors import add_physical_features


def fetch_band_edges_from_mpr(material: str):
    """Fetch conduction and valence band edge potentials (vs NHE) from Materials Project.
    Requires a Materials Project API key set in the environment variable `MATERIALS_PROJECT_API_KEY`.
    Returns a tuple (cb_potential_nhe, vb_potential_nhe) or None if unavailable.
    """
    api_key = os.getenv('MATERIALS_PROJECT_API_KEY')
    if not api_key:
        return None
    # Use the Materials Project search endpoint to find material IDs by formula
    search_url = f"https://api.materialsproject.org/v2/materials/search?formula={material}&api_key={api_key}"
    try:
        resp = requests.get(search_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Pick the first entry
        if not data.get('data'):
            return None
        material_id = data['data'][0]['material_id']
        # Retrieve band structure
        band_url = f"https://api.materialsproject.org/v2/materials/{material_id}/band_structure?api_key={api_key}"
        resp = requests.get(band_url, timeout=10)
        resp.raise_for_status()
        band_data = resp.json()
        # Extract band edges (eV vs vacuum) and convert to NHE (approx -4.5 eV)
        cb_vac = band_data['data']['cbm']
        vb_vac = band_data['data']['vbm']
        cb_nhe = cb_vac - 4.5
        vb_nhe = vb_vac - 4.5
        return cb_nhe, vb_nhe
    except Exception:
        return None


def add_band_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Compute conduction/valence band potentials (vs NHE) for each host material.
    Attempts to fetch values from the Materials Project API; falls back to the simple
    electronegativity approximation if the API is unavailable or the material is not found.
    """
    # Ensure physical features are present
    if not {"semi_electron_affinity_eV", "semi_bandgap_eV"}.issubset(df.columns):
        df = add_physical_features(df)
    # Apply fetch or fallback for each host material
    cb_list = []
    vb_list = []
    for mat in df["host_material"].fillna("unknown").astype(str).str.lower():
        result = fetch_band_edges_from_mpr(mat)
        if result:
            cb, vb = result
        else:
            # Approximation used previously
            chi = df.loc[df["host_material"].str.lower() == mat, "semi_electron_affinity_eV"].iloc[0] + 0.5 * df.loc[df["host_material"].str.lower() == mat, "semi_bandgap_eV"].iloc[0]
            cb = chi - 4.5 - 0.5 * df.loc[df["host_material"].str.lower() == mat, "semi_bandgap_eV"].iloc[0]
            vb = cb + df.loc[df["host_material"].str.lower() == mat, "semi_bandgap_eV"].iloc[0]
        cb_list.append(cb)
        vb_list.append(vb)
    df["cb_potential_nhe"] = cb_list
    df["vb_potential_nhe"] = vb_list
    df["cb_overpotential_h2"] = -df["cb_potential_nhe"]
    df["vb_overpotential_glycerol"] = df["vb_potential_nhe"] - 0.8
    return df
