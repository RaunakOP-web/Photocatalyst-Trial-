import numpy as np
import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds physics-informed engineered features to the dataset.
    This module replaces `engineer_features` from the v1 pipeline, 
    doing only the feature creation and returning the modified dataframe.
    Winsorization and missing value imputation should be handled in preprocessing.
    """
    df = df.copy()

    # Helper function for safe log1p
    def safe_log1p(col):
        return np.log1p(np.clip(df[col].fillna(0), 0, None))

    # --- 1.1 Charge-transfer & band alignment features ---
    if "cocat_work_function" in df.columns and "semi_electron_affinity_eV" in df.columns:
        df["schottky_barrier_eV"] = df["cocat_work_function"] - df["semi_electron_affinity_eV"]
    
    if "schottky_barrier_eV" in df.columns and "cocat_d_band_center" in df.columns:
        df["her_driving_force"] = df["schottky_barrier_eV"] * (-df["cocat_d_band_center"])
        
    if "cocat_work_function" in df.columns and "semi_bandgap_eV" in df.columns:
        df["wf_bg_ratio"] = df["cocat_work_function"] / (df["semi_bandgap_eV"].replace(0, np.nan))
        
    if "semi_bandgap_eV" in df.columns and "cocat_electronegativity" in df.columns:
        df["bg_eneg_interaction"] = df["semi_bandgap_eV"] * df["cocat_electronegativity"]
        
    if "semi_bandgap_eV" in df.columns:
        df["bandgap_excess_eV"] = df["semi_bandgap_eV"] - 1.23
        df["bandgap_above_visible"] = np.maximum(df["semi_bandgap_eV"] - 3.1, 0)
        
    # --- 1.2 Photon absorption efficiency features ---
    if "semi_bandgap_eV" in df.columns and "is_visible_light" in df.columns:
        df["bandgap_visible_match"] = ((df["semi_bandgap_eV"] <= 3.1) & (df["is_visible_light"] == 1)).astype(int)
        
    if "light_power_W" in df.columns:
        df["log_light_power"] = safe_log1p("light_power_W")
        
    if "light_power_W" in df.columns and "wavelength_cutoff_nm" in df.columns:
        df["photon_flux_proxy"] = df["light_power_W"] / (df["wavelength_cutoff_nm"].replace(0, np.nan))
        
    if "light_power_W" in df.columns and "catalyst_loading_mg" in df.columns:
        df["light_per_loading"] = df["light_power_W"] / (df["catalyst_loading_mg"].replace(0, np.nan))
        
    if "semi_bandgap_eV" in df.columns:
        if "is_xe_lamp" in df.columns:
            df["bg_xe_interaction"] = df["semi_bandgap_eV"] * df["is_xe_lamp"]
        if "is_hg_lamp" in df.columns:
            df["bg_hg_interaction"] = df["semi_bandgap_eV"] * df["is_hg_lamp"]
        if "is_led" in df.columns:
            df["bg_led_interaction"] = df["semi_bandgap_eV"] * df["is_led"]

    # --- 1.3 Catalyst loading and reaction volume features ---
    if "catalyst_loading_mg" in df.columns and "reaction_volume_mL" in df.columns:
        df["loading_conc_mg_mL"] = df["catalyst_loading_mg"] / (df["reaction_volume_mL"].replace(0, np.nan))
        
    if "catalyst_loading_mg" in df.columns:
        df["log_loading_mg"] = safe_log1p("catalyst_loading_mg")
        
    if "catalyst_loading_g_L" in df.columns:
        df["log_loading_gL"] = safe_log1p("catalyst_loading_g_L")
        
    if "reaction_volume_mL" in df.columns:
        df["log_reaction_vol"] = safe_log1p("reaction_volume_mL")

    # --- 1.4 Sacrificial reagent features ---
    if "glycerol_concentration_v_pct" in df.columns:
        df["log_glycerol_conc"] = safe_log1p("glycerol_concentration_v_pct")
        
    if "glycerol_concentration_v_pct" in df.columns and "reaction_volume_mL" in df.columns:
        df["glycerol_total_proxy"] = df["glycerol_concentration_v_pct"] * df["reaction_volume_mL"]
        df["log_glycerol_total"] = safe_log1p("glycerol_total_proxy")
        
    if "glycerol_concentration_v_pct" in df.columns and "catalyst_loading_g_L" in df.columns:
        df["glycerol_per_loading"] = df["glycerol_concentration_v_pct"] / (df["catalyst_loading_g_L"].replace(0, np.nan))

    # --- 1.5 Heterojunction features ---
    if "semiconductor_2" in df.columns:
        df["has_heterojunction"] = df["semiconductor_2"].notna() & (df["semiconductor_2"].astype(str).str.strip() != "") & (df["semiconductor_2"].astype(str).str.strip() != "nan")
        df["has_heterojunction"] = df["has_heterojunction"].astype(int)
        
    if "semiconductor_1_pct" in df.columns and "semiconductor_2_pct" in df.columns:
        df["semi_fraction_ratio"] = df["semiconductor_1_pct"] / (df["semiconductor_2_pct"].fillna(0) + 1)

    # --- 1.6 Cocatalyst loading features ---
    if "co_catalyst_wt_pct" in df.columns:
        df["log_cocat_loading"] = safe_log1p("co_catalyst_wt_pct")
        df["cocat_loading_from_opt"] = (df["co_catalyst_wt_pct"] - 1.5).abs()

    # --- 1.7 Semiconductor material features ---
    if "semi_dielectric" in df.columns and "semi_density" in df.columns:
        df["semi_pol_density"] = df["semi_dielectric"] * df["semi_density"]

    # --- 1.8 Preparation condition features ---
    if "calcination_temp_semiconductor_C" in df.columns:
        df["log_calc_temp_semi"] = safe_log1p("calcination_temp_semiconductor_C")
    if "calcination_temp_photocatalyst_C" in df.columns:
        df["log_calc_temp_photo"] = safe_log1p("calcination_temp_photocatalyst_C")

    # --- 1.9 Binary flags ---
    if "co_catalyst" in df.columns:
        df["has_cocatalyst"] = df["co_catalyst"].notna() & (df["co_catalyst"].astype(str).str.strip() != "") & (df["co_catalyst"].astype(str).str.strip() != "nan")
        df["has_cocatalyst"] = df["has_cocatalyst"].astype(int)
        
        noble_metals = {'pt', 'pd', 'au', 'ag', 'rh', 'ir', 'ru', 'ruo2'}
        df["is_noble_metal"] = df["co_catalyst"].astype(str).str.lower().isin(noble_metals).astype(int)

    # --- 1.10 Temporal and environmental features ---
    if "year" in df.columns:
        df["year_norm"] = (df["year"] - 2000) / 25.0
        
    if "temperature_C" in df.columns:
        df["temp_kelvin"] = df["temperature_C"] + 273.15
        
    return df
