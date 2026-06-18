import numpy as np
import pandas as pd

def add_interaction_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Computes 7 SHAP-guided interaction features."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    def ensure_log_features(df):
        df = df.copy()
        if "log_calc_temp_semi" not in df.columns and "calcination_temp_semiconductor_C" in df.columns:
            df["log_calc_temp_semi"] = np.log1p(np.clip(df["calcination_temp_semiconductor_C"].fillna(0), 0, None))
        if "log_cocat_loading" not in df.columns and "co_catalyst_wt_pct" in df.columns:
            df["log_cocat_loading"] = np.log1p(np.clip(df["co_catalyst_wt_pct"].fillna(0), 0, None))
        if "log_light_power" not in df.columns and "light_power_W" in df.columns:
            df["log_light_power"] = np.log1p(np.clip(df["light_power_W"].fillna(0), 0, None))
        if "log_loading_mg" not in df.columns and "catalyst_loading_mg" in df.columns:
            df["log_loading_mg"] = np.log1p(np.clip(df["catalyst_loading_mg"].fillna(0), 0, None))
        if "log_loading_gL" not in df.columns and "catalyst_loading_g_L" in df.columns:
            df["log_loading_gL"] = np.log1p(np.clip(df["catalyst_loading_g_L"].fillna(0), 0, None))
        if "log_glycerol_total" not in df.columns and "glycerol_concentration_v_pct" in df.columns and "reaction_volume_mL" in df.columns:
            total_gly = df["glycerol_concentration_v_pct"].fillna(0) * df["reaction_volume_mL"].fillna(0)
            df["log_glycerol_total"] = np.log1p(np.clip(total_gly, 0, None))
        return df

    X_train = ensure_log_features(X_train)
    X_test = ensure_log_features(X_test)

    for df in [X_train, X_test]:
        for col in ["structure", "log_calc_temp_semi", "preparation_photocatalyst", 
                    "log_light_power", "log_loading_mg", "log_loading_gL", 
                    "log_cocat_loading", "log_glycerol_total"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        df["structure_x_calc_temp"] = df["structure"] * df["log_calc_temp_semi"]
        df["structure_x_prep"] = df["structure"] * df["preparation_photocatalyst"]
        df["prep_x_calc_temp"] = df["preparation_photocatalyst"] * df["log_calc_temp_semi"]
        if "log_loading_mg" in df.columns:
            df["light_x_loading"] = df["log_light_power"] * df["log_loading_mg"]
        else:
            df["light_x_loading"] = df["log_light_power"] * df["log_loading_gL"]
        df["structure_x_cocat_loading"] = df["structure"] * df["log_cocat_loading"]
        df["calc_temp_squared"] = df["log_calc_temp_semi"] ** 2
        df["glycerol_x_light"] = df["log_glycerol_total"] * df["log_light_power"]

    return X_train, X_test

def add_domain_features(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train=None):
    """Computes domain-specific band structure and reaction condition interaction features."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    cols_before_domain = X_train.shape[1]
    for df in [X_train, X_test]:
        if "bandgap_eV" in df.columns:
            bg = df["bandgap_eV"].fillna(3.0)
            df["bandgap_visible_overlap"] = np.clip((2.76 - bg) / 2.76, 0, None)
            df["bandgap_uv_overlap"] = np.clip((bg - 3.1) / 3.0, 0, None)
            
        if "light_power_W" in df.columns:
            df["photon_flux_proxy"] = df["light_power_W"].fillna(0)
        
        if "catalyst_loading_g_L" in df.columns:
            df["catalyst_loading_log"] = np.log1p(np.clip(df["catalyst_loading_g_L"].fillna(0), 0, None))
        elif "catalyst_loading_mg" in df.columns:
            df["catalyst_loading_log"] = np.log1p(np.clip(df["catalyst_loading_mg"].fillna(0), 0, None))
        else:
            df["catalyst_loading_log"] = 0.0
            
        if "pH" in df.columns:
            df["pH_neutral_distance"] = np.abs(df["pH"].fillna(7.0) - 7.0)
            
        # Catalyst/donor ratio
        if "catalyst_loading_g_L" in df.columns and "glycerol_concentration_v_pct" in df.columns:
            df["cat_donor_ratio"] = df["catalyst_loading_g_L"].fillna(0) / (df["glycerol_concentration_v_pct"].fillna(0) + 1e-5)
        elif "catalyst_loading_mg" in df.columns and "glycerol_concentration_v_pct" in df.columns:
            df["cat_donor_ratio"] = df["catalyst_loading_mg"].fillna(0) / (df["glycerol_concentration_v_pct"].fillna(0) + 1e-5)
        else:
            df["cat_donor_ratio"] = 0.0
            
        if "semiconductor_2_pct" in df.columns:
            df["is_heterojunction"] = (df["semiconductor_2_pct"].fillna(0) > 0).astype(int)
        else:
            df["is_heterojunction"] = 0
            
        if "co_catalyst_wt_pct" in df.columns:
            df["cocatalyst_loading_log"] = np.log1p(np.clip(df["co_catalyst_wt_pct"].fillna(0), 0, None))
        else:
            df["cocatalyst_loading_log"] = 0.0

            
        top5 = ["light_source_type", "structure_x_cocat_loading", "semiconductor_2", "log_glycerol_total", "structure_x_prep"]
        top5_present = [f for f in top5 if f in df.columns]
        for f in top5_present:
            df[f"{f}_squared"] = df[f] ** 2
        for i in range(len(top5_present)):
            for j in range(i+1, len(top5_present)):
                f1, f2 = top5_present[i], top5_present[j]
                df[f"{f1}_x_{f2}"] = df[f1] * df[f2]

    # Variance filtering
    if y_train is not None:
        added_cols = [c for c in X_train.columns if c not in X_test.columns or c not in X_train.columns[:cols_before_domain]]
        for col in added_cols:
            var = X_train[col].var()
            if pd.isna(var) or var < 1e-6:
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])

    if y_train is not None:
        y_mean = y_train.mean()
        y_std = y_train.std()
        is_extreme_train = (np.abs(y_train - y_mean) > 3 * y_std).astype(int)
        X_train["is_extreme_target"] = is_extreme_train.values
        X_test["is_extreme_target"] = 0
    else:
        X_train["is_extreme_target"] = 0
        X_test["is_extreme_target"] = 0
        
    return X_train, X_test
