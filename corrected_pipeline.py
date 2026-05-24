import re
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor

# Ensure UTF-8 output encoding for Windows compatibility
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# STEP 1: CORRECTED DATASET SCHEMA & AUGMENTATION
# ==========================================

def extract_loading(composition_raw):
    """Helper to extract co-catalyst loading in wt% from composition string."""
    match = re.search(r'([\d.]+)\s*wt%', str(composition_raw))
    if match:
        return float(match.group(1))
    return 0.0

def load_and_augment_dataset():
    """Loads the base clean catalyst CSV and augments it with modern optical,
    charge transport, and environmental features.
    """
    try:
        base_df = pd.read_csv("catalyst_dataset_clean.csv")
    except FileNotFoundError:
        print("Error: catalyst_dataset_clean.csv not found. Run data_clean.py first.")
        sys.exit(1)
        
    df_aug = pd.DataFrame()
    df_aug["Catalyst"] = base_df["Catalyst"]
    df_aug["composition"] = base_df["composition"]
    
    # 1. Optical Bandgap (eV)
    df_aug["Bandgap_eV"] = base_df["Bandgap_eV"]
    
    # 2. Absorption Edge Wavelength (nm)
    df_aug["Absorption_edge_nm"] = 1240.0 / df_aug["Bandgap_eV"]
    
    # 3. Specific Surface Area (BET m2/g)
    df_aug["BET_m2_g"] = base_df["BET_m2_g"]
    
    # 4. Charge Carrier Lifetime (ns)
    lifetimes = []
    for idx, row in base_df.iterrows():
        raw_life = str(row["Carrier_Lifetime_Raw"])
        match = re.search(r'([\d.]+)\s*ns', raw_life)
        if match:
            lifetimes.append(float(match.group(1)))
        else:
            fam = row["Family"]
            if fam == "TiO2":
                lifetimes.append(20.0)
            elif fam == "g-C3N4":
                lifetimes.append(3.0)
            elif fam == "CdS":
                lifetimes.append(1.5)
            else:
                lifetimes.append(5.0)
    df_aug["Carrier_lifetime_ns"] = lifetimes
    
    # 5. Co-catalyst Type (categorical)
    df_aug["Co_catalyst_type"] = base_df["Co_Catalyst"].fillna("None")
    
    # 6. Co-catalyst Loading (wt%)
    df_aug["Co_catalyst_loading_wt_pct"] = base_df["Composition_Raw"].apply(extract_loading)
    
    # 7. Sacrificial Agent Type (categorical)
    df_aug["Sacrificial_agent_type"] = "Glycerol"
    
    # 8. Sacrificial Agent Concentration (vol%)
    df_aug["Sacrificial_agent_vol_pct"] = 10.0
    
    # 9. Light Source Power Density (W m-2)
    df_aug["Light_source_power_density"] = 1000.0
    
    # 10. Reaction pH
    df_aug["Reaction_pH"] = 7.0
    
    # 11. Reactor Type (categorical)
    df_aug["Reactor_type"] = "Pyrex batch"
    
    # Band edges
    df_aug["VB_eV_vs_NHE"] = base_df["VB_eV_vs_NHE"]
    df_aug["CB_eV_vs_NHE"] = base_df["CB_eV_vs_NHE"]
    
    # Targets
    df_aug["HER"] = base_df["HER_clean"]
    df_aug["AQY_420"] = base_df["AQY_clean"]
    df_aug["STH"] = base_df["STH_clean"]
    
    return df_aug

def augment_training_data(df, n_synthetic=50, noise_level=0.05, seed=42):
    """Generates synthetic training points by adding 5% Gaussian noise to
    the numeric features of randomly selected training rows.
    """
    np.random.seed(seed)
    numeric_cols = [
        "Bandgap_eV", "BET_m2_g", "Carrier_lifetime_ns",
        "Co_catalyst_loading_wt_pct", "Sacrificial_agent_vol_pct", 
        "Light_source_power_density", "Reaction_pH", "CB_eV_vs_NHE",
        "HER", "AQY_420", "STH"
    ]
    
    synthetic_rows = []
    for _ in range(n_synthetic):
        parent_idx = np.random.choice(df.index)
        parent_row = df.loc[parent_idx].copy()
        
        # Add Gaussian noise
        for col in numeric_cols:
            val = parent_row[col]
            if pd.notna(val):
                noise = np.random.normal(0, noise_level)
                if col == "CB_eV_vs_NHE":
                    parent_row[col] = val + np.random.normal(0, abs(val) * noise_level)
                else:
                    parent_row[col] = max(0.0, val * (1 + noise))
                    
        # Maintain physical consistency
        parent_row["Absorption_edge_nm"] = 1240.0 / parent_row["Bandgap_eV"]
        parent_row["VB_eV_vs_NHE"] = parent_row["CB_eV_vs_NHE"] + parent_row["Bandgap_eV"]
        synthetic_rows.append(parent_row)
        
    df_synthetic = pd.DataFrame(synthetic_rows)
    return pd.concat([df, df_synthetic], ignore_index=True)

# ==========================================
# STEP 2 & 3: MULTI-OBJECTIVE ARCHITECTURES
# ==========================================

def get_preprocessor():
    numeric_features = [
        "Bandgap_eV", "Absorption_edge_nm", "BET_m2_g", 
        "Carrier_lifetime_ns", "Co_catalyst_loading_wt_pct", 
        "Sacrificial_agent_vol_pct", "Light_source_power_density", "Reaction_pH"
    ]
    categorical_features = ["Co_catalyst_type", "Sacrificial_agent_type", "Reactor_type", "composition"]
    
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

def evaluate_models_loo(df):
    """Trains and cross-validates three models using Leave-One-Out (LOO) CV,
    printing overall R2, MAE, and RMSE for each.
    """
    features = [
        "Bandgap_eV", "Absorption_edge_nm", "BET_m2_g", "Carrier_lifetime_ns",
        "Co_catalyst_type", "Co_catalyst_loading_wt_pct", "Sacrificial_agent_type",
        "Sacrificial_agent_vol_pct", "Light_source_power_density", "Reaction_pH", "Reactor_type", "composition"
    ]
    targets = ["HER", "AQY_420", "STH"]
    
    X = df[features]
    y = df[targets]
    
    loo = LeaveOneOut()
    
    models = {
        "Option A: GradientBoosting (MultiOutput)": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, random_state=42)),
        "Option B: Gaussian Process Regressor": "GPR_SPECIAL",
        "Option C: Multi-Head MLP Neural Network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Cross-Validating {name} via Leave-One-Out CV ---")
        y_true_all = []
        y_pred_all = []
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            preprocessor = get_preprocessor()
            
            if model == "GPR_SPECIAL":
                preds = []
                for target_col in targets:
                    kernel = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e5))
                    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True, random_state=42)
                    pipe = Pipeline([("prep", preprocessor), ("gp", gp)])
                    pipe.fit(X_train, y_train[target_col])
                    preds.append(pipe.predict(X_test))
                y_pred = np.column_stack(preds)
            else:
                pipe = Pipeline([("prep", preprocessor), ("reg", model)])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                
            y_true_all.append(y_test.values[0])
            y_pred_all.append(y_pred[0])
            
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        results[name] = {}
        for i, target_col in enumerate(targets):
            y_t = y_true_all[:, i]
            y_p = y_pred_all[:, i]
            r2 = r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = root_mean_squared_error(y_t, y_p)
            results[name][target_col] = {"R2": r2, "MAE": mae, "RMSE": rmse}
            print(f"  Target [{target_col}]: R² = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}")
            
    return results

# ==========================================
# STEP 4: BANDGAP PENALTY & SOLAR SPECTRUM
# ==========================================

AM15G_WAVELENGTHS = np.array([
    280, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 
    1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 2000, 2500, 4000
])
AM15G_IRRADIANCE = np.array([
    0.0, 0.05, 0.25, 0.8, 1.2, 1.4, 1.4, 1.3, 1.25, 1.15, 1.05, 0.95, 0.85, 0.6, 0.4, 
    0.7, 0.6, 0.3, 0.05, 0.01, 0.15, 0.2, 0.1, 0.05, 0.01, 0.0
])

WL_FINE = np.linspace(280, 4000, 2000)
E_FINE = np.interp(WL_FINE, AM15G_WAVELENGTHS, AM15G_IRRADIANCE)

def calculate_max_sth(bandgap_eV):
    if bandgap_eV < 1.23:
        return 0.0
    absorption_edge_nm = 1240.0 / bandgap_eV
    mask = WL_FINE <= absorption_edge_nm
    wl_int = WL_FINE[mask]
    E_int = E_FINE[mask]
    if len(wl_int) < 2:
        return 0.0
    y = E_int * wl_int
    x = wl_int
    integrated_val = np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x))
    sth_max = integrated_val * 9.908e-5
    return min(sth_max, 100.0)

def apply_bandgap_penalty(bandgap_eV, predicted_sth):
    sth_max = calculate_max_sth(bandgap_eV)
    clamped_sth = min(predicted_sth, sth_max)
    if bandgap_eV < 1.23:
        return 0.0, 0.0
    elif bandgap_eV > 2.2:
        penalty_multiplier = (2.2 / bandgap_eV) ** 2
        return clamped_sth * penalty_multiplier, sth_max
    else:
        return clamped_sth, sth_max

# ==========================================
# STEP 5: GLYCEROL OXIDATION COMPATIBILITY FILTER
# ==========================================

def apply_glycerol_oxidation_filter(df_candidates):
    cb_check = df_candidates["CB_eV_vs_NHE"] < 0.00
    vb_check_1 = df_candidates["VB_eV_vs_NHE"] > 0.40
    vb_check_2 = df_candidates["VB_eV_vs_NHE"] < 1.23
    
    df_candidates["Glycerol_Filter_Pass"] = cb_check & vb_check_1 & vb_check_2
    df_candidates["VB_Overpotential_V"] = (df_candidates["VB_eV_vs_NHE"] - 0.40).clip(lower=0.0)
    return df_candidates

# ==========================================
# STEP 6: CORRECTED VIRTUAL SCREENING RUNNER
# ==========================================

def generate_candidates_multi():
    """Generates the candidate library crossing 12 real host materials
    with realistic co-catalysts and loadings.
    """
    np.random.seed(42)
    candidates = []
    
    hosts_config = {
        "TiO2": {"eg_min": 1.8, "eg_max": 3.2, "cb_min": -0.6, "cb_max": -0.2, "bet_mean": 60.0, "bet_std": 10.0, "life_mean": 20.0, "life_std": 2.0, "comp": "TiO2"},
        "C3N4": {"eg_min": 1.8, "eg_max": 2.7, "cb_min": -1.2, "cb_max": -0.8, "bet_mean": 45.0, "bet_std": 10.0, "life_mean": 3.0, "life_std": 0.5, "comp": "C3N4"},
        "ZrO2": {"eg_min": 1.8, "eg_max": 2.8, "cb_min": -1.2, "cb_max": -0.8, "bet_mean": 950.0, "bet_std": 50.0, "life_mean": 5.0, "life_std": 0.5, "comp": "ZrO2"},
        "CeO2": {"eg_min": 1.8, "eg_max": 2.9, "cb_min": -0.9, "cb_max": -0.5, "bet_mean": 80.0, "bet_std": 10.0, "life_mean": 5.0, "life_std": 0.5, "comp": "CeO2"},
        "ZnS": {"eg_min": 2.0, "eg_max": 3.6, "cb_min": -1.2, "cb_max": -0.8, "bet_mean": 90.0, "bet_std": 10.0, "life_mean": 5.0, "life_std": 0.5, "comp": "ZnS"},
        "ZnIn2S4": {"eg_min": 1.8, "eg_max": 2.4, "cb_min": -1.1, "cb_max": -0.7, "bet_mean": 75.0, "bet_std": 10.0, "life_mean": 5.0, "life_std": 0.5, "comp": "ZnIn2S4"},
        "BiVO4": {"eg_min": 1.8, "eg_max": 2.5, "cb_min": -0.3, "cb_max": 0.1, "bet_mean": 15.0, "bet_std": 3.0, "life_mean": 4.0, "life_std": 0.5, "comp": "BiVO4"},
        "BiFeO3": {"eg_min": 1.8, "eg_max": 2.2, "cb_min": -0.4, "cb_max": 0.0, "bet_mean": 10.0, "bet_std": 2.0, "life_mean": 5.0, "life_std": 0.5, "comp": "BiFeO3"},
        "SrTiO3": {"eg_min": 1.8, "eg_max": 3.2, "cb_min": -1.1, "cb_max": -0.7, "bet_mean": 40.0, "bet_std": 8.0, "life_mean": 5.0, "life_std": 0.5, "comp": "SrTiO3"},
        "CdS": {"eg_min": 1.8, "eg_max": 2.5, "cb_min": -0.9, "cb_max": -0.5, "bet_mean": 50.0, "bet_std": 10.0, "life_mean": 1.5, "life_std": 0.2, "comp": "CdS"},
        "WO3": {"eg_min": 2.0, "eg_max": 2.8, "cb_min": 0.1, "cb_max": 0.5, "bet_mean": 35.0, "bet_std": 5.0, "life_mean": 5.0, "life_std": 0.5, "comp": "WO3"}
    }
    
    co_catalysts = ["Pt", "NiS", "Ni", "Cu", "Au", "CuO", "None"]
    
    # Generate 500 candidates per host
    for host_name in list(hosts_config.keys()) + ["ZnCdS"]:
        for _ in range(500):
            cocat = np.random.choice(co_catalysts)
            w = np.random.uniform(0.1, 3.0) if cocat != "None" else 0.0
            
            if host_name == "ZnCdS":
                x = np.random.uniform(0.1, 0.9)
                eg_base = (1 - x) * 3.6 + x * 2.4 - 0.3 * x * (1 - x)
                eg = max(1.0, eg_base - 0.05 * w)
                cb = (1 - x) * (-1.5) + x * (-0.7) + 0.02 * w
                vb = cb + eg
                bet = max(5.0, np.random.normal(50.0, 10.0))
                lifetime = max(0.1, np.random.normal(2.5, 0.5))
                
                formula = f"{cocat}({w:.2f}wt%)/Zn{1-x:.2f}Cd{x:.2f}S" if cocat != "None" else f"Zn{1-x:.2f}Cd{x:.2f}S"
                candidates.append({
                    "composition": "Zn0.5Cd0.5S", "Formula": formula, "Host": "ZnCdS",
                    "Bandgap_eV": eg, "Absorption_edge_nm": 1240.0 / eg, "BET_m2_g": bet,
                    "Carrier_lifetime_ns": lifetime, "Co_catalyst_type": cocat, "Co_catalyst_loading_wt_pct": w,
                    "Sacrificial_agent_type": "Glycerol", "Sacrificial_agent_vol_pct": 10.0,
                    "Light_source_power_density": 1000.0, "Reaction_pH": 7.0, "Reactor_type": "Pyrex batch",
                    "CB_eV_vs_NHE": cb, "VB_eV_vs_NHE": vb
                })
            else:
                cfg = hosts_config[host_name]
                eg = np.random.uniform(cfg["eg_min"], cfg["eg_max"])
                cb = np.random.uniform(cfg["cb_min"], cfg["cb_max"])
                
                eg = max(1.0, eg - 0.04 * w)
                cb = cb + 0.02 * w
                vb = cb + eg
                
                bet = max(2.0, np.random.normal(cfg["bet_mean"], cfg["bet_std"]))
                lifetime = max(0.1, np.random.normal(cfg["life_mean"], cfg["life_std"]))
                
                formula = f"{cocat}({w:.2f}wt%)/{host_name}" if cocat != "None" else host_name
                
                candidates.append({
                    "composition": cfg["comp"], "Formula": formula, "Host": host_name,
                    "Bandgap_eV": eg, "Absorption_edge_nm": 1240.0 / eg, "BET_m2_g": bet,
                    "Carrier_lifetime_ns": lifetime, "Co_catalyst_type": cocat, "Co_catalyst_loading_wt_pct": w,
                    "Sacrificial_agent_type": "Glycerol", "Sacrificial_agent_vol_pct": 10.0,
                    "Light_source_power_density": 1000.0, "Reaction_pH": 7.0, "Reactor_type": "Pyrex batch",
                    "CB_eV_vs_NHE": cb, "VB_eV_vs_NHE": vb
                })
                
    return pd.DataFrame(candidates)
        
    return pd.DataFrame(candidates)

def run_multi_objective_screening(df_train, df_cand):
    features = [
        "Bandgap_eV", "Absorption_edge_nm", "BET_m2_g", "Carrier_lifetime_ns",
        "Co_catalyst_type", "Co_catalyst_loading_wt_pct", "Sacrificial_agent_type",
        "Sacrificial_agent_vol_pct", "Light_source_power_density", "Reaction_pH", "Reactor_type", "composition"
    ]
    
    preprocessor = get_preprocessor()
    
    models = {}
    targets = ["HER", "AQY_420", "STH"]
    for t in targets:
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e5))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
        pipe = Pipeline([
            ("prep", preprocessor),
            ("gp", gp)
        ])
        pipe.fit(df_train[features], df_train[t])
        models[t] = pipe
        
    preds = {}
    std_devs = {}
    for t in targets:
        prep_X_cand = models[t].named_steps["prep"].transform(df_cand[features])
        mean_scaled, std_scaled = models[t].named_steps["gp"].predict(prep_X_cand, return_std=True)
        preds[t] = np.clip(mean_scaled, 0.0, None)
        std_devs[t] = std_scaled
        
    df_cand["Pred_HER"] = preds["HER"]
    df_cand["Std_HER"] = std_devs["HER"]
    df_cand["Pred_AQY"] = preds["AQY_420"]
    df_cand["Std_AQY"] = std_devs["AQY_420"]
    df_cand["Pred_STH_raw"] = preds["STH"]
    df_cand["Std_STH"] = std_devs["STH"]
    
    # Bandgap Penalty & Solar Clamping
    penalized_sths = []
    max_sths = []
    for idx, row in df_cand.iterrows():
        pen_sth, max_sth = apply_bandgap_penalty(row["Bandgap_eV"], row["Pred_STH_raw"])
        penalized_sths.append(pen_sth)
        max_sths.append(max_sth)
    df_cand["Pred_STH"] = penalized_sths
    df_cand["Theoretical_Max_STH"] = max_sths
    
    # Apply Glycerol filter (Step 5)
    df_cand = apply_glycerol_oxidation_filter(df_cand)
    
    # HARD PRE-FILTER: Remove any candidate where glycerol_filter_pass = False BEFORE scoring
    df_cand = df_cand[df_cand["Glycerol_Filter_Pass"] == True].copy()
    
    if len(df_cand) == 0:
        print("Warning: No candidates passed the glycerol thermodynamic filter!")
        return pd.DataFrame()
        
    # UCB composite scoring
    her_norm = df_cand["Pred_HER"] / (df_cand["Pred_HER"].max() + 1e-5)
    aqy_norm = df_cand["Pred_AQY"] / (df_cand["Pred_AQY"].max() + 1e-5)
    sth_norm = df_cand["Pred_STH"] / (df_cand["Pred_STH"].max() + 1e-5)
    
    total_std = (df_cand["Std_HER"] / (df_cand["Std_HER"].max() + 1e-5) + 
                 df_cand["Std_AQY"] / (df_cand["Std_AQY"].max() + 1e-5) + 
                 df_cand["Std_STH"] / (df_cand["Std_STH"].max() + 1e-5)) / 3.0
                 
    beta = 0.25 # Exploration weight
    op_bonus = df_cand["VB_Overpotential_V"] / (df_cand["VB_Overpotential_V"].max() + 1e-5) * 0.1
    
    df_cand["Composite_Score"] = (her_norm + aqy_norm + sth_norm + beta * total_std + op_bonus)
    
    df_ranked = df_cand.sort_values(by="Composite_Score", ascending=False).reset_index(drop=True)
    df_ranked["Rank"] = df_ranked.index + 1
    top_10 = df_ranked.head(10).copy()
    
    return top_10

if __name__ == "__main__":
    print("==========================================================")
    print("COMMENCING CORRECTED MULTI-OBJECTIVE PHOTOCATALYTIC PIPELINE")
    print("==========================================================\n")
    
    print("Step 1: Loading and augmenting baseline dataset...")
    df_train = load_and_augment_dataset()
    print(f"Loaded {len(df_train)} baseline dataset points.")
    
    # Generate 50 additional synthetic training points using 5% Gaussian noise (Step 4)
    print("Step 1b: Generating 50 synthetic training points using 5% Gaussian noise...")
    df_train = augment_training_data(df_train, n_synthetic=50, noise_level=0.05)
    print(f"Expanded dataset size to N = {len(df_train)} samples.")
    
    print("Step 2: Evaluating multi-objective models via Leave-One-Out CV...")
    eval_results = evaluate_models_loo(df_train)
    
    print("\nStep 6: Executing virtual screening on candidate library...")
    df_candidates = generate_candidates_multi()
    print(f"Initial virtual library size: {len(df_candidates)} candidates.")
    
    # Constrain the virtual screening library to only candidates with bandgap between 1.6 and 2.4 eV BEFORE prediction (Step 3)
    df_candidates = df_candidates[(df_candidates["Bandgap_eV"] >= 1.6) & (df_candidates["Bandgap_eV"] <= 2.4)].copy()
    print(f"Constrained virtual library size (Eg: 1.6 - 2.4 eV): {len(df_candidates)} candidates.")
    
    top_10 = run_multi_objective_screening(df_train, df_candidates)
    
    # Save screening results to CSV
    top_10_csv = "screened_top_10_multi_objective.csv"
    top_10.to_csv(top_10_csv, index=False)
    print(f"\nSuccessfully saved virtual screening results to {top_10_csv}")
    
    # Display the final structured report table
    print("\n=== TOP 10 RECOMMENDED MULTI-OBJECTIVE PHOTOCATALYSTS (FILTERED) ===")
    if len(top_10) > 0:
        display_cols = [
            "Rank", "Formula", "Host", "Co_catalyst_type", "Co_catalyst_loading_wt_pct", 
            "BET_m2_g", "Bandgap_eV", "CB_eV_vs_NHE", "VB_eV_vs_NHE", "Pred_HER", "Std_HER",
            "Pred_AQY", "Std_AQY", "Pred_STH", "Std_STH", "Theoretical_Max_STH", "Glycerol_Filter_Pass", "Composite_Score"
        ]
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(top_10[display_cols].to_string(index=False))
    else:
        print("No candidates succeeded in the thermodynamic screening.")
