import patches
import pandas as pd
import numpy as np
import joblib
import os
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Molar masses
M_dict = {
    "Zn": 65.38,
    "Cd": 112.41,
    "S": 32.06,
    "Ti": 47.867,
    "O": 15.999,
    "C": 12.011,
    "N": 14.007,
    "Zr": 91.224,
    "Ce": 140.116,
    "In": 114.818,
    "Bi": 208.98,
    "V": 50.942,
    "Fe": 55.845,
    "Sr": 87.62,
    "Pt": 195.084,
    "Cu": 63.546,
    "Ni": 58.693,
    "Au": 196.967,
    "CuO": 79.545,
    "NiS": 90.753
}

def get_zn_cd_s_formula(x, cocat, w):
    # Host is Zn_{1-x}Cd_xS
    M_host = (1 - x) * M_dict["Zn"] + x * M_dict["Cd"] + M_dict["S"]
    
    # Calculate molar ratio y
    M_cocat = M_dict[cocat]
    y = (w * M_host) / ((100 - w) * M_cocat)
    
    # Format formula
    if cocat == "CuO":
        return f"Cu{y:.6f}O{y:.6f}Zn{1-x:.6f}Cd{x:.6f}S1"
    elif cocat == "NiS":
        return f"Ni{y:.6f}S{y:.6f}Zn{1-x:.6f}Cd{x:.6f}S1"
    else:
        return f"{cocat}{y:.6f}Zn{1-x:.6f}Cd{x:.6f}S1"

def get_other_host_formula(host, cocat, w):
    # Determine host molar mass and composition components
    if host == "TiO2":
        M_host = M_dict["Ti"] + 2 * M_dict["O"]
        base = "Ti1O2"
    elif host == "C3N4":
        M_host = 3 * M_dict["C"] + 4 * M_dict["N"]
        base = "C3N4"
    elif host == "ZrO2":
        M_host = M_dict["Zr"] + 2 * M_dict["O"]
        base = "Zr1O2"
    elif host == "CeO2":
        M_host = M_dict["Ce"] + 2 * M_dict["O"]
        base = "Ce1O2"
    elif host == "ZnIn2S4":
        M_host = M_dict["Zn"] + 2 * M_dict["In"] + 4 * M_dict["S"]
        base = "Zn1In2S4"
    elif host == "BiVO4":
        M_host = M_dict["Bi"] + M_dict["V"] + 4 * M_dict["O"]
        base = "Bi1V1O4"
    elif host == "BiFeO3":
        M_host = M_dict["Bi"] + M_dict["Fe"] + 3 * M_dict["O"]
        base = "Bi1Fe1O3"
    elif host == "SrTiO3":
        M_host = M_dict["Sr"] + M_dict["Ti"] + 3 * M_dict["O"]
        base = "Sr1Ti1O3"
    else:
        raise ValueError(f"Unknown host {host}")
        
    M_cocat = M_dict[cocat]
    y = (w * M_host) / ((100 - w) * M_cocat)
    
    if cocat == "CuO":
        if host == "TiO2":
            return f"Cu{y:.6f}O{2+y:.6f}Ti1"
        elif host == "CeO2":
            return f"Cu{y:.6f}O{2+y:.6f}Ce1"
        elif host == "ZrO2":
            return f"Cu{y:.6f}O{2+y:.6f}Zr1"
        else:
            return f"Cu{y:.6f}O{y:.6f}{base}"
    elif cocat == "NiS":
        if host == "ZnIn2S4":
            return f"Ni{y:.6f}S{4+y:.6f}Zn1In2"
        else:
            return f"Ni{y:.6f}S{y:.6f}{base}"
    else:
        return f"{cocat}{y:.6f}{base}"

def generate_candidates():
    candidates = []
    
    # 1. Generate 3,500 Zn1-xCdxS candidates
    for _ in range(3500):
        x = random.uniform(0.0, 1.0)
        cocat = random.choice(["Pt", "Cu", "Ni", "Au", "NiS", "CuO"])
        w = random.uniform(0.1, 3.0)
        
        formula = get_zn_cd_s_formula(x, cocat, w)
        
        # Physical properties interpolation
        eg_base = (1 - x) * 3.6 + x * 2.4 - 0.3 * x * (1 - x)
        eg = max(1.0, eg_base - 0.05 * w)
        
        cb = (1 - x) * (-1.5) + x * (-0.7) + 0.02 * w
        vb = cb + eg
        
        bet = max(10.0, np.random.normal(50.0, 10.0))
        
        candidates.append({
            "composition": formula,
            "Bandgap_eV": eg,
            "VB_eV_vs_NHE": vb,
            "CB_eV_vs_NHE": cb,
            "BET_m2_g": bet,
            "host": "Zn(1-x)CdxS",
            "x": x,
            "co_catalyst": cocat,
            "loading_wt_pct": w
        })
        
    # 2. Generate 1,500 candidates for other hosts
    hosts = {
        "TiO2": {"eg": 3.2, "vb": 2.9, "cb": -0.3, "bet_mean": 60.0, "bet_std": 10.0},
        "C3N4": {"eg": 2.7, "vb": 1.6, "cb": -1.1, "bet_mean": 45.0, "bet_std": 10.0},
        "ZrO2": {"eg": 2.7, "vb": 1.4, "cb": -1.3, "bet_mean": 650.0, "bet_std": 50.0}, # Zr-MOF
        "CeO2": {"eg": 2.9, "vb": 2.1, "cb": -0.8, "bet_mean": 80.0, "bet_std": 15.0},
        "ZnIn2S4": {"eg": 2.4, "vb": 1.4, "cb": -1.0, "bet_mean": 75.0, "bet_std": 15.0},
        "BiVO4": {"eg": 2.4, "vb": 2.2, "cb": -0.2, "bet_mean": 15.0, "bet_std": 3.0},
        "BiFeO3": {"eg": 2.2, "vb": 1.9, "cb": -0.3, "bet_mean": 10.0, "bet_std": 2.0},
        "SrTiO3": {"eg": 3.2, "vb": 2.2, "cb": -1.0, "bet_mean": 40.0, "bet_std": 8.0}
    }
    
    hosts_list = list(hosts.keys())
    
    for _ in range(1500):
        host = random.choice(hosts_list)
        h_data = hosts[host]
        cocat = random.choice(["Pt", "Cu", "Ni", "Au", "NiS", "CuO"])
        w = random.uniform(0.1, 3.0)
        
        formula = get_other_host_formula(host, cocat, w)
        
        eg = max(1.0, h_data["eg"] - 0.05 * w)
        cb = h_data["cb"] + 0.02 * w
        vb = cb + eg
        
        bet = max(5.0, np.random.normal(h_data["bet_mean"], h_data["bet_std"]))
        
        candidates.append({
            "composition": formula,
            "Bandgap_eV": eg,
            "VB_eV_vs_NHE": vb,
            "CB_eV_vs_NHE": cb,
            "BET_m2_g": bet,
            "host": host,
            "x": np.nan,
            "co_catalyst": cocat,
            "loading_wt_pct": w
        })
        
    return pd.DataFrame(candidates)

def main():
    model_path = "matpipe_model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first by running pipeline_train.py.")
        return
        
    print("Loading trained AutoML MatPipe model...")
    pipe = joblib.load(model_path)
    
    print("Generating 5,000 virtual catalyst candidates...")
    df_candidates = generate_candidates()
    print(f"Generated {len(df_candidates)} candidates.")
    
    # Filter candidates to guarantee thermodynamic edge alignment for GOR and HER
    # CB edge must be negative of NHE (for HER driving force)
    # VB edge must be positive enough for GOR (> +0.4 V vs NHE)
    df_filtered = df_candidates[
        (df_candidates["CB_eV_vs_NHE"] < 0.0) & 
        (df_candidates["VB_eV_vs_NHE"] > 0.4)
    ].copy()
    
    print(f"Filtered to {len(df_filtered)} thermodynamically viable candidates.")
    
    # Prepare features for prediction
    features = df_filtered[["composition", "Bandgap_eV", "VB_eV_vs_NHE", "CB_eV_vs_NHE", "BET_m2_g"]].copy()
    
    print("Running virtual screening predictions with MatPipe (this featurizes candidates and runs ML pipeline)...")
    
    # Predict in chunks to avoid memory issues and display progress
    chunk_size = 500
    predictions_list = []
    
    for i in range(0, len(features), chunk_size):
        chunk = features.iloc[i:i+chunk_size].copy()
        print(f"Processing candidate batch {i//chunk_size + 1}/{(len(features)-1)//chunk_size + 1}...")
        pred_chunk = pipe.predict(chunk)
        predictions_list.append(pred_chunk["HER_clean predicted"])
        
    df_filtered["HER_clean predicted"] = pd.concat(predictions_list, ignore_index=True)
    
    # Sort and select top 10 candidates
    df_top_10 = df_filtered.sort_values(by="HER_clean predicted", ascending=False).head(10)
    
    # Format outputs
    print("\n" + "="*50)
    print("      TOP 10 RECOMMENDED PHOTOCATALYSTS")
    print("="*50)
    
    for rank, (idx, row) in enumerate(df_top_10.iterrows(), 1):
        print(f"\nRank {rank}:")
        print(f"  Host Semiconductor: {row['host']}")
        if row['host'] == "Zn(1-x)CdxS":
            print(f"    Zn/Cd composition: Zn_{1-row['x']:.2f} Cd_{row['x']:.2f} S")
        print(f"  Co-catalyst: {row['loading_wt_pct']:.2f} wt% {row['co_catalyst']}")
        print(f"  Pymatgen Composition representation: {row['composition']}")
        print(f"  Physical Properties:")
        print(f"    Bandgap: {row['Bandgap_eV']:.2f} eV")
        print(f"    VB Edge: {row['VB_eV_vs_NHE']:.2f} V vs NHE")
        print(f"    CB Edge: {row['CB_eV_vs_NHE']:.2f} V vs NHE")
        print(f"    BET Surface Area: {row['BET_m2_g']:.1f} m^2/g")
        print(f"  Predicted HER Rate: {row['HER_clean predicted']:.1f} umol g^-1 h^-1")
        
    # Export results to CSV
    output_path = "screened_top_10_catalysts.csv"
    df_top_10.to_csv(output_path, index=False)
    print(f"\nTop 10 recommended catalysts exported to {output_path}")

if __name__ == "__main__":
    main()
