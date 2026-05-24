import re
import ast
import pandas as pd
import numpy as np

# Path to the HTML file
html_path = "photocatalytic_H2_glycerol_catalyst_table.html"
output_csv_path = "catalyst_dataset_clean.csv"

# Deterministic composition mapping based on host semiconductor matrix
formula_map = {
    "TiO₂ (P25) + 0.5 wt% Pt": "TiO2",
    "TiO₂ + 1 wt% Cu": "TiO2",
    "TiO₂ + 1.8 wt% Cu nanoclusters": "TiO2",
    "Defective TiO₂ + CuOₓ + trace Pt": "TiO2",
    "TiO₂ + Co₃Ni₇(PO₄)₂ bimetal phosphate": "TiO2",
    "g-C₃N₄/TiO₂ heterostructure + NiS": "C3N4",
    "TiO₂ + CdS thin-film Z-scheme": "TiO2",
    "Mesoporous TiO₂ + 2.5 wt% Pt": "TiO2",
    "Anatase TiO₂ nanotubes": "TiO2",
    "TiO₂ + 0.15–0.60 wt% Pt, sequential photodep.": "TiO2",
    "10% CdS on g-C₃N₄ nanospheres": "C3N4",
    "P-doped CdS + g-C₃N₄ with defects": "C3N4",
    "CdS chemically bathed on g-C₃N₄": "C3N4",
    "Zn₀.₄Cd₀.₆S QDs on protonated g-C₃N₄": "C3N4",
    "Amino-functionalised Zr-MOF": "MOF",
    "Zr-MOF + Pt NPs co-catalyst": "MOF",
    "CeO₂ + rGO + Au NPs": "CeO2",
    "ZnS/ZnO core-shell nanorods (Type-II)": "ZnS",
    "ZnS + ZnO nanorods + g-C₃N₄": "C3N4",
    "CuInS₂ + ZnIn₂S₄ Z-scheme heterostructure": "ZnIn2S4",
    "Zn₀.₅Cd₀.₅S solid solution": "Zn0.5Cd0.5S",
    "Bulk graphitic carbon nitride": "C3N4",
    "g-C₃N₄ + 3 wt% Pt": "C3N4",
    "TiO₂ + ultrathin g-C₃N₄ (Type-II)": "TiO2",
    "Mn-doped CdS + TiO₂ (PEC mode)": "TiO2",
    "Mo-doped BiVO₄ (PEC mode)": "BiVO4",
    "ZnCdS + BiFeO₃ + Co catalyst (PEC)": "BiFeO3",
    "SrTiO₃ + ZnIn₂S₄ heterostructure": "SrTiO3",
    "CuO nanoparticles on TiO₂": "TiO2",
    "Plasmonic Au NPs on TiO₂": "TiO2",
    "g-C₃N₄ + NiS co-catalyst": "C3N4",
    "WO₃ + g-C₃N₄ Z-scheme": "C3N4",
    "In₂S₃ + CdS type-II heterostructure": "CdS"
}

def clean_her(val):
    if val == "—" or not val:
        return None
    # Remove standard formatting
    cleaned = val.replace("~", "").replace(",", "").strip()
    # Handle the "(per batch)" and "(PEC)" annotations
    cleaned = re.sub(r'\s*\(.*?\)', '', cleaned)
    cleaned = cleaned.replace("µmol h⁻¹", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        match = re.search(r'([\d.]+)', cleaned)
        if match:
            return float(match.group(1))
        return None

def clean_bandgap(val):
    if val == "—" or not val:
        return None
    cleaned = val.replace("~", "").strip()
    parts = cleaned.split('/')
    floats = []
    for part in parts:
        match = re.search(r'([\d.]+)', part)
        if match:
            floats.append(float(match.group(1)))
    if not floats:
        return None
    # COMPOSITE RULE: Compute mathematical average (mean) of all reported bandgaps
    return sum(floats) / len(floats)

def clean_vb(val):
    if val == "—" or not val:
        return None
    # Replace unicode minus sign with standard minus sign
    cleaned = val.replace("−", "-").replace("~", "").strip()
    parts = cleaned.split('/')
    floats = []
    for part in parts:
        match = re.search(r'([-+]?[\d.]+)', part)
        if match:
            floats.append(float(match.group(1)))
    if not floats:
        return None
    # COMPOSITE RULE: Extract the more negative/less positive VB value (mathematical minimum)
    return min(floats)

def clean_cb(val):
    if val == "—" or not val:
        return None
    # Replace unicode minus sign with standard minus sign
    cleaned = val.replace("−", "-").replace("~", "").strip()
    parts = cleaned.split('/')
    floats = []
    for part in parts:
        match = re.search(r'([-+]?[\d.]+)', part)
        if match:
            floats.append(float(match.group(1)))
    if not floats:
        return None
    # COMPOSITE RULE: Extract the more negative CB value (mathematical minimum)
    return min(floats)

def clean_bet(val):
    if val == "—" or not val:
        return None
    cleaned = val.replace("~", "").strip()
    match = re.search(r'([\d.]+)', cleaned)
    if match:
        return float(match.group(1))
    return None

def clean_generic(val):
    if val == "—" or not val:
        return None
    cleaned = val.replace("~", "").strip()
    match = re.search(r'([-+]?[\d.]+)', cleaned)
    if match:
        return float(match.group(1))
    return None

def main():
    print("Reading HTML file...")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the raw array inside the script
    match = re.search(r'const raw\s*=\s*\[(.*?)\];', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'const raw' array in the HTML file.")
    
    array_content = match.group(1)
    
    rows = []
    for row_match in re.finditer(r'\[\s*(.*?)\s*\],?', array_content):
        row_str = "[" + row_match.group(1) + "]"
        try:
            row_data = ast.literal_eval(row_str)
            rows.append(row_data)
        except Exception as e:
            print(f"Failed to parse row: {row_str}. Error: {e}")

    print(f"Successfully parsed {len(rows)} rows from HTML.")

    columns = [
        "Catalyst", "Composition_Raw", "Bandgap_Raw", "VB_Raw", "CB_Raw", 
        "Heterojunction_Type", "Co_Catalyst", "Light_Source", "HER_Raw", 
        "AQY_Raw", "STH_Raw", "BET_Raw", "Photocurrent_Raw", "Carrier_Lifetime_Raw", 
        "Family", "Reference"
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Clean targets
    df["HER_clean"] = df["HER_Raw"].apply(clean_her)
    
    # Apply special composite/split rules to physical properties
    df["Bandgap_eV"] = df["Bandgap_Raw"].apply(clean_bandgap)
    df["VB_eV_vs_NHE"] = df["VB_Raw"].apply(clean_vb)
    df["CB_eV_vs_NHE"] = df["CB_Raw"].apply(clean_cb)
    df["BET_m2_g"] = df["BET_Raw"].apply(clean_bet)
    
    # AQY and STH cleaning and flagging
    df["AQY_clean"] = df["AQY_Raw"].apply(clean_generic)
    df["STH_clean"] = df["STH_Raw"].apply(clean_generic)
    
    df["AQY_reported"] = df["AQY_clean"].notna().astype(int)
    df["STH_reported"] = df["STH_clean"].notna().astype(int)
    
    # Impute missing AQY and STH values with 0.0 safely
    df["AQY_clean"] = df["AQY_clean"].fillna(0.0)
    df["STH_clean"] = df["STH_clean"].fillna(0.0)
    
    # Map composition based on deterministic host matrix formula
    def map_composition(row):
        comp = row["Composition_Raw"]
        if comp in formula_map:
            return formula_map[comp]
        else:
            print(f"Warning: '{comp}' not found in formula_map. Defaulting to 'TiO2'.")
            return "TiO2"
            
    df["composition"] = df.apply(map_composition, axis=1)

    # Median imputation for baseline physical features to resolve NaNs safely
    df["Bandgap_eV"] = df["Bandgap_eV"].fillna(df["Bandgap_eV"].median())
    df["VB_eV_vs_NHE"] = df["VB_eV_vs_NHE"].fillna(df["VB_eV_vs_NHE"].median())
    df["CB_eV_vs_NHE"] = df["CB_eV_vs_NHE"].fillna(df["CB_eV_vs_NHE"].median())
    df["BET_m2_g"] = df["BET_m2_g"].fillna(df["BET_m2_g"].median())

    # Save cleaned data to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned dataset successfully saved to {output_csv_path}")
    print(df[["Catalyst", "composition", "HER_clean", "Bandgap_eV", "VB_eV_vs_NHE", "CB_eV_vs_NHE", "BET_m2_g"]].head(10).to_string().encode('ascii', errors='replace').decode('ascii'))

if __name__ == "__main__":
    main()
