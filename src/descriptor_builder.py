"""
descriptor_builder.py
Phase 4 — Scientific Feature Engineering
Adds physics-informed material descriptors to the preprocessed feature matrix.
Descriptors are derived from host_material and co_catalyst (string columns in
df_clean.csv) and merged back onto X_train / X_test before saving.

Features added:
  - bandgap_ev          : optical band gap (eV) of host semiconductor
  - electron_affinity_ev: electron affinity (eV) of host
  - electronegativity   : Pauling electronegativity of host
  - ionic_radius_ang    : dominant cation ionic radius (Å)
  - crystal_field_d_elec: d-electron count of TM cation (proxy for CFSE)
  - cocatalyst_group    : coarse group of co-catalyst (noble, earth-abundant, none)
  - heterojunction_type : Z-scheme, type-II, homojunction, none
  - defect_engineering  : 1 if preparation mentions doping/defect, else 0
  - surface_area_proxy  : 1 if BET/surface area data exists in the row, else 0

Run standalone (after preprocess.py) or imported as add_descriptors(df_clean, X).
"""

import os
import re
import yaml
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# MATERIAL PROPERTY LOOK-UP TABLES
# Values from standard photocatalysis literature (Fujishima, Chen, Wang reviews).
# All properties are for the most common polymorph / oxidation state.
# ─────────────────────────────────────────────────────────────────────────────

BANDGAP_EV = {
    "tio2":      3.20,  "anatase tio2": 3.20, "rutile tio2": 3.03,
    "zno":       3.37,  "zns":          3.54,  "cds":         2.42,
    "cdse":      1.74,  "g-c3n4":       2.70,  "c3n4":        2.70,
    "bi2wo6":    2.75,  "bi2o3":        2.80,  "bivo4":       2.40,
    "fe2o3":     2.20,  "wо3":          2.60,  "wo3":         2.60,
    "nb2o5":     3.40,  "ta2o5":        4.00,  "srtio3":      3.20,
    "zntio3":    3.10,  "sno2":         3.60,  "in2o3":       2.93,
    "gan":       3.40,  "gan/zno":      2.70,  "zrо2":        5.00,
    "zro2":      5.00,  "moo3":         2.90,  "v2o5":        2.24,
    "znо":       3.37,  "cuо":          1.70,  "cuo":         1.70,
    "cu2o":      2.17,  "nife2o4":      2.00,  "cufeo2":      2.40,
    "zif-8":     5.00,  "mof":          3.50,  "graphene":    0.00,
    "rgo":       0.00,  "cnt":          0.00,  "carbon":      0.00,
    "p25":       3.20,  "p-25":         3.20,
}

ELECTRON_AFFINITY_EV = {
    "tio2": 4.21, "anatase tio2": 4.21, "rutile tio2": 4.42,
    "zno":  4.35, "zns": 3.90,  "cds":  4.08, "g-c3n4": 1.40,
    "c3n4": 1.40, "bivo4": 3.33, "fe2o3": 5.88, "wo3": 5.30,
    "cu2o": 3.74, "cuo": 5.30,  "srtio3": 4.17,
}

ELECTRONEGATIVITY = {
    # Pauling electronegativity of primary cation
    "tio2": 1.54, "anatase tio2": 1.54, "rutile tio2": 1.54,
    "zno":  1.65, "zns":  1.65, "cds":  1.69, "cdse": 1.69,
    "g-c3n4": 3.04, "c3n4": 3.04, "bivo4": 1.90, "fe2o3": 1.83,
    "wo3":  2.36, "cuo":  1.90, "cu2o": 1.90, "srtio3": 1.54,
    "nb2o5": 1.60, "ta2o5": 1.50, "zro2": 1.33, "v2o5": 1.63,
}

IONIC_RADIUS_ANG = {
    # Dominant cation CN=6, Å (Shannon 1976)
    "tio2": 0.605, "anatase tio2": 0.605, "rutile tio2": 0.605,
    "zno":  0.740, "zns":  0.740, "cds":  0.950, "cdse": 0.950,
    "fe2o3": 0.645, "wo3": 0.600, "bivo4": 1.030, "nb2o5": 0.640,
    "ta2o5": 0.640, "zro2": 0.720, "srtio3": 1.180, "cuo": 0.730,
    "cu2o": 0.770, "ga2o3": 0.620, "in2o3": 0.800,
}

D_ELECTRON_COUNT = {
    # Number of d-electrons for the primary TM cation
    "tio2": 0, "anatase tio2": 0, "rutile tio2": 0, "srtio3": 0,
    "v2o5": 0, "nb2o5": 0, "ta2o5": 0, "zro2": 0,
    "moo3": 0, "wo3": 0,
    "fe2o3": 5, "nife2o4": 5, "bivo4": 10, "cuo": 9, "cu2o": 10,
    "cds": 10, "cdse": 10, "zno": 10, "zns": 10,
    "g-c3n4": -1, "c3n4": -1, "carbon": -1, "graphene": -1,
}

COCATALYST_GROUP = {
    # Noble metals
    "pt": "noble", "au": "noble", "pd": "noble", "rh": "noble",
    "ru": "noble", "ir": "noble", "ag": "noble",
    # Earth-abundant metals
    "ni": "earth_abundant", "co": "earth_abundant", "fe": "earth_abundant",
    "cu": "earth_abundant", "mn": "earth_abundant", "mo": "earth_abundant",
    "ni2p": "earth_abundant", "mos2": "earth_abundant", "wc": "earth_abundant",
    "cos2": "earth_abundant", "fes2": "earth_abundant",
    "nip": "earth_abundant", "cop": "earth_abundant",
    # Carbon-based
    "rgo": "carbon", "graphene": "carbon", "cnt": "carbon",
    "graphene oxide": "carbon", "reduced graphene oxide": "carbon",
    # No co-catalyst
    "none": "none", "": "none",
}

HETEROJUNCTION_KEYWORDS = {
    "z-scheme": "z_scheme", "z scheme": "z_scheme", "direct z-scheme": "z_scheme",
    "type-ii": "type_ii", "type ii": "type_ii", "staggered": "type_ii",
    "p-n junction": "p_n", "pn junction": "p_n", "p–n": "p_n",
    "homojunction": "homojunction",
    "isotype": "isotype",
}

DOPING_KEYWORDS = [
    "dop", "nitrogen", "sulfur", "carbon", "fluorine", "phosphor", "defect",
    "vacancy", "interstitial", "substitut", "codop", "co-dop",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def _lookup(table, key, default=np.nan):
    key = _normalise(key)
    # Exact match first
    if key in table:
        return table[key]
    # Partial match (longest matching substring wins)
    best_match = None
    best_len = 0
    for k, v in table.items():
        if k and k in key and len(k) > best_len:
            best_match = v
            best_len = len(k)
    return best_match if best_match is not None else default


def _cocatalyst_group(s):
    key = _normalise(s)
    if key in ("", "none", "nan", "null"):
        return "none"
    for pattern, group in COCATALYST_GROUP.items():
        if pattern and pattern in key:
            return group
    return "unknown"


def _heterojunction_type(structure_col, preparation_col):
    combined = _normalise(structure_col) + " " + _normalise(preparation_col)
    for kw, label in HETEROJUNCTION_KEYWORDS.items():
        if kw in combined:
            return label
    if any(sep in combined for sep in ["/", "-", "composite", "hybrid"]):
        return "composite"
    return "none"


def _defect_flag(preparation_col):
    text = _normalise(preparation_col)
    return int(any(kw in text for kw in DOPING_KEYWORDS))


def _surface_area_proxy(row):
    """1 if any BET-related column has a numeric value."""
    bet_cols = [c for c in row.index if "bet" in c.lower() or "surface_area" in c.lower()]
    for c in bet_cols:
        try:
            if pd.notna(row[c]) and float(row[c]) > 0:
                return 1
        except (ValueError, TypeError):
            pass
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def add_descriptors(df_clean: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """
    Given the cleaned dataframe (df_clean, which still has original string columns)
    and an already encoded feature matrix X (same index), computes new descriptors
    from the string columns in df_clean and appends them to X.

    Returns a NEW DataFrame with the descriptor columns appended.
    """
    desc = pd.DataFrame(index=df_clean.index)

    host = df_clean.get("host_material", pd.Series("", index=df_clean.index))
    cocat = df_clean.get("co_catalyst", pd.Series("", index=df_clean.index))
    structure = df_clean.get("structure", pd.Series("", index=df_clean.index))
    preparation = df_clean.get("preparation_photocatalyst",
                               df_clean.get("preparation_semiconductor",
                                            pd.Series("", index=df_clean.index)))

    desc["bandgap_ev"] = host.apply(lambda x: _lookup(BANDGAP_EV, x))
    desc["electron_affinity_ev"] = host.apply(lambda x: _lookup(ELECTRON_AFFINITY_EV, x))
    desc["electronegativity"] = host.apply(lambda x: _lookup(ELECTRONEGATIVITY, x))
    desc["ionic_radius_ang"] = host.apply(lambda x: _lookup(IONIC_RADIUS_ANG, x))
    desc["crystal_field_d_elec"] = host.apply(lambda x: _lookup(D_ELECTRON_COUNT, x))

    desc["cocatalyst_group"] = cocat.apply(_cocatalyst_group)
    desc["heterojunction_type"] = [
        _heterojunction_type(s, p)
        for s, p in zip(structure, preparation)
    ]
    desc["defect_engineering"] = preparation.apply(_defect_flag)
    desc["surface_area_proxy"] = df_clean.apply(_surface_area_proxy, axis=1)

    # Ordinal-encode cocatalyst_group (noble=3, earth_abundant=2, carbon=1, none/unknown=0)
    cg_map = {"noble": 3, "earth_abundant": 2, "carbon": 1, "none": 0, "unknown": 0}
    desc["cocatalyst_group_ord"] = desc["cocatalyst_group"].map(cg_map).fillna(0).astype(int)
    desc = desc.drop(columns=["cocatalyst_group"])

    # Ordinal-encode heterojunction_type
    hj_map = {"z_scheme": 4, "type_ii": 3, "p_n": 3, "composite": 2,
               "isotype": 1, "homojunction": 1, "none": 0}
    desc["heterojunction_ord"] = desc["heterojunction_type"].map(hj_map).fillna(0).astype(int)
    desc = desc.drop(columns=["heterojunction_type"])

    # Align to X's index (only keep rows present in X)
    desc = desc.loc[desc.index.intersection(X.index)]
    X_aug = X.copy()
    for col in desc.columns:
        X_aug[col] = desc[col]

    return X_aug


def main():
    """Standalone: re-save X_train and X_test with descriptors appended."""
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    proc_dir = cfg["paths"]["proc_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("Loading preprocessed data...")
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
    df_clean = pd.read_csv(os.path.join(proc_dir, "df_clean.csv"), index_col=0)

    # Re-align indices from the same stratified split used in preprocess.py
    from sklearn.model_selection import train_test_split
    data_cfg = cfg["data"]
    strat_bins = pd.qcut(df_clean["log_HER"], 10, labels=False, duplicates="drop")
    _, _, _, y_test_aligned = train_test_split(
        df_clean[[]], df_clean["log_HER"],
        test_size=data_cfg["test_size"],
        stratify=strat_bins,
        random_state=data_cfg["random_state"]
    )
    test_idx = y_test_aligned.index
    train_idx = df_clean.index.difference(test_idx)

    X_train.index = train_idx
    X_test.index = test_idx
    df_clean_train = df_clean.loc[train_idx]
    df_clean_test = df_clean.loc[test_idx]

    print("Adding scientific descriptors to X_train...")
    X_train_aug = add_descriptors(df_clean_train, X_train)

    print("Adding scientific descriptors to X_test...")
    X_test_aug = add_descriptors(df_clean_test, X_test)

    # Save augmented splits
    X_train_aug.to_csv(os.path.join(proc_dir, "X_train_desc.csv"), index=False)
    X_test_aug.to_csv(os.path.join(proc_dir, "X_test_desc.csv"), index=False)
    print(f"Saved X_train_desc.csv ({X_train_aug.shape}) and X_test_desc.csv ({X_test_aug.shape})")

    # Print coverage stats
    new_cols = [c for c in X_train_aug.columns if c not in X_train.columns]
    print(f"\nNew descriptor columns ({len(new_cols)}): {new_cols}")
    for col in new_cols:
        pct = X_train_aug[col].notna().mean() * 100
        print(f"  {col:30s}  coverage = {pct:.1f}%")

    joblib.dump(new_cols, os.path.join(proc_dir, "descriptor_cols.joblib"))
    print("\ndescriptor_builder.py complete.")


if __name__ == "__main__":
    main()
