import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from src.formula_map import FORMULA_MAP

def add_matminer_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates matminer descriptors for host_material and co_catalyst.
    Fills NaN values with column-wise medians.
    """
    print("\n--- Generating matminer descriptors ---")
    
    # Initialize featurizer with specified properties
    features_list = [
        "Number", "AtomicWeight", "MeltingT", "ElectronAffinity", "Electronegativity",
        "NsValence", "NpValence", "NdValence", "NfValence", "NValence",
        "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled",
        "GSbandgap", "GSmagmom", "SpaceGroupNumber"
    ]
    stats_list = ["mean", "avg_dev", "minimum", "maximum", "range"]
    
    ep = ElementProperty(data_source="magpie", features=features_list, stats=stats_list)
    feature_labels = ep.feature_labels()
    n_features = len(feature_labels)
    
    # Prepare list for storing featurized data
    semi_features = []
    cocat_features = []
    
    semi_full_count = 0
    semi_nan_count = 0
    cocat_full_count = 0
    cocat_nan_count = 0
    
    # Process each row
    for idx, row in df.iterrows():
        # Semiconductor (host_material)
        semi_name = row.get("host_material")
        semi_formula = FORMULA_MAP.get(str(semi_name).strip().lower()) if pd.notna(semi_name) else None
        
        semi_feat_row = [np.nan] * n_features
        if semi_formula:
            try:
                comp = Composition(semi_formula)
                semi_feat_row = ep.featurize(comp)
                semi_full_count += 1
            except Exception as e:
                semi_nan_count += 1
        else:
            semi_nan_count += 1
            
        semi_features.append(semi_feat_row)
        
        # Co-catalyst
        cocat_name = row.get("co_catalyst")
        cocat_formula = FORMULA_MAP.get(str(cocat_name).strip().lower()) if pd.notna(cocat_name) else None
        
        cocat_feat_row = [np.nan] * n_features
        if cocat_formula and str(cocat_formula).lower() != "none":
            try:
                comp = Composition(cocat_formula)
                cocat_feat_row = ep.featurize(comp)
                cocat_full_count += 1
            except Exception as e:
                cocat_nan_count += 1
        else:
            cocat_nan_count += 1
            
        cocat_features.append(cocat_feat_row)
        
    # Create DataFrames
    semi_df = pd.DataFrame(semi_features, columns=[f"mm_semi_{lbl}" for lbl in feature_labels], index=df.index)
    cocat_df = pd.DataFrame(cocat_features, columns=[f"mm_cocat_{lbl}" for lbl in feature_labels], index=df.index)
    
    # Concatenate the new columns
    df_new = pd.concat([df, semi_df, cocat_df], axis=1)
    
    # Fill NaN values in new columns with the column-wise median computed across all non-NaN rows
    mm_cols = [col for col in df_new.columns if col.startswith("mm_")]
    medians = df_new[mm_cols].median()
    # If any median is NaN (all values in column are NaN, e.g. if everything was unknown), fill with 0.0
    medians = medians.fillna(0.0)
    df_new[mm_cols] = df_new[mm_cols].fillna(medians)
    
    print(f"Semiconductor: {semi_full_count} rows featurized, {semi_nan_count} rows fell back to medians.")
    print(f"Co-catalyst: {cocat_full_count} rows featurized, {cocat_nan_count} rows fell back to medians.")
    print(f"Added {len(mm_cols)} new matminer descriptor columns total.")
    print("----------------------------------------\n")
    
    return df_new
