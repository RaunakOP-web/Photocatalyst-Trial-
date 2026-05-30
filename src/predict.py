"""
predict.py
Given a CSV of new catalyst conditions, predicts HER and ranks candidates.
Usage: python src/predict.py --input my_candidates.csv
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os

PROC_DIR   = "data/processed"
MODELS_DIR = "models"

def predict(input_path, output_path=None):
    model        = joblib.load(f"{MODELS_DIR}/best_model.joblib")
    encoder      = joblib.load(f"{PROC_DIR}/encoder.joblib")
    feature_list = joblib.load(f"{PROC_DIR}/feature_list.joblib")

    df = pd.read_csv(input_path)

    # Fill missing categoricals
    cat_cols = ["host_material", "co_catalyst", "semiconductor_2"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    X = df[feature_list].copy()
    present_cats = [c for c in cat_cols if c in X.columns]
    X[present_cats] = encoder.transform(X[present_cats])

    log_preds = model.predict(X)
    her_preds = np.expm1(log_preds)

    df["predicted_HER_umol_g_h"] = her_preds
    df_ranked = df.sort_values("predicted_HER_umol_g_h", ascending=False)

    if output_path is None:
        output_path = input_path.replace(".csv", "_predictions.csv")

    df_ranked.to_csv(output_path, index=False)
    print(f"\nTop 5 predicted catalysts:")
    print(df_ranked[["host_material","co_catalyst","co_catalyst_wt_pct",
                      "predicted_HER_umol_g_h"]].head())
    print(f"\nFull predictions saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to candidates CSV")
    parser.add_argument("--output", default=None,  help="Output path (optional)")
    args = parser.parse_args()
    predict(args.input, args.output)
