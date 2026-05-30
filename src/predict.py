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

    # ── Explainability Integration ───────────────────────
    try:
        from src.graphify.explain import PredictExplainer
        explainer = PredictExplainer()
        
        # Get the top candidate features
        top_row = df_ranked.iloc[0]
        feats = {
            "host_material": top_row.get("host_material", "unknown"),
            "co_catalyst": top_row.get("co_catalyst", "none"),
            "co_catalyst_wt_pct": float(top_row.get("co_catalyst_wt_pct", 0.0)) if not pd.isna(top_row.get("co_catalyst_wt_pct")) else 0.0,
            "pH": float(top_row.get("pH", 7.0)) if not pd.isna(top_row.get("pH")) else 7.0,
            "temperature_C": float(top_row.get("temperature_C", 25.0)) if not pd.isna(top_row.get("temperature_C")) else 25.0
        }
        pred_her = float(top_row["predicted_HER_umol_g_h"])
        
        print("\n[Graphify] Generating explanation for the top-performing candidate...")
        exp = explainer.explain_prediction(feats, pred_her)
        explainer.print_ascii_explanation(exp)
        
        # Save explanations for all top 5 candidates to a JSON report
        explanations_report = []
        for i in range(min(5, len(df_ranked))):
            cand_row = df_ranked.iloc[i]
            cand_feats = {
                "host_material": cand_row.get("host_material", "unknown"),
                "co_catalyst": cand_row.get("co_catalyst", "none"),
                "co_catalyst_wt_pct": float(cand_row.get("co_catalyst_wt_pct", 0.0)) if not pd.isna(cand_row.get("co_catalyst_wt_pct")) else 0.0,
                "pH": float(cand_row.get("pH", 7.0)) if not pd.isna(cand_row.get("pH")) else 7.0,
                "temperature_C": float(cand_row.get("temperature_C", 25.0)) if not pd.isna(cand_row.get("temperature_C")) else 25.0
            }
            cand_her = float(cand_row["predicted_HER_umol_g_h"])
            cand_exp = explainer.explain_prediction(cand_feats, cand_her)
            explanations_report.append(cand_exp)
            
        res_dir = "data/results"
        os.makedirs(res_dir, exist_ok=True)
        with open(f"{res_dir}/predictions_explanations.json", "w") as f:
            json.dump(explanations_report, f, indent=2)
        print(f"[Graphify] Saved explanation report to {res_dir}/predictions_explanations.json")
    except Exception as e:
        print(f"[Graphify Warning] Explainability generation skipped: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to candidates CSV")
    parser.add_argument("--output", default=None,  help="Output path (optional)")
    args = parser.parse_args()
    predict(args.input, args.output)

