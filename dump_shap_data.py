import os
import json
import joblib
import shap
import pandas as pd
import numpy as np
import yaml

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

proc_dir = cfg["paths"]["proc_dir"]
models_dir = cfg["paths"]["models_dir"]
results_dir = cfg["paths"]["results_dir"]

X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
best_model = joblib.load(os.path.join(models_dir, "best_model.joblib"))

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
features = X_test.columns.tolist()

# Sort features by importance
sorted_indices = np.argsort(mean_abs_shap)[::-1]

feature_importance = []
for idx in sorted_indices:
    feature_importance.append({
        "feature": features[idx],
        "mean_abs_shap": float(mean_abs_shap[idx])
    })

# Dump beeswarm details for the top 15 features
beeswarm_data = []
for idx in sorted_indices[:15]:
    feature_name = features[idx]
    # We want a sample of test points (up to 126)
    pts = []
    for i in range(len(X_test)):
        pts.append({
            "shap_val": float(shap_values[i, idx]),
            "feat_val": float(X_test.iloc[i, idx])
        })
    beeswarm_data.append({
        "feature": feature_name,
        "points": pts
    })

out_data = {
    "feature_importance": feature_importance,
    "beeswarm": beeswarm_data
}

output_path = os.path.join(results_dir, "shap_summary_data.json")
with open(output_path, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"Saved SHAP summary data to {output_path}!")
