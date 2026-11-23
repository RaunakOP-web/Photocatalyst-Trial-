"""
build_dashboard.py - Write the data shown on the dashboard (index.html) from
the pipeline outputs, and check those outputs for consistency.

Run it after the pipeline (train.py, discover.py, SHAP analysis):

    python build_dashboard.py            # update index.html, print warnings
    python build_dashboard.py --check    # only print warnings, write nothing

Every number on the dashboard comes from the files listed in SOURCES. The
page itself holds no hand-typed values, so it cannot drift from the results.
"""
import argparse
import ast
import json
import os
import re
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCES = {
    "training": "data/results/training_results.json",
    "conformal": "data/results/conformal_summary.json",
    "shap": "data/results/shap_summary_data.json",
    "candidates": "optimized_candidates.csv",
    "dataset": "data/processed/df_clean.csv",
    "grid": "src/discovery/candidate_generation.py",
    "deployed": "models/best_model_name.txt",
    "config": "config.yaml",
}
# Optional files, used only for consistency checks.
CHECK_ONLY = {
    "discovery": "data/results/discovery_candidates.csv",
    "ad": "data/results/ad_summary.json",
    "weights": "models/ensemble_weights.json",
}
HTML = "index.html"
BLOCK = re.compile(r'(<script id="dashboard-data" type="application/json">)(.*?)(</script>)', re.S)

# Feature -> category for the SHAP doughnut. Matched in order; first hit wins.
SHAP_CATEGORIES = [
    ("Light source", ("light_", "is_hg_lamp", "is_xe_lamp", "is_uv", "is_visible_light", "is_led", "wavelength_")),
    ("Co-catalyst", ("cocat_", "co_catalyst")),
    ("Process", ("preparation_", "calcination_")),
    ("Reaction conditions", ("reaction_", "catalyst_loading", "glycerol_", "ph", "temperature_")),
    ("Material", ("semiconductor_", "semi_", "form", "structure", "bandgap", "host_material")),
]

warnings = []


def warn(msg):
    warnings.append(msg)


def path(rel):
    return os.path.join(ROOT, rel)


def category(feature):
    f = feature.lower()
    for name, prefixes in SHAP_CATEGORIES:
        if any(f == p or f.startswith(p) for p in prefixes):
            return name
    warn(f"SHAP feature '{feature}' has no category; add it to SHAP_CATEGORIES.")
    return "Other"


def read_grid():
    """Read the screening grid literal from candidate_generation.py without importing it."""
    tree = ast.parse(open(path(SOURCES["grid"]), encoding="utf-8").read())
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                names[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "grid":
            grid = {}
            for k, v in zip(node.value.keys, node.value.values):
                grid[ast.literal_eval(k)] = names[v.id] if isinstance(v, ast.Name) else ast.literal_eval(v)
            return grid
    raise RuntimeError("No 'grid' dict found in " + SOURCES["grid"])


def build():
    for key, rel in SOURCES.items():
        if not os.path.exists(path(rel)):
            sys.exit(f"Missing pipeline output: {rel}")

    training = json.load(open(path(SOURCES["training"])))
    conformal = json.load(open(path(SOURCES["conformal"])))
    shap = json.load(open(path(SOURCES["shap"])))["feature_importance"]
    cand = pd.read_csv(path(SOURCES["candidates"]))
    df = pd.read_csv(path(SOURCES["dataset"]))
    grid = read_grid()
    deployed = open(path(SOURCES["deployed"])).read().strip()
    config = yaml.safe_load(open(path(SOURCES["config"])))

    n_grid = 1
    for v in grid.values():
        n_grid *= len(v)

    models = []
    for name, m in training.items():
        models.append({
            "name": name,
            "cv_r2": m["CV_R2_mean"], "cv_std": m["CV_R2_std"],
            "lomo": m["LOMO_CV_R2_mean"], "lomo_std": m["LOMO_CV_R2_std"],
            "test_r2_log": m["Test_R2_log"],
            "spearman": m.get("Spearman_rho_log"),
            "mae": m["Test_MAE_umol_g_h"], "rmse": m["Test_RMSE_umol_g_h"],
            "composite": m.get("composite_selection_score"),
        })

    host_counts = df["host_material"].astype(str).str.lower().value_counts().to_dict()
    shortlist = []
    for _, r in cand.iterrows():
        host = str(r["host_material"]).lower()
        shortlist.append({
            "host": host, "cocat": str(r["co_catalyst"]).lower(),
            "wt": float(r["co_catalyst_wt_pct"]), "light": str(r["light_type"]),
            "her": round(float(r["pred_her_umol_g_h"]), 1),
            "lo": round(float(r["pred_her_lower_umol_g_h"]), 1),
            "hi": round(float(r["pred_her_upper_umol_g_h"]), 1),
            "novelty": round(float(r["novelty_score"]), 4),
            "in_training": host_counts.get(host, 0),
        })

    data = {
        "sources": SOURCES,
        "deployed_model": deployed,
        "models": models,
        "conformal": {
            "coverage": conformal["empirical_coverage"],
            "target": conformal["target_coverage"],
            "n_test": conformal["n_test"],
        },
        "dataset_rows": int(len(df)),
        "grid": grid,
        "n_grid": n_grid,
        "shortlist": shortlist,
        "shap": [[s["feature"], round(s["mean_abs_shap"], 4), category(s["feature"])] for s in shap],
    }

    # ---- consistency checks -------------------------------------------------
    by_name = {m["name"]: m for m in models}
    if deployed not in by_name:
        warn(f"Deployed model '{deployed}' ({SOURCES['deployed']}) is not in {SOURCES['training']}.")
    scored = [m for m in models if m["composite"] is not None]
    if scored:
        best = max(scored, key=lambda m: m["composite"])
        if best["name"] != deployed:
            warn(f"Deployed model is {deployed}, but {best['name']} has the higher composite "
                 f"selection score ({best['composite']} vs {by_name.get(deployed, {}).get('composite')}). "
                 f"{SOURCES['deployed']} predates the current selection rule in src/train.py; re-train.")
    if os.path.exists(path(CHECK_ONLY["weights"])):
        w = json.load(open(path(CHECK_ONLY["weights"])))
        missing = sorted(set(w) - set(by_name))
        if missing:
            warn(f"{CHECK_ONLY['weights']} blends {sorted(w)}, but {SOURCES['training']} has "
                 f"{sorted(by_name)} (different training script / run).")

    unseen = sorted({s["host"] for s in shortlist if not s["in_training"]})
    if unseen:
        warn(f"Shortlisted hosts with no rows in the training data: {', '.join(unseen)}. "
             "Their predictions are extrapolations.")
    groups = cand.groupby(["co_catalyst", "co_catalyst_wt_pct", "glycerol_concentration_v_pct",
                           "light_type", "catalyst_loading_g_L"])["pred_her_umol_g_h"]
    degenerate = [k for k, g in groups if len(g) > 1 and g.nunique() == 1]
    if degenerate:
        warn(f"{len(degenerate)} shortlist settings give identical predicted HER for different hosts; "
             "the model does not distinguish those hosts.")
    if (cand["ucb_her_umol_g_h"] - cand["pred_her_upper_umol_g_h"]).abs().max() > 1e-6:
        warn("ucb_her_umol_g_h differs from pred_her_upper_umol_g_h in the shortlist.")
    if "ad_label" in cand and cand["ad_label"].nunique() == 1:
        warn(f"Every shortlisted candidate has ad_label '{cand['ad_label'].iloc[0]}', "
             "including hosts absent from training: the applicability-domain filter removed nothing.")

    leakage = set(config.get("data", {}).get("leakage_cols", []))
    leaked = [s[0] for s in data["shap"] if s[0] in leakage]
    if leaked:
        warn(f"{SOURCES['shap']} contains {len(leaked)} features that config.yaml lists as leakage_cols "
             f"({', '.join(leaked)}): it comes from an older run. Re-run the SHAP analysis.")

    if os.path.exists(path(CHECK_ONLY["discovery"])):
        with open(path(CHECK_ONLY["discovery"])) as f:
            n_disc = sum(1 for _ in f) - 1
        if n_disc != n_grid:
            warn(f"{CHECK_ONLY['discovery']} has {n_disc:,} rows, but the grid in {SOURCES['grid']} "
                 f"has {n_grid:,} combinations (stale file).")
    if os.path.exists(path(CHECK_ONLY["ad"])):
        ad = json.load(open(path(CHECK_ONLY["ad"])))
        if ad.get("discovery_total") not in (None, n_grid):
            warn(f"{CHECK_ONLY['ad']} reports {ad['discovery_total']:,} screened candidates, "
                 f"but the grid has {n_grid:,} (stale file).")
        if ad.get("test_total") not in (None, conformal["n_test"]):
            warn(f"{CHECK_ONLY['ad']} test_total {ad['test_total']} differs from conformal n_test {conformal['n_test']}.")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report warnings")
    args = ap.parse_args()

    data = build()
    for w in warnings:
        print("WARNING:", w)
    if args.check:
        return
    html = open(path(HTML), encoding="utf-8").read()
    if not BLOCK.search(html):
        sys.exit(f"{HTML} has no dashboard-data block.")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = BLOCK.sub(lambda m: m.group(1) + payload + m.group(3), html)
    open(path(HTML), "w", encoding="utf-8").write(html)
    print(f"Updated {HTML} from {len(SOURCES)} pipeline outputs ({len(warnings)} warnings).")


if __name__ == "__main__":
    main()
