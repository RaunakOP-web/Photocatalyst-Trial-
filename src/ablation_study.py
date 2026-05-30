"""
ablation_study.py
Phase 8 — Systematic Ablation Studies

Tests five well-defined model configurations:
  Model A: All features          (baseline)
  Model B: No synthesis features (preparation, form, structure, calcination)
  Model C: No co-catalyst features
  Model D: No optical features   (light source, wavelength, irradiance)
  Model E: Only material descriptors (bandgap, electronegativity, etc.)

Reports Delta LOMO-CV R² for each ablation, saved as:
  data/results/ablation_results.csv
  data/results/ablation_plot.png
"""

import os
import yaml
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

# ─────────────────────────────────────────────────────────────────────────────

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths = CFG["paths"]
data_cfg = CFG["data"]
PROC = paths["proc_dir"]
RES = paths["results_dir"]
MOD = paths["models_dir"]


def lomo_cv_r2(X, y, groups, params, n_splits=5):
    """LOMO-CV R² using GroupKFold with LightGBM."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr_idx, val_idx in gkf.split(X, y, groups=groups):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = LGBMRegressor(**params)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        scores.append(r2_score(y_val, preds))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    os.makedirs(RES, exist_ok=True)
    print("Phase 8 - Ablation Studies")

    # ── Load assets ──────────────────────────────────────────────────────────
    X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).squeeze()
    df_clean = pd.read_csv(os.path.join(PROC, "df_clean.csv"), index_col=0)

    from sklearn.model_selection import train_test_split
    strat_bins = pd.qcut(df_clean["log_HER"], 10, labels=False, duplicates="drop")
    _, _, _, y_test_al = train_test_split(
        df_clean[[]], df_clean["log_HER"],
        test_size=data_cfg["test_size"],
        stratify=strat_bins,
        random_state=data_cfg["random_state"]
    )
    train_idx = df_clean.index.difference(y_test_al.index)
    X_train.index = train_idx
    y_train.index = train_idx

    host_groups = df_clean.loc[train_idx, "host_material"].fillna("unknown").values

    # Best LightGBM params
    lgb_params = json.load(open(os.path.join(RES, "best_params_LightGBM.json")))
    lgb_params["verbose"] = -1
    lgb_params["random_state"] = data_cfg["random_state"]

    all_features = X_train.columns.tolist()

    # ── Define feature groups ────────────────────────────────────────────────

    # Synthesis features
    SYNTHESIS_KEYWORDS = [
        "preparation", "calcination", "form", "structure", "synthesis",
    ]

    # Co-catalyst features
    COCAT_KEYWORDS = [
        "co_catalyst", "cocatalyst", "co-catalyst",
    ]

    # Optical / light source features
    OPTICAL_KEYWORDS = [
        "light", "wavelength", "irradiance", "intensity", "optical", "uv",
        "visible", "solar", "xenon", "mercury", "led", "lamp",
    ]

    def _match_keywords(col, keywords):
        col_lower = col.lower()
        return any(kw.lower() in col_lower for kw in keywords)

    syn_cols = [c for c in all_features if _match_keywords(c, SYNTHESIS_KEYWORDS)]
    cocat_cols = [c for c in all_features if _match_keywords(c, COCAT_KEYWORDS)]
    optical_cols = [c for c in all_features if _match_keywords(c, OPTICAL_KEYWORDS)]

    # Material descriptor columns (physics-informed features)
    # These are: bandgap_ev, electron_affinity_ev, electronegativity,
    # ionic_radius_ang, crystal_field_d_elec, cocatalyst_group_ord,
    # heterojunction_ord, defect_engineering, surface_area_proxy
    DESCRIPTOR_KEYWORDS = [
        "bandgap", "affinity", "electronegativity", "ionic_radius",
        "crystal_field", "d_elec", "cocatalyst_group", "heterojunction",
        "defect_engineering", "surface_area_proxy",
    ]
    desc_cols = [c for c in all_features if _match_keywords(c, DESCRIPTOR_KEYWORDS)]

    # Ensure no overlap — feature belongs to first matching group only
    def exclusive_group(group_a, group_b):
        """Return only features in group_a that are NOT in group_b."""
        return [c for c in group_a if c not in group_b]

    syn_cols = exclusive_group(syn_cols, [])
    cocat_cols = exclusive_group(cocat_cols, syn_cols)
    optical_cols = exclusive_group(optical_cols, syn_cols + cocat_cols)

    print(f"\n  Feature groups defined:")
    print(f"    Synthesis features ({len(syn_cols)}):   {syn_cols}")
    print(f"    Co-catalyst features ({len(cocat_cols)}): {cocat_cols}")
    print(f"    Optical features ({len(optical_cols)}):  {optical_cols}")
    print(f"    Descriptor features ({len(desc_cols)}):  {desc_cols}")
    other_cols = [
        c for c in all_features
        if c not in syn_cols + cocat_cols + optical_cols + desc_cols
    ]
    print(f"    Other features ({len(other_cols)}):      {other_cols[:10]}...")

    # ── Define the five models ────────────────────────────────────────────────
    models_def = {
        "Model A: All features (baseline)": all_features,
        "Model B: No synthesis features": [c for c in all_features if c not in syn_cols],
        "Model C: No co-catalyst features": [c for c in all_features if c not in cocat_cols],
        "Model D: No optical features": [c for c in all_features if c not in optical_cols],
        "Model E: Only material descriptors": desc_cols if desc_cols else all_features[:1],
    }

    results = []

    for label, feat_list in models_def.items():
        feat_available = [c for c in feat_list if c in X_train.columns]
        if len(feat_available) < 2:
            print(f"  Skipping '{label}' — only {len(feat_available)} features available")
            continue
        X_sub = X_train[feat_available]
        r2, std = lomo_cv_r2(X_sub, y_train, host_groups, lgb_params)
        delta = r2 - results[0]["lomo_r2"] if results else 0.0
        print(f"  {label}: R²={r2:.4f} ± {std:.4f}  Delta={delta:+.4f}  (n_features={len(feat_available)})")
        results.append({
            "model": label,
            "n_features": len(feat_available),
            "lomo_r2": round(r2, 4),
            "lomo_std": round(std, 4),
            "delta_r2": round(delta, 4),
        })

    # ── Save results ─────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(RES, "ablation_results.csv"), index=False)
    print(f"\n  Saved ablation_results.csv")
    print(df_res.to_string(index=False))

    # ── Bar chart — delta R² ─────────────────────────────────────────────────
    df_plot = df_res.iloc[1:].copy()  # exclude baseline
    if not df_plot.empty:
        colours = []
        for d in df_plot["delta_r2"]:
            if d < -0.05:
                colours.append("#d62728")  # strong red — major drop
            elif d < -0.01:
                colours.append("#ff7f0e")  # orange — moderate drop
            elif d > 0.01:
                colours.append("#2ca02c")  # green — improvement
            else:
                colours.append("#7f7f7f")  # gray — negligible

        fig, ax = plt.subplots(figsize=(10, max(4, len(df_plot) * 0.8)))
        bars = ax.barh(df_plot["model"], df_plot["delta_r2"],
                       color=colours, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.axvline(0, color="k", lw=1.0, linestyle="-")
        ax.set_xlabel("Delta LOMO-CV R² vs. Baseline (Model A)", fontsize=11)
        ax.set_title("Ablation Study: Feature Group Contribution", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        # Annotate bars
        for bar, row in zip(bars, df_plot.itertuples()):
            x = bar.get_width()
            offset = 0.003
            if x >= 0:
                ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                        f"Delta={row.delta_r2:+.4f}", va="center", ha="left", fontsize=9)
            else:
                ax.text(x - offset, bar.get_y() + bar.get_height() / 2,
                        f"Delta={row.delta_r2:+.4f}", va="center", ha="right", fontsize=9)

        # Add n_features annotation
        for bar, row in zip(bars, df_plot.itertuples()):
            ax.text(0.98, bar.get_y() + bar.get_height() / 2,
                    f"n={row.n_features}",
                    transform=ax.get_xaxis_transform(),
                    va="center", ha="right", fontsize=8, color="gray", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(RES, "ablation_plot.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(RES, "ablation_plot.pdf"), bbox_inches="tight")
        plt.close()
        print("  Saved ablation_plot.png / .pdf")

    print("\nPhase 8 complete.")


if __name__ == "__main__":
    main()
