"""
extract_stats.py  —  Print paper statistics from pipeline outputs.
Run from Photocatalyst-Trial- directory: python extract_stats.py
"""
import sys, os, json
import pandas as pd
import numpy as np
import yaml

# Force UTF-8 output so special chars display correctly
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

proc_dir = cfg["paths"]["proc_dir"]
res_dir  = cfg["paths"]["results_dir"]

SEP = "=" * 60

print("\n" + SEP)
print("  PAPER STATISTICS")
print(SEP)

# ── Dataset ──────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(proc_dir, "df_clean.csv"), index_col=0)
target = cfg["data"]["target"]
print(f"\n[Dataset]")
print(f"  N records (clean)  : {len(df):,}")
print(f"  N host materials   : {df['host_material'].nunique()}")
print(f"  HER max            : {df[target].max():,.0f}  umol/g/h")
print(f"  HER median         : {df[target].median():,.0f}  umol/g/h")
tio2_pct = df["host_material"].str.contains("tio2", case=False, na=False).mean() * 100
print(f"  TiO2 fraction      : {tio2_pct:.1f}%")

X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
print(f"  Train / Test split : {len(X_train)} / {len(X_test)}")
print(f"  N features         : {X_train.shape[1]}")

# ── Model metrics ─────────────────────────────────────────────────
print(f"\n[Model Metrics]")
with open(os.path.join(res_dir, "training_results.json")) as f:
    m = json.load(f)

headers = ["Model", "CV_R2", "LOMO_R2", "Test_R2_log", "Test_R2_orig", "MAE_umol"]
rows = []
for name in ["LightGBM", "XGBoost", "Ridge"]:
    d = m[name]
    rows.append([
        name,
        f"{d['CV_R2_mean']:.4f}+/-{d['CV_R2_std']:.4f}",
        f"{d['LOMO_CV_R2_mean']:.4f}+/-{d['LOMO_CV_R2_std']:.4f}",
        f"{d['Test_R2_log']:.4f}",
        f"{d['Test_R2_original']:.4f}",
        f"{d['Test_MAE_umol_g_h']:,.0f}",
    ])
    
col_w = [12, 22, 22, 14, 14, 12]
header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
print(header_line)
print("  " + "-" * (sum(col_w) + 2 * len(col_w)))
for row in rows:
    print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))

# Best model
best_model_path = os.path.join(cfg["paths"]["models_dir"], "best_model_name.txt")
if os.path.exists(best_model_path):
    best = open(best_model_path).read().strip()
    print(f"\n  Best model (composite score): {best}")
    lgb = m[best]
    print(f"  LOMO-CV R2  = {lgb['LOMO_CV_R2_mean']:.4f} +/- {lgb['LOMO_CV_R2_std']:.4f}")
    print(f"  Test R2 log = {lgb['Test_R2_log']:.4f}")
    print(f"  Test MAE    = {lgb['Test_MAE_umol_g_h']:,.0f} umol/g/h")

# ── Uncertainty ───────────────────────────────────────────────────
uq_path = os.path.join(res_dir, "uncertainty_summary.json")
if os.path.exists(uq_path):
    with open(uq_path) as f:
        uq = json.load(f)
    print(f"\n[Uncertainty Quantification]")
    print(f"  Bootstrap (n=200) 90% CI coverage : {uq['bootstrap_empirical_coverage']:.3f}  (target >= 0.90)")
    print(f"  Conformal 90% CI coverage         : {uq['conformal_empirical_coverage']:.3f}  (target >= 0.90)")
    print(f"  Conformal threshold (log units)   : {uq['conformal_threshold_log']:.4f}")
    print(f"  Mean CI width (log units)         : {uq['bootstrap_mean_ci_width_log']:.4f}")
    print(f"  Mean model disagreement (std)     : {uq['mean_model_disagreement_std']:.4f}")

# ── Discovery ─────────────────────────────────────────────────────
disc_path = os.path.join(res_dir, "discovery_candidates.csv")
if os.path.exists(disc_path):
    disc = pd.read_csv(disc_path)
    print(f"\n[Discovery Pipeline]")
    print(f"  Total candidates screened : {len(disc):,}")
    print(f"  Top 10 by UCB score:")
    top10 = disc.nlargest(10, "ucb_log")[
        ["host_material", "co_catalyst", "cocatalyst_wt_pct",
         "light_source_type", "pred_her_umol_g_h", "ucb_her_umol_g_h", "novelty_score"]
    ]
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"    {i:2d}. {row['host_material']}/{row['co_catalyst']}  "
              f"wt={row['cocatalyst_wt_pct']}%  "
              f"light={row['light_source_type']}  "
              f"pred={row['pred_her_umol_g_h']:>10,.0f}  "
              f"ucb={row['ucb_her_umol_g_h']:>10,.0f}  "
              f"novelty={row['novelty_score']:.3f}")

# ── Ablation ──────────────────────────────────────────────────────
abl_path = os.path.join(res_dir, "ablation_results.csv")
if os.path.exists(abl_path):
    abl = pd.read_csv(abl_path)
    print(f"\n[Ablation Study - Delta LOMO-CV R2]")
    for _, row in abl.iterrows():
        bar = "#" * max(0, int(abs(row["delta_r2"]) * 200))
        sign = "+" if row["delta_r2"] >= 0 else "-"
        print(f"  {row.get('ablation', row.get('model',''))[:45]:45s}: {row['delta_r2']:+.4f}  {sign}{bar}")

# ── Figures ───────────────────────────────────────────────────────
fig_dir = os.path.join(res_dir, "figures")
if os.path.exists(fig_dir):
    figs = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
    print(f"\n[Manuscript Figures] ({len(figs)} PNG files in data/results/figures/)")
    for f in sorted(figs):
        size_kb = os.path.getsize(os.path.join(fig_dir, f)) // 1024
        print(f"  {f:<45s} {size_kb:>5d} KB")

print("\n" + SEP)
print("  COPY THESE VALUES INTO docs/paper_draft.md")
print(SEP + "\n")
