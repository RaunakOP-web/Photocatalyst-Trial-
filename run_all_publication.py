"""
run_all_publication.py
Master publication-grade pipeline runner.

Runs the full publication pipeline in order:
  Phase 0: Core ML pipeline (preprocess → train → evaluate → predict)
  Phase 4: Scientific descriptor engineering
  Phase 5: Uncertainty quantification
  Phase 6: Novel material discovery
  Phase 8: Ablation studies
  Phase 11: Manuscript figure generation

Usage:
  python run_all_publication.py               # Full run
  python run_all_publication.py --skip-core   # Skip Phase 0 (already run)
  python run_all_publication.py --only-figs   # Only regenerate figures
  python run_all_publication.py --extract-stats  # Print paper statistics
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml

# ─────────────────────────────────────────────────────────────────────────────

def run_step(label, script_path, cwd="."):
    """Run a Python script and measure wall time."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=cwd,
        capture_output=False
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ERROR: {script_path} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  [OK] {label} completed in {elapsed:.1f}s")


def extract_stats():
    """Print all numeric statistics for manuscript placeholder filling."""
    print("\n" + "="*65)
    print("  PAPER STATISTICS (fill into paper_draft.md)")
    print("="*65)

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    proc_dir = cfg["paths"]["proc_dir"]
    res_dir  = cfg["paths"]["results_dir"]

    import pandas as pd
    import numpy as np

    # Dataset stats
    try:
        df = pd.read_csv(os.path.join(proc_dir, "df_clean.csv"), index_col=0)
        N_clean = len(df)
        N_materials = df["host_material"].nunique()
        HER_max = df[cfg["data"]["target"]].max()
        HER_median = df[cfg["data"]["target"]].median()
        tio2_mask = df["host_material"].str.contains("tio2", na=False, case=False)
        TiO2_pct = tio2_mask.mean() * 100
        print(f"\nDataset:")
        print(f"  N_clean      = {N_clean}")
        print(f"  N_materials  = {N_materials}")
        print(f"  HER_max      = {HER_max:,.0f} µmol/g/h")
        print(f"  HER_median   = {HER_median:,.0f} µmol/g/h")
        print(f"  TiO2_pct     = {TiO2_pct:.0f}%")
    except Exception as e:
        print(f"  Dataset stats error: {e}")

    # Model metrics
    try:
        with open(os.path.join(res_dir, "training_results.json")) as f:
            m = json.load(f)
        lgb  = m.get("LightGBM", {})
        xgb  = m.get("XGBoost", {})
        ridge = m.get("Ridge", {})
        print(f"\nModel Metrics:")
        for name, d in [("LightGBM", lgb), ("XGBoost", xgb), ("Ridge", ridge)]:
            print(f"  {name}:")
            print(f"    CV_R2_mean        = {d.get('CV_R2_mean', 0):.4f}")
            print(f"    LOMO_CV_R2_mean   = {d.get('LOMO_CV_R2_mean', 0):.4f} ± {d.get('LOMO_CV_R2_std', 0):.4f}")
            print(f"    Test_R2_log       = {d.get('Test_R2_log', 0):.4f}")
            print(f"    Test_R2_original  = {d.get('Test_R2_original', 0):.4f}")
            print(f"    Test_MAE_umol_g_h = {d.get('Test_MAE_umol_g_h', 0):,.0f}")
    except Exception as e:
        print(f"  Metric stats error: {e}")

    # Uncertainty stats
    uq_path = os.path.join(res_dir, "uncertainty_summary.json")
    if os.path.exists(uq_path):
        with open(uq_path) as f:
            uq = json.load(f)
        print(f"\nUncertainty:")
        print(f"  Bootstrap coverage = {uq.get('bootstrap_empirical_coverage', 'N/A'):.3f}")
        print(f"  Conformal coverage = {uq.get('conformal_empirical_coverage', 'N/A'):.3f}")
        print(f"  Mean CI width (log)= {uq.get('bootstrap_mean_ci_width_log', 'N/A'):.4f}")
    else:
        print("\n  Uncertainty stats: Run uncertainty_quantification.py first")

    # Discovery stats
    disc_path = os.path.join(res_dir, "discovery_candidates.csv")
    if os.path.exists(disc_path):
        disc = pd.read_csv(disc_path)
        top = disc.head(1)
        print(f"\nDiscovery:")
        print(f"  N_candidates = {len(disc):,}")
        print(f"  Top candidate: {top['host_material'].values[0]}/{top['co_catalyst'].values[0]}")
        print(f"  Top pred HER = {top['pred_her_umol_g_h'].values[0]:,.0f} µmol/g/h")
        print(f"  Top UCB HER  = {top['ucb_her_umol_g_h'].values[0]:,.0f} µmol/g/h")
    else:
        print("\n  Discovery stats: Run discovery_pipeline.py first")

    # Ablation stats
    abl_path = os.path.join(res_dir, "ablation_results.csv")
    if os.path.exists(abl_path):
        abl = pd.read_csv(abl_path)
        print(f"\nAblation (Delta LOMO-CV R²):")
        for _, row in abl.iterrows():
            label = row.get('model', row.get('ablation', 'Unknown'))
            print(f"  {str(label)[:55]:55s} {row['delta_r2']:+.4f}")
    else:
        print("\n  Ablation stats: Run ablation_study.py first")

    print("\n" + "="*65)


def main():
    parser = argparse.ArgumentParser(description="Publication pipeline runner")
    parser.add_argument("--skip-core", action="store_true",
                        help="Skip Phase 0 (preprocess/train/evaluate/predict)")
    parser.add_argument("--only-figs", action="store_true",
                        help="Only regenerate manuscript figures")
    parser.add_argument("--extract-stats", action="store_true",
                        help="Print paper statistics and exit")
    args = parser.parse_args()

    if args.extract_stats:
        extract_stats()
        return

    print("\n" + "="*65)
    print("  PHOTOCATALYST ML — PUBLICATION PIPELINE")
    print("="*65)

    total_start = time.time()

    if not args.only_figs:

        if not args.skip_core:
            # ── Phase 0: Core pipeline ────────────────────────────────────
            run_step("Phase 0a — Preprocessing",      "src/preprocess.py")
            run_step("Phase 0b — Model Training",     "src/train.py")
            run_step("Phase 0c — Evaluation",         "src/evaluate.py")

        # ── Phase 4: Scientific descriptors ──────────────────────────────
        run_step("Phase 4  — Scientific Descriptors", "src/descriptor_builder.py")

        # ── Phase 5: Uncertainty quantification ──────────────────────────
        run_step("Phase 5  — Uncertainty Quantification", "src/uncertainty_quantification.py")

        # ── Phase 6: Discovery pipeline ───────────────────────────────────
        run_step("Phase 6  — Novel Material Discovery", "src/discovery_pipeline.py")

        # ── Phase 8: Ablation studies ─────────────────────────────────────
        run_step("Phase 8  — Ablation Study", "src/ablation_study.py")

        # ── Phase 9: SHAP analysis ───────────────────────────────────────────
        run_step("Phase 9  — SHAP Analysis", "src/shap_analysis.py")

    # ── Phase 11: Manuscript figures ──────────────────────────────────────
    run_step("Phase 11 — Manuscript Figures", "src/manuscript_figures.py")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  ALL PUBLICATION PHASES COMPLETE ({total_elapsed/60:.1f} min total)")
    print(f"  Figures: data/results/figures/")
    print(f"  Results: data/results/")
    print(f"{'='*65}")

    # Print stats summary
    print("\nExtracting paper statistics...\n")
    extract_stats()


if __name__ == "__main__":
    main()
