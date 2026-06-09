"""
run_all.py - Execute the full pipeline end-to-end.

Pipeline order:
  1. preprocess.py         - data cleaning, group-aware split, feature engineering
  2. train.py              - 7-model HPO + LOGO-CV + stacking
  3. evaluate.py           - metrics, plots, SHAP, CV gap analysis
  4. ablation_study.py     - feature group importance

Usage:
    python run_all.py                          # full run
    python run_all.py --skip-preprocess        # skip preprocessing
    python run_all.py --skip-train             # skip training
    python run_all.py --only-evaluate          # evaluate + ablation only
"""
import subprocess
import argparse
import time
import shutil
import os


def run(cmd, label=None):
    label = label or cmd
    print(f"\n{'='*65}\n  {label}\n{'='*65}")
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train",      action="store_true")
    parser.add_argument("--only-evaluate",   action="store_true")
    args = parser.parse_args()

    # Backup existing training_results.json before overwriting
    v1_path = "data/results/training_results_v1.json"
    src_path = "data/results/training_results.json"
    if os.path.exists(src_path) and not os.path.exists(v1_path):
        shutil.copy(src_path, v1_path)
        print(f"  Backed up training_results.json -> training_results_v1.json")

    if not args.only_evaluate:
        if not args.skip_preprocess:
            run("python src/preprocess.py",
                "Phase 1 - Preprocessing (group-aware split + feature engineering)")
        if not args.skip_train:
            run("python src/train.py",
                "Phase 2 - Model Training (7 models + stacking, LOGO-CV)")

    run("python src/evaluate.py",
        "Phase 3 - Evaluation (metrics, plots, SHAP, CV gap)")
    run("python src/shap_analysis.py",
        "Phase 4 - SHAP Analysis")

    print(f"\n{'='*65}")
    print("  Pipeline complete. All results are in data/results/")
    print(f"{'='*65}")
