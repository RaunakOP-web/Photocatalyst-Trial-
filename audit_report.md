# Repository Audit Report

This audit report identifies active, obsolete, duplicate, and candidate files for Keep, Delete, or Review as part of Phase 1.

## KEEP

### Core Pipeline Files (to be restructured into `src/` or root orchestration)
* `src/preprocess.py` - Core preprocessing, cleaning, splitting.
* `src/train.py` - Core training and HPO pipeline (to be refactored into a root script and modular `src/` modules).
* `src/evaluate.py` - Core model evaluation and plotting script.
* `src/predict.py` - Core candidate list inference script.
* `src/material_features.py` - Core physics-informed descriptors calculation.
* `src/interaction_features.py` - Core interaction and domain feature additions.
* `src/ensemble_wrapper.py` - Custom ensemble structures (Stacking, Blend, Routing wrappers).
* `src/discovery_pipeline.py` - Main discovery combinatorial screen.
* `src/shap_analysis.py` - Generates detailed SHAP publication plots.
* `src/ablation_study.py` - Generates LOGO-CV ablation results.
* `src/uncertainty_quantification.py` - Estimates bootstrap & conformal UQ bounds.
* `src/manuscript_figures.py` - Creates publication-grade manuscript figures.
* `src/applicability_domain.py` - Estimates domain coverage density.
* `src/matminer_features.py` - Fetches MatMiner properties.
* `src/formula_map.py` - Maps chemical formulas to compositions.
* `src/outlier_removal.py` - Cleans training outliers.
* `src/conformal.py` - Standard split conformal prediction utilities.
* `src/patches.py` - Workarounds for TabPFN constructor and matplotlib threads.
* `config.yaml` - Pipeline configurations.
* `requirements.txt` - Project dependencies.

## DELETE

### Duplicate / Legacy Folders and Code
* `temp_repo/` (Entire folder) - Completely duplicate copy of the repository.
* `archive_v2/` (Entire folder) - Obsolete backup scripts (`train_v2.py`, etc.) from previous runs.
* `automatminer*.log` - Automatic miner run debug output log files.

### Unused Scripts / Dead Code
* `src/tune_weights.py` - Leftover experimental code with no references.
* `src/update_ensemble.py` - Unused helper utility.
* `src/compute_spearman.py` - Single-use experimental script.
* `run_all.py` - Root script that runs Phases sequentially. Will be replaced by standard root scripts.
* `run_all_publication.py` - Root runner that will be simplified and integrated.
* `dump_shap_data.py` - Redundant script whose functionality is covered inside evaluation/SHAP modules.

## REVIEW

### Frontend / Dashboard files
* `dashboard.py` - Local Streamlit/Vantage dashboard script. Can be kept in a separate tools or viz subfolder if required by the user, but marked for review.
* `index.html` / `src/app.js` - Single-page React dashboard frontend. Marked for review.
* `vercel.json` / `.vercelignore` - Configuration files for hosting the dashboard on Vercel.
