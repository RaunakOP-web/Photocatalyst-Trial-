# Migration Notes

This document summarizes the major refactoring, simplification, and validation safety changes implemented during the transition of this materials informatics research codebase.

## 1. Architectural Changes
- **Directory Layout**: Restructured the project into standard ML structures: `configs/` (yaml configurations), `tests/` (unit testing suite), and `src/` modular packages (`src/data/`, `src/features/`, `src/models/`, `src/ensemble/`, `src/evaluation/`, `src/discovery/`, and `src/utils/`).
- **Orchestration**: Refactored entry points (`preprocess.py`, `train.py`, `evaluate.py`, `discover.py`) to the root level. Root scripts are highly readable, config-driven, and run-decoupled. Root `train.py` is under 150 lines.

## 2. Simplification & Ablation Decisions
- **Model Pool**: Standardized on a blending ensemble of XGBoost, LightGBM, CatBoost, and ExtraTrees.
- **Ablations**: Removed overengineered sub-specialist routing, residual correction models, MLP, and TabPFN integration. Ablations showed that simplifying the architecture and relying on SLSQP-optimized weights on the Unified Blending Ensemble significantly improved performance ($R^2$ improved from 0.7077 to 0.8487 on log-scale test set).

## 3. Validation & Leakage Safety Upgrades (Phase 6)
- **Post-Split Outlier Filtering**: Modified outlier removal to execute *after* train/test split (only on the training split) to eliminate data leakage.
- **Fold-Level Target Encoding**: Moved Target Encoding fit/transformations inside the CV loop during hyperparameter tuning. The final target encoder is fit on the full training set during final fitting.
- **Group-Aware Tuning**: Shifted HPO tuning splits from standard `KFold` to `GroupKFold` using the host materials (`host_material` groups) to resolve group leakage.

## 4. Test Suite Integration
- Introduced a testing suite using `pytest` inside the `tests/` folder with **16 unit tests** covering preprocessing, feature engineering, HPO tuning, prediction wrappers, metrics, and virtual screening.
- Core files under test have achieved >85% statement coverage.

## 5. Manual Follow-Up
- Ensure that `pytest` and `pytest-cov` are added to any local container environments.
- To replicate the final publication-grade results, run the root pipeline in order:
  ```bash
  python preprocess.py
  python train.py
  python evaluate.py
  python discover.py
  ```
