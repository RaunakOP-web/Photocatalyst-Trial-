# Reproducibility Report

This report outlines the measures taken to guarantee end-to-end reproducibility across the model training and candidate screening pipeline.

## 1. Centralized Seed Management
A single, global seed (`random_state: 42`) is read from `config.yaml` and propagated to all downstream steps:
- **Splits**: Stratified train/test splitting (`generate_stratified_split`).
- **Outliers**: IsolationForest outlier detection.
- **Tuning**: Optuna TPE sampler seed and internal GroupKFold splits.
- **Models**: Explicitly passed to XGBoost (`random_state`), LightGBM (`random_state`), CatBoost (`random_seed`), and ExtraTrees (`random_state`).

## 2. Config-Driven Orchestration
All hyperparameters, filepath routes, dataset specifications, and feature lists are loaded dynamically from a standardized configuration structure. No paths or parameter configurations are hardcoded.

## 3. Environment Capture
The execution environment relies on exact specifications listed in `requirements.txt`:
- Python version: 3.14.4
- Core packages:
  - `pandas>=2.0`
  - `numpy>=1.24`
  - `scikit-learn>=1.4`
  - `xgboost>=2.0`
  - `lightgbm>=4.0`
  - `catboost>=1.2`
  - `optuna>=3.4`

## 4. Metadata & Version Logging
The pipeline records training metadata:
- Best parameters are saved to `data/results/best_params_[Model].json`.
- The final ensemble model is serialized to `models/best_model.joblib`.
- Target encoder mapping is saved to `models/target_encoder.joblib`.
- Performance metrics are written to `data/results/training_results.json`.
