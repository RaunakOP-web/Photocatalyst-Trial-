# Validation Safety Report

This report documents the validation audit, identified data leakage risks, and the production-grade validation boundaries established to ensure generalizability and reproducibility.

## Identified Risks & Fixes

### 1. Outlier Removal Leakage
* **Risk**: The previous pipeline ran the IsolationForest outlier removal on the entire dataset *before* train/test splitting. This allowed the target distribution of the test set to influence training set filtering, producing overly optimistic test results.
* **Fix**: Refactored `src/data/preprocess.py` to perform the stratified train/test split first. Outlier detection is now applied **exclusively to the training split**, keeping the test set completely unseen.

### 2. Target Encoding Leakage
* **Risk**: Categorical columns were target-encoded during the preprocessing step on the entire training set. During hyperparameter optimization (HPO), cross-validation (CV) splits were performed on this pre-encoded training data. This leaked information from validation targets into the training folds.
* **Fix**: Refactored preprocessing to keep categorical columns in their raw string format. Target encoding is now executed **inside each cross-validation fold loop** during hyperparameter tuning (fit on the train fold, transform on the val fold). The final encoder is fitted on the entire training set during the final model fit.

### 3. Group Leakage
* **Risk**: The HPO tuning scripts previously utilized standard `KFold` CV, ignoring group-aware boundaries. Since the experiments are grouped by `host_material`, random CV splits split samples of the same host material between training and validation folds, causing group leakage.
* **Fix**: Enforced **GroupKFold** during hyperparameter tuning using the `host_material` groups to guarantee that no host material is shared between training and validation folds, matching the Leave-One-Group-Out (LOGO-CV) evaluation standard.

## Summary of Validation Setup

| Stage | Splitter | Groups | Target Encoding | Outlier Removal |
| --- | --- | --- | --- | --- |
| **HPO Tuning** | `GroupKFold(n_splits=5)` | `host_material` | Inside CV Loop | Training set only |
| **Final Fit** | Train / Test | None | Entire `X_train` | Training set only |
| **Evaluation** | Stratified split | None | Using Saved Encoder | None (Test set unmodified) |
| **Discovery** | Comb. Grid | None | Using Saved Encoder | None |
