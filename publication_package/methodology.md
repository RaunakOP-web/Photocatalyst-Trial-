# Methodology: Materials Informatics Pipeline

## M1. Preprocessing and Feature Engineering
All raw data is cleaned, deduplicated, and enriched with physical descriptors. Categorical variables are target-encoded using a leakage-free target encoder fitted on training folds. The active features are selected dynamically based on model feature importances and domain expertise.

## M2. Model Training and Validation
To prevent data leakage, model validation is performed using a Leave-One-Group-Out Cross-Validation (LOGO-CV) framework based on host material groups. The models are evaluated on a holdout test set (15% of the total dataset). We train a Blending Ensemble composed of XGBoost, LightGBM, CatBoost, and ExtraTrees.

## M3. Applicability Domain and Conformal Bounds
Prior to screening, an applicability domain (AD) is constructed using four methods: distance-based k-NN, Isolation Forest anomaly detection, Hat-matrix leverage, and Mahalanobis distance. Only candidates scoring $\ge 2$ inside-domain metrics are considered valid. Conformal prediction intervals are calculated at 90% confidence using MAPIE to bound predictions.
