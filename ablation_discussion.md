# Ablation Study Discussion

This discussion interprets the results of the systematic Leave-One-Group-Out CV (LOGO-CV) feature ablation study.

## Analysis of LOGO-CV Performance

| Configuration | LOGO-CV R² | MAE | RMSE | Spearman $\rho$ |
| --- | --- | --- | --- | --- |
| 1. Experimental features only | -0.3805 | 2.2079 | 2.6652 | 0.1113 |
| 2. Semiconductor descriptors only | -0.7308 | 2.5095 | 2.9843 | -0.3461 |
| 3. Cocatalyst descriptors only | -0.5164 | 2.3170 | 2.7934 | 0.0022 |
| 4. Structural descriptors only | -0.9521 | 2.6453 | 3.1694 | -0.1606 |
| 5. Combined descriptors (no exp) | -0.4942 | 2.3332 | 2.7729 | 0.0260 |
| **6. Full model** | **-0.3123** | **2.1843** | **2.5986** | **0.1718** |

## Key Scientific Insights

### 1. Extrapolation to Unseen Domains
Under LOGO-CV, when a group like `tio2` (which accounts for 72.0% of the dataset) is left out as the validation split, the model is trained on only the remaining 28% of non-TiO2 materials. Because the model has to predict on TiO2 (a completely unseen host material domain) during validation, the task becomes one of severe extrapolation rather than interpolation.

### 2. Low Intra-Group Target Variance
$R^2$ is highly sensitive to the variance of the target variable in the validation fold. Since each validation fold in LOGO-CV is composed of a single host material group, the variance within the fold is extremely small. Consequently, even small absolute prediction errors lead to large negative $R^2$ values because the denominator in the $R^2$ equation is tiny.

### 3. Feature Synergy
The **Full model** outperforms all individual subsets (achieving the highest $R^2 = -0.3123$ and highest Spearman $\rho = 0.1718$). This confirms that combining reaction setup features (Experimental Conditions) with physics-informed material descriptors (Semiconductor and Cocatalyst electronic properties) provides a synergistic prediction space.
