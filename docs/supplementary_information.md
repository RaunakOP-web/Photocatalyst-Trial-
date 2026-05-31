# Supplementary Information

## Machine Learning-Guided Discovery of High-Performance Photocatalysts for Glycerol-to-Hydrogen Conversion via Photoreforming

**Authors:** [Author 1], [Author 2], [Author 3]

---

## Section S1. Dataset Construction and Cleaning

### S1.1. Raw dataset

The raw dataset comprised 886 records extracted from published literature on glycerol photoreforming experiments (2000–2024). Each record contains up to 73 columns describing catalyst composition, reaction conditions, illumination parameters, synthesis methods, and hydrogen evolution rate (HER, µmol g⁻¹ h⁻¹).

### S1.2. Cleaning pipeline

The following sequential cleaning steps were applied:

1. **Quality flag filter**: 30 rows with `data_quality_flag = 'LIKELY_ERROR'` were removed (anomalous values, inconsistent reporting).
2. **Zero-HER removal**: 3 rows with HER = 0 and `confidence_HER = 'LOW'` were removed (control experiments or extraction errors).
3. **Deduplication**: 17 duplicate `experiment_hash` groups were identified. Within each group, the row with the highest `metadata_completeness_score` was retained.
4. **Column pruning**: Columns with >95% missing values or zero variance were excluded.
5. **Leakage prevention**: Columns `AQY_pct`, `AQE_pct`, `STH_pct`, and `HER_reported` were excluded to prevent target leakage.
6. **Provenance exclusion**: Metadata columns (DOI, journal, authors, year, etc.) were excluded from the feature matrix.

**Final dataset: 838 records, 32 features, 21 host material families.**

### S1.3. Dataset statistics

| Statistic | Value |
|---|---|
| Total raw records | 886 |
| Records after cleaning | 838 |
| Host material families | 21 |
| Unique papers | 81 |
| Year range | 2000–2024 |
| HER min | 0.0 µmol g⁻¹ h⁻¹ |
| HER median | 1,719 µmol g⁻¹ h⁻¹ |
| HER max | 269,120 µmol g⁻¹ h⁻¹ |
| TiO₂ fraction | 72% |
| Train set size | 712 (85%) |
| Test set size | 126 (15%) |

---

## Section S2. Hyperparameter Optimisation

### S2.1. Optuna configuration

- **Algorithm**: Tree-Structured Parzen Estimator (TPE)
- **Trials**: 60 per model
- **Timeout**: 600 seconds per model
- **Inner CV**: 5-fold, shuffled, seed=42
- **Objective**: Maximise mean R² across folds

### S2.2. Best hyperparameters

**Table S-HP1. LightGBM best parameters (from `best_params_LightGBM.json`):**

| Parameter | Value |
|---|---|
| n_estimators | See results file |
| max_depth | See results file |
| learning_rate | See results file |
| subsample | See results file |
| colsample_bytree | See results file |
| min_child_samples | See results file |
| reg_alpha | See results file |
| reg_lambda | See results file |

*Note: Exact values are stored in `data/results/best_params_LightGBM.json` and `data/results/best_params_XGBoost.json` in the repository.*

---

## Section S3. Cross-Validation Details

### S3.1. Standard 5-fold CV

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---|---|---|---|---|---|
| LightGBM | — | — | — | — | — | 0.778 ± 0.032 |
| XGBoost | — | — | — | — | — | 0.785 ± 0.020 |
| Ridge | — | — | — | — | — | 0.441 ± 0.050 |

*Note: Per-fold scores are computed during training. Mean ± SD are from `training_results.json`.*

### S3.2. Leave-One-Material-Out CV (LOMO-CV)

LOMO-CV uses GroupKFold (5 splits) with `host_material` as the grouping variable. Each fold withholds all experiments from one or more material families, testing the model's ability to generalise to entirely unseen material classes.

| Model | LOMO-CV R² Mean | LOMO-CV R² SD |
|---|---|---|
| LightGBM | **0.772** | 0.114 |
| XGBoost | 0.773 | 0.113 |
| Ridge | 0.448 | 0.145 |

### S3.3. Model selection criterion

Composite score = 0.6 × LOMO-CV R² + 0.4 × Test R² (original scale).

| Model | LOMO-CV R² | Test R² (orig) | Composite |
|---|---|---|---|
| LightGBM | 0.772 | 0.460 | **0.647** |
| XGBoost | 0.773 | 0.407 | 0.626 |
| Ridge | 0.448 | 0.070 | 0.297 |

---

## Section S4. Ablation Study Results

Feature group ablation quantifies the contribution of each feature category to LOMO-CV R²:

**Table S4. Ablation results (from `ablation_results.csv`).**

| Ablation | LOMO-CV R² | ΔR² vs Baseline |
|---|---|---|
| Baseline (all features) | 0.737 | — |
| No synthesis features | 0.717 | −0.020 |
| No optical features | 0.724 | −0.013 |
| No co-catalyst features | 0.726 | −0.011 |
| No log transform | −0.061 | **−0.798** |

The log transformation is the single most impactful modelling decision. Without it, R² drops by 0.798, confirming that addressing the extreme right-skew of HER is essential.

---

## Section S5. Full Virtual Screening Results

### S5.1. Candidate library

The combinatorial library was constructed from:
- **Host materials (14):** g-C₃N₄, BiVO₄, Bi₂WO₆, Fe₂O₃, WO₃, Nb₂O₅, Ta₂O₅, SrTiO₃, Cu₂O, Ga₂O₃, In₂O₃, ZnS, MoO₃, V₂O₅
- **Co-catalysts (17):** Pt, Ni, Cu, Co, MoS₂, Ni₂P, rGO, Pd, Au, Ru, Rh, Ir, Fe, Mn, WC, CoS₂, FeS₂
- **Cocatalyst loadings (6):** 0.1, 0.5, 1.0, 2.0, 3.0, 5.0 wt%
- **Glycerol concentrations (5):** 5, 10, 20, 30, 50 vol%
- **Light sources (3):** visible, solar, UV-vis
- **Catalyst loadings (4):** 0.5, 1.0, 1.5, 2.0 g/L

**Total candidates: 14 × 17 × 6 × 5 × 3 × (4-5) ≈ 91,800**

### S5.2. Filtering

Candidates with predicted HER > 3 × max training HER (807,360 µmol g⁻¹ h⁻¹) were removed as physically implausible extrapolations.

---

## Section S6. Applicability Domain Methodology

### S6.1. k-NN distance scoring

For each query point x, the AD score is computed as the mean Euclidean distance to the k = 5 nearest training neighbours in the 32-dimensional feature space:

AD(x) = (1/k) Σᵢ₌₁ᵏ ‖x − xᵢ_NN‖₂

### S6.2. Threshold determination

The AD threshold is computed from the training data using leave-one-out k-NN distances:

threshold = µ_train + 2σ_train

where µ_train = 25.28 and σ_train = 52.45, giving threshold = 130.17.

### S6.3. Results

| Dataset | Within AD | Outside AD | % Within |
|---|---|---|---|
| Test set (n=126) | 125 | 1 | 99.2% |
| Discovery candidates (n=91,800) | 91,800 | 0 | 100.0% |
| WO₃/Pd (top candidate) | ✓ | — | AD score = 18.8 |

---

## Section S7. Conformal Prediction Methodology

### S7.1. Split conformal procedure

1. Split training data (n=712) into proper training (n=569, 80%) and calibration (n=143, 20%).
2. Refit LightGBM on the proper training set.
3. Compute nonconformity scores on the calibration set: sᵢ = |yᵢ − ŷᵢ|
4. Set conformal quantile: q̂ = Quantile(s₁,...,sₙ; level=⌈(n+1)(1−α)⌉/n)
5. Prediction interval: C(x) = [ŷ(x) − q̂, ŷ(x) + q̂]

### S7.2. Results

| Parameter | Value |
|---|---|
| α (significance level) | 0.10 |
| Target coverage | 90% |
| Empirical coverage (test set) | **92.9%** |
| q̂ (conformal quantile, log scale) | 2.088 |
| Mean interval width (log scale) | 4.175 |
| Mean interval width (original scale) | 70,874 µmol g⁻¹ h⁻¹ |
| n_calibration | 143 |
| n_test | 126 |

### S7.3. Coverage guarantee

Split conformal prediction guarantees marginal coverage:

P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − α

under the assumption that (Xᵢ, Yᵢ) are exchangeable. This is a finite-sample guarantee, not asymptotic, and holds regardless of the model or data distribution.

---

## Section S8. Supplementary Figures

The following figures are included in the supplementary materials:

- **Fig. S1.** SHAP beeswarm plot (full 32 features)
- **Fig. S2.** SHAP material-family breakdown (TiO₂, g-C₃N₄, CdS, ZnO)
- **Fig. S3.** SHAP local explanations for top 20 test predictions
- **Fig. S4.** Learning curve (training vs. validation R² vs. dataset size)
- **Fig. S5.** Residual distribution histogram
- **Fig. S6.** Per-material R² bar chart (all 21 materials)
- **Fig. S7.** Co-catalyst SHAP dependence plot
- **Fig. S8.** Light source SHAP dependence plot
- **Fig. S9.** Discovery Pareto plot (UCB vs. novelty score)
- **Fig. S10.** Bootstrap vs. conformal coverage comparison
- **Fig. S11.** Uncertainty distribution by material family
- **Fig. S12.** Ablation study bar chart
- **Fig. S13–S15.** SHAP dependence plots for remaining top features

*All figures are available at 300 DPI (PNG) and vector (PDF/SVG) in the repository under `data/results/figures/`.*

---

## Table S1. Full Feature List

| # | Feature Name | Category | Type | Unit | Description |
|---|---|---|---|---|---|
| 1 | host_material | Catalyst identity | Categorical | — | Primary semiconductor (target-encoded) |
| 2 | semiconductor_2 | Catalyst identity | Categorical | — | Secondary semiconductor for heterojunctions |
| 3 | semiconductor_3 | Catalyst identity | Categorical | — | Tertiary component |
| 4 | semiconductor_1_pct | Catalyst identity | Numeric | % | Primary semiconductor weight fraction |
| 5 | semiconductor_2_pct | Catalyst identity | Numeric | % | Secondary semiconductor weight fraction |
| 6 | co_catalyst | Catalyst identity | Categorical | — | Co-catalyst identity (target-encoded) |
| 7 | co_catalyst_wt_pct | Catalyst identity | Numeric | wt% | Co-catalyst loading |
| 8 | form | Synthesis | Categorical | — | Catalyst form/morphology |
| 9 | structure | Synthesis | Categorical | — | Crystal structure |
| 10 | bandgap_eV | Descriptor | Numeric | eV | Band gap energy |
| 11 | preparation_semiconductor | Synthesis | Categorical | — | Semiconductor synthesis method |
| 12 | calcination_temp_semiconductor_C | Synthesis | Numeric | °C | Calcination temperature |
| 13 | calcination_time_semiconductor_h | Synthesis | Numeric | h | Calcination time |
| 14 | preparation_photocatalyst | Synthesis | Categorical | — | Photocatalyst preparation method |
| 15 | calcination_temp_photocatalyst_C | Synthesis | Numeric | °C | Photocatalyst calcination temperature |
| 16 | calcination_time_photocatalyst_h | Synthesis | Numeric | h | Photocatalyst calcination time |
| 17 | light_type | Illumination | Categorical | — | Light type classification |
| 18 | light_source_type | Illumination | Categorical | — | Light source type (UV, visible, solar) |
| 19 | light_power_W | Illumination | Numeric | W | Light power |
| 20 | wavelength_cutoff_nm | Illumination | Numeric | nm | Wavelength cutoff |
| 21 | is_xe_lamp | Illumination | Binary | 0/1 | Xenon lamp flag |
| 22 | is_hg_lamp | Illumination | Binary | 0/1 | Mercury lamp flag |
| 23 | is_led | Illumination | Binary | 0/1 | LED flag |
| 24 | is_uv | Illumination | Binary | 0/1 | UV irradiation flag |
| 25 | is_visible_light | Illumination | Binary | 0/1 | Visible light flag |
| 26 | catalyst_loading_g_L | Conditions | Numeric | g/L | Catalyst loading concentration |
| 27 | catalyst_loading_mg | Conditions | Numeric | mg | Catalyst mass |
| 28 | reaction_volume_mL | Conditions | Numeric | mL | Reaction volume |
| 29 | glycerol_concentration_v_pct | Conditions | Numeric | vol% | Glycerol concentration |
| 30 | glycerol_concentration_std | Conditions | Numeric | — | Standardised glycerol concentration |
| 31 | temperature_C | Conditions | Numeric | °C | Reaction temperature |
| 32 | pH | Conditions | Numeric | — | Solution pH |

## Table S2. Full Model Comparison

| Metric | LightGBM | XGBoost | Ridge |
|---|---|---|---|
| 5-Fold CV R² (mean ± SD) | 0.778 ± 0.032 | 0.785 ± 0.020 | 0.441 ± 0.050 |
| LOMO-CV R² (mean ± SD) | 0.772 ± 0.114 | 0.773 ± 0.113 | 0.448 ± 0.145 |
| Test R² (log scale) | 0.768 | 0.766 | 0.432 |
| Test R² (original scale) | 0.460 | 0.407 | 0.070 |
| Test MAE (log scale) | 0.760 | 0.769 | 1.342 |
| Test MAE (µmol g⁻¹ h⁻¹) | 6,738 | 6,756 | 9,496 |
| Test RMSE (µmol g⁻¹ h⁻¹) | 19,979 | 20,938 | 26,225 |
| Composite selection score | 0.647 | 0.626 | 0.297 |
| Spearman ρ (log) | 0.917 | — | — |

## Table S3. Top 50 Virtual Screening Candidates

*The full table of top-50 candidates ranked by predicted HER, including host material, co-catalyst, loading, glycerol concentration, light source, predicted HER, conformal interval bounds, novelty score, and applicability domain label, is provided in the file `data/results/top_novel_candidates.csv` in the repository.*

| Rank | Host Material | Co-catalyst | Loading (wt%) | Predicted HER (µmol g⁻¹ h⁻¹) | AD Label |
|---|---|---|---|---|---|
| 1 | WO₃ | Pd | 2.0 | 4,682 | within_AD |
| 2–50 | See `top_novel_candidates.csv` | — | — | — | — |
