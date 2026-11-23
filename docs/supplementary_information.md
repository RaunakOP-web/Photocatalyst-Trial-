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
| HER min | 0.14 µmol g⁻¹ h⁻¹ |
| HER median | 1,719 µmol g⁻¹ h⁻¹ |
| HER mean | 11,300 µmol g⁻¹ h⁻¹ |
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
| LightGBM | — | — | — | — | — | 0.792 |
| XGBoost | — | — | — | — | — | 0.779 |
| Ridge | — | — | — | — | — | 0.409 |

*— : not reported for the run described in the manuscript; regenerate from `data/results/training_results.json` after re-running the pipeline.*

### S3.2. Host-grouped CV

Host-grouped CV uses GroupKFold (5 splits) with `host_material` as the grouping variable. Each fold withholds all experiments from one or more material families. Because several families are held out together, it is less strict than leaving out one family at a time.

| Model | Grouped CV R² Mean | Grouped CV R² SD |
|---|---|---|
| LightGBM | **0.784** | 0.122 |
| XGBoost | 0.771 | 0.107 |
| Ridge | 0.419 | 0.165 |

### S3.3. Model selection criterion

Composite score = 0.6 × grouped CV R² + 0.4 × Test R² (original scale).

| Model | Grouped CV R² | Test R² (orig) | Composite |
|---|---|---|---|
| LightGBM | 0.784 | 0.559 | **0.694** |
| XGBoost | 0.771 | 0.386 | 0.617 |
| Ridge | 0.419 | −0.085 (clipped to 0) | 0.251 |

---

## Section S4. Ablation Study Results

Feature group ablation quantifies the contribution of each feature category to grouped CV R²:

**Table S4. Ablation results (from `ablation_results.csv`).**

*These values come from an earlier pipeline run than the one in the manuscript; regenerate them with `src/ablation_study.py` before submission.*

| Ablation | Grouped CV R² | ΔR² vs Baseline |
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
- **Host materials (15):** g-C₃N₄, BiVO₄, Bi₂WO₆, Fe₂O₃, WO₃, Nb₂O₅, Ta₂O₅, SrTiO₃, Cu₂O, Ga₂O₃, In₂O₃, ZnS, MoO₃, V₂O₅, NiFe₂O₄
- **Co-catalysts (17):** Pt, Ni, Cu, Co, MoS₂, Ni₂P, rGO, Pd, Au, Ru, Rh, Ir, Fe, Mn, WC, CoS₂, FeS₂
- **Cocatalyst loadings (6):** 0.1, 0.5, 1.0, 2.0, 3.0, 5.0 wt%
- **Glycerol concentrations (5):** 5, 10, 20, 30, 50 vol%
- **Light sources (3):** visible, solar, UV-vis
- **Catalyst loadings (4):** 0.5, 1.0, 1.5, 2.0 g/L

**Total candidates: 15 × 17 × 6 × 5 × 3 × 4 = 91,800**

### S5.2. Filtering

Candidates with predicted HER > 3 × max training HER (807,360 µmol g⁻¹ h⁻¹) were removed as physically implausible extrapolations.

---

## Section S6. Applicability Domain Methodology

### S6.1. k-NN distance scoring

For each query point x, the AD score is computed as the mean Euclidean distance to the k = 5 nearest training neighbours in the 40-dimensional feature space:

AD(x) = (1/k) Σᵢ₌₁ᵏ ‖x − xᵢ_NN‖₂

### S6.2. Threshold determination

The AD threshold is computed from the training data using leave-one-out k-NN distances:

threshold = µ_train + 1σ_train = 88.14

(as in manuscript §2.6; µ_train and σ_train are the mean and standard deviation of the training-set leave-one-out k-NN distances).

### S6.3. Results

| Dataset | Within AD | Outside AD | % Within |
|---|---|---|---|
| Test set (n=126) | 123 | 3 | 97.6% |
| Discovery candidates (n=91,800) | 91,800 | 0 | 100.0% |
| SrTiO₃/Ru (top candidate) | ✓ | — | AD score = 7.11 |

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
| Empirical coverage (test set) | **94.4%** |
| q̂ (conformal quantile, log scale) | 2.051 |
| Interval width (log scale) | 4.10 |
| Example interval (original scale) | [392.8, 25,619.9] µmol g⁻¹ h⁻¹ |
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
| 5-Fold CV R² (mean) | 0.792 | 0.779 | 0.409 |
| Grouped CV R² (mean ± SD) | 0.784 ± 0.122 | 0.771 ± 0.107 | 0.419 ± 0.165 |
| Test R² (log scale) | 0.778 | 0.765 | 0.398 |
| Test R² (original scale) | 0.559 | 0.386 | −0.085 |
| Test MAE (log scale) | — | — | — |
| Test MAE (µmol g⁻¹ h⁻¹) | 5,775 | 6,999 | 10,306 |
| Test RMSE (µmol g⁻¹ h⁻¹) | — | — | — |
| Composite selection score | 0.694 | 0.617 | 0.251 |
| Spearman ρ (log) | 0.919 | 0.910 | 0.680 |

*— : not reported for the run described in the manuscript; regenerate from `data/results/training_results.json` after re-running the pipeline.*

## Table S3. Top Virtual Screening Candidates

Top 10 unique candidates ranked by UCB (same as manuscript Table 3). The full top-50 list must be regenerated with `discover.py`: the candidate files committed in the repository come from other pipeline runs.

| Rank | Host Material | Co-catalyst | Co-catalyst wt% | Glycerol vol% | Light Source | Predicted HER (µmol g⁻¹ h⁻¹) | UCB HER (µmol g⁻¹ h⁻¹) | AD score | AD label |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SrTiO₃ | Ru | 2.0 | 5.0 | Solar | 1,466.7 | 11,406.1 | 7.11 | within_AD |
| 2 | SrTiO₃ | Pt | 2.0 | 5.0 | Solar | 1,461.9 | 11,368.8 | 3.52 | within_AD |
| 3 | SrTiO₃ | Rh | 1.0 | 20.0 | Solar | 1,432.8 | 11,142.4 | 5.48 | within_AD |
| 4 | Fe₂O₃ | Rh | 0.1 | 20.0 | Solar | 1,428.5 | 11,108.8 | 81.65 | within_AD |
| 5 | SrTiO₃ | Cu | 3.0 | 5.0 | Visible | 1,415.5 | 11,008.2 | 4.67 | within_AD |
| 6 | SrTiO₃ | Pd | 0.1 | 50.0 | Visible | 1,415.3 | 11,006.6 | 10.96 | within_AD |
| 7 | SrTiO₃ | CoS₂ | 1.0 | 10.0 | Visible | 1,365.8 | 10,621.6 | 4.88 | within_AD |
| 8 | SrTiO₃ | WC | 0.1 | 30.0 | Visible | 1,365.8 | 10,621.6 | 4.88 | within_AD |
| 9 | SrTiO₃ | Mn | 5.0 | 5.0 | UV-vis | 1,365.8 | 10,621.6 | 4.88 | within_AD |
| 10 | SrTiO₃ | FeS₂ | 2.0 | 30.0 | Visible | 1,365.8 | 10,621.6 | 4.88 | within_AD |
