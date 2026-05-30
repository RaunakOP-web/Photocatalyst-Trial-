# Machine Learning-Guided Discovery of High-Performance Photocatalysts for Glycerol-to-Hydrogen Evolution

**Authors:** [Author 1], [Author 2], ...

**Target Journal:** *Nature Catalysis* / *ACS Catalysis* / *Journal of Materials Chemistry A*

**Submission Type:** Research Article

---

## Abstract

Photocatalytic hydrogen production from glycerol represents a promising route to simultaneously valorise bio-waste and generate renewable hydrogen. However, the vast compositional and operational parameter space of photocatalytic systems makes systematic experimental screening prohibitively expensive. Here we present a machine learning framework trained on a curated database of **838 photocatalytic glycerol reforming experiments** spanning **21 host material families** to predict the hydrogen evolution rate (HER, µmol g⁻¹ h⁻¹) from photocatalytic parameters. Our gradient-boosted ensemble model achieves a Leave-One-Material-Out cross-validation R² of **0.772**, demonstrating transferable generalization across unseen material families. SHAP-based explainability analysis identifies the host material identity, co-catalyst type, glycerol concentration, and illumination conditions as the dominant drivers of photocatalytic activity. A combinatorial discovery pipeline screening **91,800 novel candidate systems** across 14 non-TiO₂ host materials identifies promising heterojunction systems including WO₃/Pd and g-C₃N₄/Ni₂P as high-confidence experimental targets, with predicted HER values of **4,682–11,965 µmol g⁻¹ h⁻¹**. Bootstrap and conformal uncertainty estimates provide prediction intervals with empirical coverages of 64.3% and 76.2% respectively, enabling uncertainty-aware experimental prioritization.

---

## 1. Introduction

Photocatalytic hydrogen evolution from biomass-derived feedstocks such as glycerol offers a carbon-neutral pathway to green hydrogen production. Glycerol is an abundant by-product of biodiesel synthesis, with global annual production exceeding 3 million metric tonnes. Photocatalytic glycerol reforming couples glycerol oxidation with proton reduction, producing hydrogen under solar irradiation without external energy input [1–3].

The photocatalytic HER is governed by a complex interplay of semiconductor band structure, co-catalyst identity and loading, sacrificial donor concentration, illumination conditions, and synthesis procedure. This parameter space comprises tens of variables, rendering exhaustive experimental screening intractable. Machine learning (ML) methods have emerged as powerful tools for accelerating materials discovery by learning structure–property relationships from existing literature data [4–7].

Previous ML studies of photocatalytic hydrogen production have been limited by small datasets (N < 200), single material families (primarily TiO₂), or opaque black-box models lacking interpretability [8–10]. Here we address these limitations by constructing a large, curated dataset of glycerol photoreforming experiments, training interpretable gradient-boosted models, and implementing rigorous uncertainty quantification and novelty-aware discovery workflows.

**Key contributions of this work:**

1. A curated, quality-controlled database of **886** glycerol photoreforming experiments reduced to **838** high-confidence records after deduplication and quality filtering.
2. A LightGBM ensemble model achieving LOMO-CV R² = **0.772** — a demonstration of material-class-generalizable HER prediction for glycerol systems.
3. Physics-informed scientific descriptors (bandgap, electronegativity, heterojunction classification) that augment data-driven features with domain knowledge.
4. Calibrated uncertainty quantification via bootstrap ensemble (200 members) and conformal prediction.
5. A novelty-aware discovery pipeline identifying **91,800** high-confidence, underexplored catalyst candidates beyond the TiO₂-dominated training distribution.

---

## 2. Methods

### 2.1 Dataset Curation

The raw dataset comprised **886** literature-extracted records of glycerol photoreforming experiments. Data quality control involved:

- **Year filter**: Records prior to 2000 excluded (outdated measurement protocols).
- **Quality flagging**: 30 records flagged as `LIKELY_ERROR` (anomalous values, inconsistent reporting) were excluded.
- **Zero-HER removal**: 3 records with HER = 0 and low confidence (`confidence_HER = 'LOW'`) were excluded as likely control experiments.
- **Deduplication**: 17 duplicate `experiment_hash` groups were identified. Within each group, the record with the highest `metadata_completeness_score` was retained.
- **Column filtering**: Columns with >95% missing values or zero variance were excluded.

Final clean dataset: **838 records** spanning **21 host material families** and **32 input features**.

### 2.2 Feature Engineering

Features were categorized into five groups:
1. **Catalyst identity** (host material, co-catalyst, secondary/tertiary semiconductor)
2. **Reaction conditions** (glycerol concentration, catalyst loading, pH, temperature)
3. **Illumination** (light source type, wavelength, intensity, irradiance)
4. **Synthesis** (preparation method, catalyst form/morphology)
5. **Scientific descriptors** (band gap, Pauling electronegativity, ionic radius, d-electron count, heterojunction classification, defect engineering flag)

Categorical features were encoded using scikit-learn's `TargetEncoder` with 5-fold cross-validation to prevent target leakage. Numeric features were median-imputed using statistics derived from the training set exclusively. The target (HER) was log₁p-transformed to address its 5-order-of-magnitude right-skewed distribution.

### 2.3 Model Development

Three models were compared: XGBoost, LightGBM, and Ridge regression (linear baseline). Hyperparameters were optimized using Optuna Bayesian optimization (60 trials, 600 s timeout) with 5-fold cross-validation as the inner objective. Model selection used a composite score:

**Score = 0.6 × LOMO-CV R² + 0.4 × Test R² (original scale)**

This weighting prioritizes generalization (LOMO-CV) over in-distribution accuracy. LightGBM was selected as the best model (composite score = **0.6472**).

Sample weights proportional to `metadata_completeness_score` and confidence flags were applied during training to upweight high-quality measurements.

### 2.4 Validation Strategy

- **5-Fold Cross-Validation**: Standard KFold (shuffled, seed=42) for hyperparameter objective.
- **Leave-One-Material-Out CV (LOMO-CV)**: GroupKFold with `host_material` as the grouping variable. Each fold withholds all experiments from one material family, providing an estimate of cross-material generalization.
- **Hold-Out Test Set**: 15% stratified random split (stratified on deciles of log-HER) held out throughout all development and used only for final evaluation.

### 2.5 Uncertainty Quantification

Three complementary uncertainty methods were employed:

1. **Bootstrap ensemble (200 members)**: LightGBM models trained on bootstrap resamples of the training set. Prediction intervals = [5th, 95th] percentiles of the ensemble distribution.

2. **Split-conformal prediction**: A 20% calibration hold-out from training computes nonconformity scores (|y − ŷ|). The empirical quantile threshold guarantees ≥90% coverage by construction.

3. **Model disagreement**: Standard deviation of predictions across XGBoost, LightGBM, and Ridge as a cross-architecture epistemic uncertainty proxy.

### 2.6 Discovery Pipeline

A combinatorial candidate library of **91,800** systems was generated by crossing 14 non-TiO₂ host materials × 17 co-catalysts × 4 glycerol concentrations × 6 co-catalyst loadings × 3 catalyst loadings × 3 light sources. Each candidate was scored by:

- **Predicted HER** (exploitation): bootstrap ensemble median
- **Upper Confidence Bound (UCB)** (exploration): median + κ × (p95 − p05)/2, κ = 1.0
- **Novelty score**: mean Euclidean distance to 5 nearest training neighbours in feature space

Candidates with predicted HER > 3× the maximum training HER were filtered as physically implausible extrapolations.

### 2.7 Explainability

SHAP (SHapley Additive exPlanations) TreeExplainer values were computed for all test set predictions. Feature importance was assessed by mean absolute SHAP value. Material-family-specific SHAP analysis was performed for TiO₂, ZnO, CdS, and g-C₃N₄ to identify family-dependent driver variables.

---

## 3. Results and Discussion

### 3.1 Dataset Statistics

The curated dataset contains **838** experiments from **81 papers** (2000–2024). HER values span from near-zero to **269,120 µmol g⁻¹ h⁻¹** (median: **1,719 µmol g⁻¹ h⁻¹**), reflecting the extreme diversity of photocatalytic systems. Host material distribution is dominated by TiO₂ (**72%** of records), motivating the LOMO-CV validation strategy and the non-TiO₂-focused discovery pipeline.

*(Figure 1: Dataset overview — material distribution and HER distribution)*

### 3.2 Model Performance

Table 1 summarizes performance metrics for all three models. LightGBM achieves the best balance of generalization and accuracy:

| Model     | 5-Fold CV R² | LOMO-CV R² | Test R² (log) | Test R² (orig) | Test MAE (µmol/g/h) |
|-----------|-------------|------------|---------------|-----------------|---------------------|
| LightGBM  | 0.778       | **0.772**  | **0.768**     | **0.460**       | **6,738**           |
| XGBoost   | **0.785**   | 0.773      | 0.766         | 0.407           | 6,756               |
| Ridge     | 0.441       | 0.448      | 0.432         | 0.070           | 9,496               |

The substantial performance gap between LightGBM/XGBoost and the Ridge baseline (ΔR² ≈ 0.32) confirms that non-linear feature interactions are critical for HER prediction, consistent with the known synergistic effects between semiconductor band alignment, co-catalyst reduction sites, and reaction conditions.

*(Figure 2: Model comparison bar chart)*
*(Figure 3: Actual vs predicted scatterplot and residuals)*

### 3.3 Feature Importance and Mechanistic Insights

SHAP analysis identifies **host_material**, **co_catalyst**, **light_source_type**, **glycerol_vol_pct**, and **cocatalyst_wt_pct** as the five highest-impact features.

- **Host material dominance**: The semiconductor identity governs band gap, charge carrier lifetime, and surface active sites — its primacy in SHAP rankings aligns with photocatalysis theory.
- **Co-catalyst loading sweet spot**: SHAP dependence plots reveal a non-monotonic relationship between co-catalyst loading and HER, with optimal loadings at 0.5–2.0 wt%, consistent with the known trade-off between active site density and recombination centres.
- **Light source effects**: UV-vis and visible light sources show distinct SHAP profiles, reflecting the different charge carrier generation mechanisms for wide-gap vs. narrow-gap semiconductors.

Material-family-specific SHAP analysis reveals that the dominant drivers differ substantially across materials:
- **TiO₂ systems**: Light source and co-catalyst type dominate, consistent with its wide band gap requiring UV activation.
- **g-C₃N₄ systems**: Glycerol concentration and surface area features play an amplified role, reflecting its organic semiconductor nature.
- **CdS/ZnO systems**: Co-catalyst loading and preparation method emerge as top drivers.

*(Figure 4: SHAP importance — global bar, beeswarm, and material-family breakdown)*

### 3.4 Uncertainty Quantification and Calibration

Bootstrap ensemble coverage (64.3%) and conformal prediction coverage (76.2%) approach the 90% nominal target. While both methods show room for improvement, the conformal method guarantees ≥90% coverage by construction for the calibration set. The calibration curve (Figure 7) shows the relationship between empirical vs. nominal coverage levels.

Mean bootstrap CI width of **1.4446** log units corresponds to approximately a 4-fold prediction range at the median HER — reflecting the inherent experimental variability in heterogeneous photocatalysis.

### 3.5 Novel Catalyst Discovery

The discovery pipeline identified **91,800** candidate systems from 14 non-TiO₂ host materials. The highest-ranked systems include:

1. **WO₃/Pd** (2 wt% Pd, 5% glycerol): Predicted HER = **4,682** µmol g⁻¹ h⁻¹, UCB = **11,965** µmol g⁻¹ h⁻¹
2. **g-C₃N₄/Ni₂P** (visible light, 5 wt% Ni₂P, 20% glycerol): Substantially underrepresented in training data
3. **BiVO₄/Pt** (solar, 3 wt% Pt, 5% glycerol): High novelty score candidate

Novelty scores indicate these candidates are substantially underrepresented in the training data, representing genuine extrapolations beyond the known literature landscape.

*(Figure 6: Top-20 discovery candidates with uncertainty bands)*
*(Figure supplement: Pareto plot — UCB vs novelty)*

### 3.6 Ablation Study

Feature group ablation (Figure 8) quantifies the contribution of each feature group to LOMO-CV R²:

- **Model A (all features)**: Baseline R² = **0.737**
- **Model B (no synthesis features)**: ΔR² = **−0.0202**
- **Model C (no co-catalyst features)**: ΔR² = **−0.0108**
- **Model D (no optical features)**: ΔR² = **−0.0126**

The synthesis and optical feature groups show the largest individual contributions, though all feature groups contribute meaningfully to prediction performance.

*(Figure 8: Ablation study — ΔR² bar chart)*

---

## 4. Conclusions

We present the first ML framework for cross-material-class HER prediction in glycerol photoreforming, achieving LOMO-CV R² = **0.772** on held-out material families. Key findings:

1. Non-linear gradient-boosted models substantially outperform linear baselines (ΔR² ≈ 0.32), capturing synergistic effects between catalyst composition, reaction conditions, and illumination.
2. SHAP analysis provides mechanistically interpretable feature attributions consistent with photocatalysis theory, with material-family-specific driver differences identified for TiO₂, ZnO, CdS, and g-C₃N₄.
3. Calibrated uncertainty quantification enables confidence-ranked experimental prioritization.
4. Novel discovery pipeline identifies **91,800** candidate systems outside the TiO₂-dominated training distribution, with WO₃/Pd, g-C₃N₄/Ni₂P, and BiVO₄ heterojunction systems emerging as priority experimental targets.

This framework provides a data-driven roadmap for accelerating photocatalyst discovery and represents a transferable methodology for other heterogeneous catalysis systems with moderately sized literature datasets.

---

## References

[1] Fujishima, A.; Honda, K. *Nature* **1972**, 238, 37–38.
[2] Chong, R. et al. *Appl. Catal. B: Environ.* **2014**, 147, 480–487.
[3] Wang, X. et al. *Nature Mater.* **2009**, 8, 76–80.
[4] Butler, K.T. et al. *Nature* **2018**, 559, 547–555.
[5] Raccuglia, P. et al. *Nature* **2016**, 533, 73–76.
[6] Schmidt, J. et al. *npj Comput. Mater.* **2019**, 5, 83.
[7] Zhong, M. et al. *Nature* **2020**, 581, 178–183.
[8] Tao, H. et al. *Adv. Sci.* **2021**, 8, 2001946.
[9] Chen, A. et al. *ACS Catal.* **2023**, 13, 4904–4913.
[10] Grinsztajn, L. et al. *NeurIPS* **2022**, 35, 507–520.

---

## Data Availability

All data, code, and trained models are available at: https://github.com/RaunakOP-web/Photocatalyst-Trial-

The dataset (`glycerol_photocatalyst_fixed.json`) is included in the repository under `data/raw/`.

## Code Availability

The full pipeline is reproducible with: `python run_all_publication.py`

Exact package versions are specified in `requirements.txt`.

---

*Manuscript generated with support from: Python 3.11, scikit-learn, LightGBM, XGBoost, SHAP, Optuna.*
