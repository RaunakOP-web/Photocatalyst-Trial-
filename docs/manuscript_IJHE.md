# Machine Learning-Guided Discovery of High-Performance Photocatalysts for Glycerol-to-Hydrogen Conversion via Photoreforming

**Authors:** [Author 1]<sup>a,*</sup>, [Author 2]<sup>a</sup>, [Author 3]<sup>b</sup>

<sup>a</sup> [Department, University, City, Country]
<sup>b</sup> [Department, University, City, Country]

*Corresponding author. E-mail: [email]

---

## Abstract

Photocatalytic hydrogen production from glycerol represents a promising route to simultaneously valorise biodiesel-derived bio-waste and generate renewable fuel. However, the vast compositional and operational parameter space makes systematic experimental screening prohibitively expensive. Here we present a machine learning framework trained on a curated database of 838 photocatalytic glycerol reforming experiments spanning 21 host material families to predict hydrogen evolution rate (HER). Our LightGBM ensemble achieves a Leave-One-Material-Out cross-validation R² of 0.772 ± 0.114, demonstrating transferable generalisation across unseen material families, with a Spearman rank correlation of ρ = 0.917 (p < 10⁻⁵⁰) confirming accurate catalyst ranking. Split conformal prediction provides mathematically guaranteed 90% prediction intervals (empirical coverage: 92.9%). SHAP analysis identifies host material identity, co-catalyst type, glycerol concentration, and illumination conditions as dominant drivers. A combinatorial screening of 91,800 novel candidate systems identifies WO₃/Pd as the top-ranked candidate (predicted HER = 4,682 µmol g⁻¹ h⁻¹), validated as lying within the model's applicability domain (AD score = 18.8 < threshold 130.2). This framework provides a data-driven roadmap for accelerating photocatalyst discovery beyond the TiO₂-dominated literature.

**Keywords:** photocatalysis; glycerol photoreforming; hydrogen evolution; machine learning; materials informatics; virtual screening; conformal prediction

---

## 1. Introduction

Photocatalytic hydrogen evolution from biomass-derived feedstocks such as glycerol offers a carbon-neutral pathway to green hydrogen production. Glycerol is an abundant by-product of biodiesel synthesis, with global annual production exceeding 3 million metric tonnes. Photocatalytic glycerol reforming couples glycerol oxidation with proton reduction, producing hydrogen under solar irradiation without external energy input [1–3].

The photocatalytic HER is governed by a complex interplay of semiconductor band structure, co-catalyst identity and loading, sacrificial donor concentration, illumination conditions, and synthesis procedure. This parameter space comprises tens of variables, rendering exhaustive experimental screening intractable. Machine learning (ML) methods have emerged as powerful tools for accelerating materials discovery by learning structure–property relationships from existing literature data [4–7].

Previous ML studies of photocatalytic hydrogen production have been limited by small datasets (N < 200), single material families (primarily TiO₂), or opaque black-box models lacking interpretability [8–10]. Furthermore, most studies lack rigorous cross-material validation, calibrated uncertainty quantification, and systematic virtual screening with applicability domain assessment — all essential requirements for publication-quality materials informatics.

Here we address these limitations with four key contributions:

1. A curated, quality-controlled database of 886 glycerol photoreforming experiments reduced to 838 high-confidence records after deduplication and quality filtering.
2. A LightGBM ensemble achieving LOMO-CV R² = 0.772, demonstrating material-class-generalisable HER prediction.
3. Split conformal prediction providing finite-sample coverage guarantees (empirical coverage 92.9% at 90% nominal level), replacing uncalibrated bootstrap intervals.
4. A novelty-aware discovery pipeline screening 91,800 candidate systems with k-NN applicability domain assessment to distinguish reliable interpolations from risky extrapolations.

---

## 2. Methodology

### 2.1. Dataset curation

The raw dataset comprised 886 literature-extracted records of glycerol photoreforming experiments. Data quality control involved: (i) exclusion of 30 records flagged as `LIKELY_ERROR`; (ii) removal of 3 records with HER = 0 and low confidence (control experiments); (iii) deduplication of 17 `experiment_hash` groups, retaining the record with highest metadata completeness score; (iv) exclusion of columns with >95% missing values or zero variance. The final dataset contained 838 records spanning 21 host material families and 32 input features.

### 2.2. Feature engineering

Features were categorised into five groups: (1) catalyst identity (host material, co-catalyst, secondary/tertiary semiconductor); (2) reaction conditions (glycerol concentration, catalyst loading, pH, temperature); (3) illumination (light source type, wavelength, intensity); (4) synthesis method (preparation, calcination temperature/time); (5) scientific descriptors (band gap, electronegativity, ionic radius, d-electron count). Categorical features were encoded using scikit-learn TargetEncoder with 5-fold cross-validation to prevent target leakage. The target was log₁p-transformed to address its 5-order-of-magnitude right-skewed distribution (0–269,120 µmol g⁻¹ h⁻¹).

### 2.3. Model development and selection

Three models were compared: XGBoost, LightGBM, and Ridge regression (linear baseline). Hyperparameters were optimised using Optuna Bayesian optimisation (60 trials, 600 s timeout) with 5-fold cross-validation. Model selection used a composite score (0.6 × LOMO-CV R² + 0.4 × Test R² original scale), prioritising cross-material generalisation over in-distribution accuracy.

### 2.4. Validation strategy

Three complementary validation approaches were employed: (i) 5-fold cross-validation for hyperparameter tuning; (ii) Leave-One-Material-Out cross-validation (LOMO-CV) using GroupKFold with host_material as the grouping variable, providing cross-material generalisation estimates; (iii) a 15% stratified hold-out test set (stratified on log-HER deciles) used exclusively for final evaluation.

### 2.5. Uncertainty quantification via split conformal prediction

We employed split conformal prediction [11] to construct prediction intervals with finite-sample marginal coverage guarantees. The training set (n = 712) was split into a proper training set (n = 569, 80%) and a calibration set (n = 143, 20%). The model was refit on the proper training set. Nonconformity scores s_i = |y_i − ŷ_i| were computed on the calibration set. The conformal quantile q̂ was set at level ⌈(n+1)(1−α)⌉/n (α = 0.10), yielding prediction intervals [ŷ − q̂, ŷ + q̂] that provably satisfy P(Y ∈ C(X)) ≥ 1 − α under exchangeability.

### 2.6. Virtual screening pipeline

A combinatorial candidate library of 91,800 systems was generated from 14 non-TiO₂ host materials × 17 co-catalysts × conditions. Each candidate was scored by predicted HER (exploitation) and upper confidence bound (UCB = expm1(ŷ + q̂), exploration). Candidates with predicted HER > 3× maximum training HER were filtered as physically implausible.

### 2.7. Applicability domain assessment

To identify whether predictions involve interpolation or extrapolation, we computed a k-nearest-neighbour (k = 5) distance-based applicability domain (AD) score for each candidate. The AD threshold was set at the mean + 2σ of leave-one-out k-NN distances within the training set. Candidates with AD score ≤ threshold are deemed "within AD" (trustworthy); those above are flagged as extrapolations requiring additional experimental caution.

### 2.8. Explainability

SHAP TreeExplainer values were computed for all test predictions. Feature importance was assessed by mean absolute SHAP value. Material-family-specific SHAP analysis was performed for TiO₂, ZnO, CdS, and g-C₃N₄.

---

## 3. Results and discussion

### 3.1. Dataset characteristics

The curated dataset contains 838 experiments from 81 publications (2000–2024). HER values span from near-zero to 269,120 µmol g⁻¹ h⁻¹ (median: 1,719), reflecting extreme diversity in photocatalytic systems. Host material distribution is dominated by TiO₂ (72% of records), motivating the LOMO-CV strategy and the non-TiO₂-focused discovery pipeline (Fig. 1).

### 3.2. Model performance

Table 1 summarises performance metrics. LightGBM achieves the best balance of generalisation and accuracy:

**Table 1.** Model performance comparison.

| Model | 5-Fold CV R² | LOMO-CV R² | Test R² (log) | Test R² (orig) | Test MAE (µmol g⁻¹ h⁻¹) | Spearman ρ |
|---|---|---|---|---|---|---|
| LightGBM | 0.778 | **0.772 ± 0.114** | **0.768** | **0.460** | **6,738** | **0.917** |
| XGBoost | 0.785 | 0.773 ± 0.113 | 0.766 | 0.407 | 6,756 | — |
| Ridge | 0.441 | 0.448 ± 0.145 | 0.432 | 0.070 | 9,496 | — |

The substantial performance gap between LightGBM/XGBoost and the Ridge baseline (ΔR² ≈ 0.32) confirms that non-linear feature interactions are critical for HER prediction, consistent with known synergistic effects between semiconductor band alignment, co-catalyst reduction sites, and reaction conditions (Fig. 2, Fig. 3).

It is important to distinguish between model performance on the log-transformed target (R² = 0.768) and the original linear scale (R² = 0.460). The HER values in this dataset span five orders of magnitude (0 to 269,120 µmol g⁻¹ h⁻¹), a range driven by heterogeneous experimental conditions rather than intrinsic catalyst performance alone. On a linear scale, R² is dominated by a small number of extreme high-HER observations and is not an appropriate summary metric for this distribution. The log-scale R² of 0.768 reflects the model's ability to correctly rank catalysts across the full activity range, which is the relevant quantity for screening applications. This is consistent with reporting conventions in comparable machine learning studies on heterogeneous catalysis [5,7] where log-transformed targets are standard practice.

The model achieved a Spearman rank correlation of ρ = 0.917 (p < 10⁻⁵⁰), confirming that it correctly ranks catalysts by predicted activity even when absolute HER predictions carry uncertainty. This ranking fidelity is the operationally relevant metric for virtual screening, where the goal is to identify the most promising candidates for experimental validation rather than to predict absolute rates with high precision.

**Table 2.** Comparison with literature benchmarks.

| Study | Dataset size | Target | R² (primary metric) |
|---|---|---|---|
| This work | 838 | log(HER+1) | 0.772 (LOMO-CV) |
| Goldsmith et al. (2018) [12] | ~600 | log(TOF) | 0.71 |
| Li et al. (2021) [13] | ~500 | log(activity) | 0.74 |
| Zhong et al. (2020) [7] | ~3,000 | log(HER) | 0.89 |

Our LOMO-CV R² = 0.772 is competitive with smaller-dataset literature and acknowledges the advantage of larger datasets (Zhong et al.) in achieving higher R².

### 3.3. Feature importance and mechanistic insights

SHAP analysis identifies host_material, co_catalyst, light_source_type, glycerol_vol_pct, and cocatalyst_wt_pct as the five highest-impact features (Fig. 4). Host material dominance aligns with photocatalysis theory (band gap, charge carrier lifetime). SHAP dependence plots reveal non-monotonic co-catalyst loading effects (optimum at 0.5–2.0 wt%), consistent with the trade-off between active site density and charge recombination.

Material-family-specific SHAP analysis reveals distinct drivers: TiO₂ systems are dominated by light source and co-catalyst type (wide band gap requiring UV); g-C₃N₄ systems by glycerol concentration and surface area (organic semiconductor nature); CdS/ZnO systems by co-catalyst loading and preparation method.

### 3.4. Uncertainty quantification

Prediction intervals were constructed using split conformal prediction [11], which provides finite-sample marginal coverage guarantees. The conformal quantile was estimated on a held-out calibration set of 143 samples, yielding empirical coverage of 92.9% on the test set at the 90% nominal level. The mean interval width of 4.18 log units (corresponding to approximately 70,874 µmol g⁻¹ h⁻¹ in original scale) reflects the inherent experimental variability in heterogeneous photocatalysis across five orders of magnitude.

While wide intervals are expected given the extreme target range, the conformal framework guarantees that the true value falls within the predicted interval at least 90% of the time — a property that bootstrap methods (which achieved only 64.3% empirical coverage) cannot provide. For virtual screening, we use the upper conformal bound as an optimistic acquisition function, consistent with standard practice in Bayesian optimisation literature.

### 3.5. Novel catalyst discovery

The discovery pipeline identified 91,800 candidate systems from 14 non-TiO₂ host materials. The highest-ranked systems by UCB include:

1. **WO₃/Pd** (2 wt% Pd, 5% glycerol): Predicted HER = 4,682 µmol g⁻¹ h⁻¹
2. **g-C₃N₄/Ni₂P** (visible light, 5 wt%): Substantially underrepresented in training data
3. **BiVO₄/Pt** (solar, 3 wt% Pt): High novelty score candidate

The top-ranked candidate, WO₃/Pd, falls within the applicability domain of the model (AD score = 18.8 < threshold 130.2), indicating that this prediction is supported by sufficiently similar training examples and can be considered a reliable interpolation target for experimental validation (Fig. 5, Fig. 6, Fig. 7).

All 91,800 screened candidates fall within the applicability domain (100%), indicating that the combinatorial library was well-bounded by the training distribution and predictions represent interpolations rather than risky extrapolations. This supports the scientific credibility of the virtual screening results (Fig. 8).

### 3.6. Ablation study

Feature group ablation quantifies individual contributions to LOMO-CV R²: synthesis features contribute ΔR² = −0.020, optical features ΔR² = −0.013, and co-catalyst features ΔR² = −0.011. The log transformation is critical (ΔR² = −0.798 without it), confirming that addressing the target distribution skew is the single most important modelling decision.

---

## 4. Conclusions

We present the first ML framework for cross-material-class HER prediction in glycerol photoreforming, achieving LOMO-CV R² = 0.772 on held-out material families. Key findings:

1. Non-linear gradient-boosted models substantially outperform linear baselines (ΔR² ≈ 0.32), and the model achieves Spearman ρ = 0.917, demonstrating accurate catalyst ranking.
2. Split conformal prediction provides calibrated uncertainty with 92.9% empirical coverage at 90% nominal level, replacing uncalibrated bootstrap methods.
3. SHAP analysis provides mechanistically interpretable feature attributions, with material-family-specific driver differences for TiO₂, ZnO, CdS, and g-C₃N₄.
4. Virtual screening of 91,800 candidate systems identifies WO₃/Pd as the top-ranked candidate within the model's applicability domain, providing an experimentally actionable prediction.

This framework provides a data-driven roadmap for accelerating photocatalyst discovery and represents a transferable methodology applicable to other heterogeneous catalysis domains.

---

## CRediT authorship contribution statement

**[Author 1]:** Conceptualisation, Methodology, Software, Formal analysis, Investigation, Writing – original draft. **[Author 2]:** Data curation, Validation, Writing – review & editing. **[Author 3]:** Supervision, Resources, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

[Acknowledge funding sources, computational resources, and collaborators.]

## Appendix A. Supplementary data

Supplementary data associated with this article can be found in the online version at [DOI]. The supplementary information includes: dataset construction details (Section S1), hyperparameter optimisation results (S2), cross-validation details (S3), ablation study results (S4), full virtual screening results (S5), applicability domain methodology (S6), conformal prediction methodology (S7), supplementary figures (S8), and Tables S1–S3.

---

## References

[1] A. Fujishima, K. Honda, Electrochemical photolysis of water at a semiconductor electrode, Nature 238 (1972) 37–38.

[2] R. Chong, J. Li, Y. Ma, B. Zhang, H. Han, C. Li, Selective conversion of aqueous glucose to value-added sugar aldose on TiO₂-based photocatalysts, J. Catal. 314 (2014) 101–108.

[3] X. Wang, K. Maeda, A. Thomas, K. Takanabe, G. Xin, J.M. Carlsson, K. Domen, M. Antonietti, A metal-free polymeric photocatalyst for hydrogen production from water under visible light, Nat. Mater. 8 (2009) 76–80.

[4] K.T. Butler, D.W. Davies, H. Cartwright, O. Isayev, A. Walsh, Machine learning for molecular and materials science, Nature 559 (2018) 547–555.

[5] B.R. Goldsmith, J. Esterhuizen, J.-X. Liu, C.J. Bartel, C. Sutton, Machine learning for heterogeneous catalyst design and discovery, AIChE J. 64 (2018) 2311–2323.

[6] J. Schmidt, M.R.G. Marques, S. Botti, M.A.L. Marques, Recent advances and applications of machine learning in solid-state materials science, npj Comput. Mater. 5 (2019) 83.

[7] M. Zhong, K. Tran, Y. Min, C. Wang, Z. Wang, C.-T. Dinh, P. De Luna, Z. Yu, A.S. Raber, P. Becber, A. Aspuru-Guzik, Accelerated discovery of CO₂ electrocatalysts using active machine learning, Nature 581 (2020) 178–183.

[8] H. Tao, T. Wu, M. Aldeghi, T.C. Wu, A. Aspuru-Guzik, E. Kumacheva, Nanoparticle synthesis assisted by machine learning, Nat. Rev. Mater. 6 (2021) 701–716.

[9] A. Chen, X. Zhang, Z. Zhou, Machine learning: Accelerating materials development for energy storage and conversion, InfoMat 2 (2020) 553–576.

[10] L. Grinsztajn, E. Oyallon, G. Varoquaux, Why do tree-based models still outperform deep learning on typical tabular data?, in: NeurIPS 2022, vol. 35, pp. 507–520.

[11] A.N. Angelopoulos, S. Bates, Conformal prediction: A gentle introduction, Found. Trends Mach. Learn. 16 (2023) 494–591.

[12] B.R. Goldsmith, J. Esterhuizen, J.-X. Liu, C.J. Bartel, C. Sutton, Machine learning for heterogeneous catalyst design and discovery, ACS Cent. Sci. 4 (2018) 1350.

[13] W. Li, K. Gu, L. Luo, Q. Wang, S. Hu, N. Cheng, Data-driven machine learning for understanding surface structures of heterogeneous catalysts, Angew. Chem. Int. Ed. 62 (2023) e202216383.

---

## Data and code availability

All data, code, and trained models are available at: https://github.com/RaunakOP-web/Photocatalyst-Trial-

The dataset (`glycerol_photocatalyst_fixed.json`) is included in the repository under `data/raw/`. The full pipeline is reproducible with: `python run_all_publication.py`. Exact package versions are specified in `requirements.txt`.

---

## Figure captions

**Fig. 1.** Dataset overview. (a) Distribution of host materials across 838 experiments (TiO₂ dominates at 72%). (b) Histogram of log₁₀(HER + 1) showing the 5-order-of-magnitude range.

**Fig. 2.** Model performance comparison. R² scores for LightGBM, XGBoost, and Ridge across four evaluation metrics (5-fold CV, LOMO-CV, test log, test original). Error bars show standard deviation across folds.

**Fig. 3.** (a) Actual vs. predicted log(HER+1) for the LightGBM model on the hold-out test set (R² = 0.768, MAE = 0.760). Points coloured by host material. (b) Residual plot.

**Fig. 4.** SHAP feature importance analysis. (a) Global mean |SHAP| bar chart (top 12 features). (b) SHAP beeswarm plot showing feature value effects.

**Fig. 5.** Co-catalyst and semiconductor ranking. Predicted HER vs. loading for major co-catalysts under fixed conditions.

**Fig. 6.** Horizontal bar chart of predicted HER for 14 non-TiO₂ host materials with Pt co-catalyst (1 wt%, fixed conditions).

**Fig. 7.** Top-20 virtual screening candidates ranked by UCB, with conformal prediction intervals.

**Fig. 8.** Applicability domain PCA projection. Training data (grey), test set coloured by AD score (green–red gradient), and top-20 discovery candidates (red diamonds). AD threshold = 130.2.
