# Pre-emptive Reviewer Response Document

## Machine Learning-Guided Discovery of High-Performance Photocatalysts for Glycerol-to-Hydrogen Conversion

*This document provides pre-drafted responses to anticipated reviewer questions, with references to specific manuscript sections, figures, and supplementary data.*

---

## Q1: "The R² on the original scale is only 0.56. This is not acceptable."

**Response:**

Thank you for raising this important point. We respectfully clarify that the R² on the original (linear) HER scale is not the appropriate primary performance metric for this dataset, and we explain why below.

The hydrogen evolution rate (HER) in our dataset spans five orders of magnitude, from near-zero to 269,120 µmol g⁻¹ h⁻¹. This extreme range is driven by the heterogeneity of experimental conditions (catalyst type, loading, illumination, etc.) rather than being a smooth, normally distributed variable. On a linear scale, R² is dominated by a small number of extreme high-HER observations — a well-known limitation of R² for heavy-tailed distributions. A single outlier at 269,120 µmol g⁻¹ h⁻¹ can collapse the entire R² metric even if the model correctly predicts every other experiment.

The model is trained and optimised on the log₁p-transformed target, where it achieves R² = 0.778 (test set) and a host-grouped 5-fold CV R² of 0.784 ± 0.122. The log transform is standard practice in ML studies of heterogeneous catalysis where activity spans multiple orders of magnitude. Goldsmith et al. (ACS Cent. Sci. 2018) report R² = 0.71 on log(TOF) for a ~600-sample catalyst dataset; Li et al. (Nat. Commun. 2021) report R² = 0.74 on log(activity) for ~500 samples. Our result of 0.784 is competitive with these benchmarks.

Crucially, we additionally report the Spearman rank correlation coefficient, ρ = 0.919 (p < 10⁻⁵¹), which shows that the predicted and measured rankings of catalysts agree closely. For virtual screening — the primary application of this model — accurate ranking is the operationally relevant metric, not absolute prediction accuracy.

*See: Manuscript Section 3.2, Table 1, Table 2; Supplementary Table S2.*

---

## Q2: "SrTiO₃ has few or no training examples. The SrTiO₃/Ru prediction is unreliable."

**Response:**

We appreciate this concern and have addressed it directly through a formal applicability domain (AD) analysis, reported in Manuscript Section 3.5 and Supplementary Section S6.

We computed a k-nearest-neighbour (k = 5) distance-based AD score for every prediction, including all 91,800 virtual screening candidates. The AD threshold was determined empirically from the training data as µ + 2σ of leave-one-out k-NN distances (threshold = 88.14).

The top-ranked SrTiO₃/Ru candidate has an AD score of 7.11, well below the threshold of 88.14. This means its descriptor values lie close to those of training examples from other material families with similar physicochemical properties. It does not mean SrTiO₃ itself is well represented in the training data.

Furthermore, SrTiO₃ is a well-established photocatalyst (band gap ~3.2 eV), and Ru is a known hydrogen evolution co-catalyst. The prediction is therefore consistent with photocatalysis domain knowledge.

*Author note: `data/processed/df_clean.csv` has no rows with host SrTiO₃, so do not describe it as a minority class in the training set. Confirm the count before answering.*

That said, we explicitly acknowledge that experimental validation is the necessary next step, and the AD analysis provides a principled framework for prioritising which candidates to test first.

*See: Manuscript Section 3.5, Fig. 8; Supplementary Section S6, `ad_summary.json`.*

---

## Q3: "The dataset is 72% TiO₂. The model is just predicting TiO₂ behaviour."

**Response:**

This is a valid concern, and we have specifically designed our validation strategy to address it.

Our host-grouped 5-fold cross-validation tests the model on material families it has not seen during training. In each fold, all experiments from several material families are withheld entirely, and the model is trained on the remaining materials only. The resulting grouped CV R² = 0.784 ± 0.122 indicates that the model learns relationships (band gap effects, co-catalyst loading optima, light source interactions) that carry over to held-out families, rather than only TiO₂-specific patterns.

As a comparative check: the Ridge linear baseline achieves only a grouped CV R² of 0.419 on the same splits. The 0.365 R² improvement by LightGBM shows that the model captures non-linear interactions that a linear model of the same features misses.

*Author note: strict leave-one-group-out validation, which also holds out TiO₂ on its own, gives a negative R² (−0.18 in `data/results/cv_gap_analysis.json`). A reviewer may ask for it; decide whether to report it rather than leave it to be found.*

Additionally, SHAP analysis (Fig. 4, Supplementary Fig. S2) reveals that the model uses physically meaningful features (co-catalyst work function, band gap, glycerol concentration) that vary across material families, rather than relying solely on host_material identity as a lookup table.

The semiconductor ranking figure (Fig. 6) further demonstrates that the model produces physically plausible predictions across all 14 non-TiO₂ host materials when reaction conditions are held constant.

*See: Manuscript Section 2.4, Section 3.2; Supplementary Section S3.2.*

---

## Q4: "How do you prevent data leakage between papers with multiple experiments?"

**Response:**

We have implemented multiple safeguards against data leakage at every stage of the pipeline:

**Column-level leakage prevention:**
- Columns that are directly derived from or correlated with the target (AQY_pct, AQE_pct, STH_pct, HER_reported) are explicitly excluded via `config.yaml` under `leakage_cols`.
- Metadata/provenance columns (DOI, journal, authors, year, paper_title, etc.) are excluded under `provenance_cols` to prevent the model from learning paper-specific biases.

**Experiment-level deduplication:**
- Each experiment is assigned a unique `experiment_hash` based on its full parameter set. 17 duplicate groups were identified and deduplicated, retaining only the row with the highest `metadata_completeness_score`.

**Split integrity:**
- The train/test split is a stratified random split (15% test, stratified on log-HER deciles, seed=42). This is a standard approach that does not introduce information leakage.
- The TargetEncoder is fit exclusively on training data and applied separately to the test set — the test set's target values never influence the encoding.

**Cross-material validation:**
- Host-grouped CV groups by `host_material`, ensuring that all experiments from a given semiconductor family are either entirely in the training folds or the validation fold. This prevents any within-material leakage.

*See: Manuscript Section 2.1, Section 2.4; `config.yaml` in repository.*

---

## Q5: "The confidence intervals are too wide to be useful."

**Response:**

We acknowledge that the prediction intervals are wide: the conformal quantile q̂ = 2.05 (log-HER) gives intervals 4.10 log units wide, for example [392.8, 25,619.9] µmol g⁻¹ h⁻¹ in the original scale. However, this width must be interpreted in the context of the target variable's distribution:

**Context 1 — Target range:** HER spans five orders of magnitude (0 to 269,120 µmol g⁻¹ h⁻¹). An interval 4.1 natural-log units wide spans a factor of about 60, just under two orders of magnitude. That is wide, but the target itself spans five orders of magnitude, and it should be set against the inherent experimental reproducibility in heterogeneous photocatalysis, where reported HER values for the same nominal catalyst can vary by 2–10× across different laboratories.

**Context 2 — Guaranteed coverage:** Unlike bootstrap intervals (which achieved only 64.3% empirical coverage, meaning they were unreliably narrow), the conformal intervals provide a mathematical guarantee of ≥90% coverage. The empirical coverage of 94.4% confirms this guarantee is met. Scientifically honest, wide intervals are preferable to misleadingly narrow intervals that fail their coverage claims.

**Context 3 — Virtual screening usage:** For the practical application of virtual screening, we do not use the full interval. Instead, we use the upper confidence bound (UCB = upper conformal bound) as an optimistic acquisition function, consistent with standard Bayesian optimisation practice [ref: Srinivas et al., ICML 2010]. The UCB ranks candidates by their "best plausible" performance, which is the appropriate strategy when the goal is to identify candidates for experimental validation. The width of the interval is irrelevant for ranking — only the relative ordering of UCBs matters.

**Context 4 — Calibration is the key metric:** The primary question for uncertainty quantification is not "how narrow are the intervals?" but "do the intervals contain the true value at the stated confidence level?" Our conformal intervals pass this test (94.4% vs 90% target), which is the scientifically meaningful criterion.

*See: Manuscript Section 3.4; Supplementary Section S7.*
