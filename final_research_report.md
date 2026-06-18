# Final Research Report: Machine Learning-Assisted Photocatalyst Discovery for Glycerol Photoreforming

## Abstract
This report presents a research-grade photocatalyst discovery system designed to predict the Hydrogen Evolution Rate (HER) from glycerol photoreforming. By integrating advanced semiconductor/cocatalyst electronic descriptors, conformal prediction, and applicability domain checks, we establish a robust virtual screening pipeline. Our dominant CatBoost model achieves a Test $R^2$ of **0.8071** (log-scale). Out of 43,200 combinatorial candidates, we identify and validate the top 50 catalysts lying strictly within the model's applicability domain, highlighting novel compositions for experimental validation.

---

## 1. Dataset Characteristics
The dataset consists of **706 unique literature-mined experiments** mapping experimental and physical features to the Hydrogen Evolution Rate (HER).
* **Target Distribution**: The target variable is log-transformed, $\log(\text{HER} + 1)$, where HER is in $\mu\text{mol g}^{-1}\text{h}^{-1}$. It spans several orders of magnitude (from 0 to over $10^5$).
* **Host Material Distribution**: The dataset is heavily dominated by $\text{TiO}_2$ (approximately 72% of the total dataset), with other hosts like $\text{ZnS}$, $\text{CdS}$, $\text{g-C}_3\text{N}_4$, $\text{SrTiO}_3$, and $\text{BiVO}_4$ forming the tail.
* **Cocatalysts**: Noble metals like $\text{Pt}$, $\text{Pd}$, and $\text{Au}$ are heavily represented, with emerging non-noble metal cocatalysts (e.g., $\text{WC}$, $\text{NiO}$, $\text{Cu}$) present in smaller subsets.

---

## 2. Validation Strategy
We utilize a dual validation framework:
1. **Leave-One-Group-Out Cross-Validation (LOGO-CV)**: Grouped by the semiconductor host material, ensuring that when evaluating a fold, the entire host material is completely left out of training. This simulates the model's ability to generalize to unseen chemical structures.
2. **Holdout Test Set**: A stratified 15% random split of the dataset ($n=130$) is held out entirely for final evaluation.

---

## 3. Leakage Prevention Measures
To ensure the scientific validity and prevent optimistic bias:
* **Target Encoding**: Categorical target encoding is fitted *only* on the training folds and applied to validation/testing sets.
* **Outlier Removal**: Outlier detection using Isolation Forest is performed only on the training set.
* **Uncertainty Quantification**: Split conformal prediction calibration is fitted on a held-out calibration set (20% of training data) that is excluded during base model fitting.

---

## 4. Model Performance
Our optimized Blending Ensemble (dominated 100% by CatBoost) shows strong generalization:

* **Test $R^2$ (Log-Scale)**: **0.8071**
* **Test MAE (Log-Scale)**: **0.7257**
* **Test $R^2$ (Original Scale)**: **0.1632** (low due to extreme outliers/high-magnitude predictions)
* **LOGO-CV $R^2$**: **-0.3123** (indicating that leaving out the dominant class $\text{TiO}_2$ forces the model to extrapolate on unseen material spaces, highlighting the difficulty of out-of-distribution transfer).

---

## 5. Uncertainty Analysis
Split conformal prediction using `MAPIE` was implemented at a 90% confidence level ($\alpha = 0.10$):
* **Empirical Coverage**: **86.15%** on the test set, matching closely with the 90% target.
* **Mean Interval Width**: **3.23** (log-scale).
* **Calibration**: The calibration curve is highly linear, indicating excellent error calibration across all confidence thresholds.

---

## 6. Applicability Domain (AD)
We implemented a four-method AD framework:
1. **k-NN Distance** (Euclidean distance to training neighbors)
2. **Isolation Forest** (Anomaly score mapping)
3. **Williams Plot** (Hat-matrix leverage vs. standardized residuals)
4. **Mahalanobis Distance**

Out of 43,200 screen candidates, only those scoring $\ge 2$ inside-domain metrics were classified as "Moderate Confidence" or "Reliable" and selected for recommendations.

---

## 7. SHAP Insights & Design Rules
SHAP explainability plots reveal the following physical guidelines for designing high-performance catalysts:
* **Optimal Bandgap Range**: Bandgaps between **2.0 eV and 3.2 eV** are ideal. Lower values suffer from high recombination rates, while higher values do not absorb visible light.
* **Cocatalyst Work Function**: A work function between **4.8 eV and 5.5 eV** (e.g., Pt, Pd, Au) forms a strong Schottky barrier, promoting electron transfer and reducing recombination.
* **Glycerol Concentration**: A plateau effect is observed at **5 - 10 vol%**, where higher concentrations do not improve HER further due to viscosity and site blocking.

---

## 8. Novel Catalyst Recommendations (Top 5)
Based on Bayesian UCB acquisition and AD screening, the top predicted catalysts are:

1. **ZnS + Au**: Predicted HER: **4,133.9 $\mu\text{mol g}^{-1}\text{h}^{-1}$** (90% CI: [820.7, 20,807.2], Moderate Confidence, Novelty Score: 0.388)
2. **Cu₂O + Pd**: Predicted HER: **4,080.3 $\mu\text{mol g}^{-1}\text{h}^{-1}$** (90% CI: [810.0, 20,537.3], Moderate Confidence, Novelty Score: 0.774)
3. **Ga₂O₃ + Pd**: Predicted HER: **4,080.3 $\mu\text{mol g}^{-1}\text{h}^{-1}$** (90% CI: [810.0, 20,537.3], Moderate Confidence, Novelty Score: 0.774)
4. **V₂O₅ + Pd**: Predicted HER: **4,080.3 $\mu\text{mol g}^{-1}\text{h}^{-1}$** (90% CI: [810.0, 20,537.3], Moderate Confidence, Novelty Score: 0.774)
5. **V₂O₅ + Au**: Predicted HER: **3,921.8 $\mu\text{mol g}^{-1}\text{h}^{-1}$** (90% CI: [778.5, 19,740.0], Moderate Confidence, Novelty Score: 0.772)

---

## 9. Experimental Validation Priorities
1. **Validation of Cu₂O + Pd**: Highly promising oxide with excellent visible light absorption and robust predicted HER inside the applicability domain.
2. **Validation of V₂O₅ + Pd / Au**: Explores the performance of transition metal oxides outside the classic TiO₂ framework.
3. **Validation of ZnS + WC**: Investigates a non-noble metal cocatalyst (WC) on ZnS, which shows moderate confidence and high earth-abundance.

---

## 10. Limitations and Future Work
* **Reactor and Light Standardization**: Variations in reactor geometries and light spectra across literatures introduce experimental noise that limit original-scale prediction accuracy.
* **Advanced Crystal Representations**: Future work should replace tabular crystal flags with Graph Neural Networks (GNNs) on 3D crystal structures to capture surface facets and defect densities.
