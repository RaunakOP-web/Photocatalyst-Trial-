# Cover Letter

**To:** Editor, *International Journal of Hydrogen Energy*

**Date:** [Submission Date]

**Re:** Manuscript submission — "Machine Learning-Guided Discovery of High-Performance Photocatalysts for Glycerol-to-Hydrogen Conversion via Photoreforming"

---

Dear Editor,

We are pleased to submit our manuscript entitled "Machine Learning-Guided Discovery of High-Performance Photocatalysts for Glycerol-to-Hydrogen Conversion via Photoreforming" for consideration as a Research Article in the *International Journal of Hydrogen Energy*.

This work presents a machine learning framework for predicting hydrogen evolution rates from glycerol photoreforming experiments and identifying novel high-performance photocatalyst candidates. The framework is trained on a curated database of 838 experiments spanning 21 semiconductor families, representing — to our knowledge — the largest ML study dedicated to glycerol photocatalyst screening.

**Key findings:**

- Our LightGBM ensemble achieves a Leave-One-Material-Out cross-validation R² of **0.772 ± 0.114** and Spearman ρ = **0.917**, demonstrating transferable generalisation to unseen material families and accurate catalyst ranking.
- **Split conformal prediction** provides calibrated uncertainty quantification with **92.9% empirical coverage** at the 90% nominal level, replacing uncalibrated bootstrap methods with mathematically guaranteed prediction intervals.
- A combinatorial virtual screening of **91,800 candidate systems** across 14 non-TiO₂ host materials identifies WO₃/Pd as the top-ranked candidate (predicted HER = 4,682 µmol g⁻¹ h⁻¹), validated as lying **within the model's applicability domain**.

**Why IJHE:** This work directly addresses IJHE's core scope of hydrogen production from renewable sources. Glycerol photoreforming is a key pathway for coupling biodiesel waste valorisation with green hydrogen generation. The combination of ML-driven materials discovery, rigorous uncertainty quantification, and experimentally actionable predictions aligns with the journal's emphasis on advancing hydrogen energy technologies through computational and experimental innovation.

**Novelty statement:** While prior ML studies of photocatalysis have used small datasets, single material families, or uncalibrated uncertainty estimates, this work is the first to combine: (i) an 838-experiment curated dataset spanning 21 material families; (ii) Leave-One-Material-Out validation proving cross-material generalisation; (iii) split conformal prediction with finite-sample coverage guarantees; (iv) k-NN applicability domain assessment for virtual screening candidates — all applied specifically to glycerol photoreforming.

**Suggested reviewers:**

1. **Prof. Karthik Shankar** — University of Alberta, Canada. Expert in photocatalytic hydrogen production and machine learning for materials. Email: kshankar@ualberta.ca
2. **Prof. Junwang Tang** — Tsinghua University, China (formerly UCL). Expert in photocatalytic water splitting and solar fuels. Email: tangjw@mail.tsinghua.edu.cn
3. **Prof. Keisuke Takahashi** — Hokkaido University, Japan. Expert in machine learning for catalyst discovery. Email: keisuke.takahashi@sci.hokudai.ac.jp

**Conflict of interest:** The authors declare no competing financial interests or personal relationships that could have influenced the reported work.

All data, code, and trained models are publicly available at https://github.com/RaunakOP-web/Photocatalyst-Trial- for full reproducibility.

We believe this work will be of significant interest to the IJHE readership and look forward to your response.

Sincerely,

[Author 1]
[Department, University]
[Email]
[ORCID]
