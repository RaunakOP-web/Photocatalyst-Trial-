# Submission Readiness Checklist

This checklist verifies that the glycerol photocatalyst HER prediction paper and supporting files meet all submission criteria for target journals, particularly the **International Journal of Hydrogen Energy (IJHE)**.

## 1. Manuscript & Supplementary Information (IJHE Guidelines)
- [x] **Word Count Limit**: ≤ 8,000 words. (Actual: ~6,800 words, including references and tables).
- [x] **Reference Style**: Numbered references in brackets, e.g., [1], [2], listed in order of citation.
- [x] **Required Sections**: 
  - Title and Author/Affiliation Placeholders
  - Abstract
  - Introduction
  - Methodology
  - Results and Discussion
  - Conclusions
  - CRediT authorship contribution statement
  - Declaration of competing interest
  - Acknowledgements
  - Appendix A (with detailed tables/figures if needed)
  - References
- [x] **Main Figures**: Limit of 8-10 figures. (Actual: 8 figures in `docs/manuscript_IJHE.md`, plus Fig 9 in AD check).
- [x] **Supplementary Information**:
  - Contains S1-S8 sections detailing preprocessing, hyperparameter search, cross-validation, ablation study, virtual screening, applicability domain, and conformal UQ.
  - Contains Tables S1-S3 (Feature List, Model Comparison, Top-50 Candidates with AD labels).

## 2. Technical Implementation & Scientific Fixes
- [x] **Critical Fix 1: Conformal UQ**
  - [x] Refitted LightGBM model on proper-train subset (`models/conformal_model.joblib`).
  - [x] Calculated conformal quantile `q_hat` on held-out calibration set.
  - [x] Empirical coverage on test set meets/exceeds 90% target (Actual: **92.86%**).
  - [x] Overwrote virtual screening candidates with conformal prediction intervals in `discovery_candidates.csv` and `top_novel_candidates.csv`.
- [x] **Critical Fix 2: R² Framing & Spearman Correlation**
  - [x] Framed the original test R² of 0.46 in context of heterogeneous datasets and compared it to literature benchmarks.
  - [x] Calculated Spearman rank correlation on test set (Actual: **0.917**, p < 0.001) to demonstrate excellent ranking capability.
  - [x] Appended Spearman metrics to `data/results/training_results.json`.
- [x] **Critical Fix 3: Applicability Domain (AD) Check**
  - [x] Implemented k-NN distance-based AD check (k=5).
  - [x] Set threshold based on training distribution (mean + 2σ).
  - [x] Verified that test set points are mostly within AD (Actual: **99.2%**).
  - [x] Added `ad_score`, `within_ad`, and `ad_label` to `discovery_candidates.csv`.
  - [x] Confirmed WO₃/Pd is inside the AD (score 38.43 < threshold 130.17) and thus a trustworthy candidate.
  - [x] Generated PCA projection plot (`fig_applicability_domain.png` / `fig9_applicability_domain.png`).

## 3. Mandatory Scripts Execution Status
| Phase | Script Name | Verification Status | Notes |
|---|---|---|---|
| Phase 0a | `src/preprocess.py` | Verified / Success | Preprocesses data and creates clean df |
| Phase 0b | `src/train.py` | Verified / Success | Trains models, executes hyperparameter tuning |
| Phase 0c | `src/evaluate.py` | Verified / Success | Computes standard metrics (log/original R2, MAE) |
| Phase 4 | `src/descriptor_builder.py` | Verified / Success | Builds scientific descriptors |
| Phase 5 | `src/conformal.py` | Verified / Success | Calculates conformal UQ calibration & saves model |
| Phase 6 | `src/discovery_pipeline.py` | Verified / Success | Combinatorial screening and conformal scoring |
| Phase 8 | `src/ablation_study.py` | Verified / Success | Performs feature ablation analysis |
| Phase 9 | `src/shap_analysis.py` | Verified / Success | Generates SHAP explanation values & plots |
| Phase 10 | `src/applicability_domain.py` | Verified / Success | Checks AD on test set and discovery candidates |
| Phase 11 | `src/manuscript_figures.py` | Verified / Success | Generates and compiles all submission figures |

## 4. Package Artifacts to Submit
- [x] **Manuscript**: `docs/manuscript_IJHE.md`
- [x] **Supplementary Info**: `docs/supplementary_information.md`
- [x] **Cover Letter**: `docs/cover_letter.md`
- [x] **Pre-emptive Rebuttals**: `docs/reviewer_responses_preemptive.md`
- [x] **Figures**: High-resolution PNGs/PDFs/SVGs in `data/results/figures/`
