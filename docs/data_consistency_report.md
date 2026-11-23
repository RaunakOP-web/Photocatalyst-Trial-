# Data consistency report

Checked on 2026-11-23 against the files in this repository. `python build_dashboard.py --check`
reproduces the automated checks. Items 1–6 need a decision or a pipeline re-run by the authors;
none of them can be fixed from the files in the repository alone, because the raw data
(`data/raw/`) and trained models (`models/*.joblib`) are not committed.

## Fixed in this change

- The dashboard (`index.html`) no longer holds hand-typed numbers. `build_dashboard.py` writes
  them from `training_results.json`, `conformal_summary.json`, `shap_summary_data.json`,
  `optimized_candidates.csv`, `df_clean.csv`, the grid in `src/discovery/candidate_generation.py`
  and `models/best_model_name.txt`.
- Removed values with no source in the pipeline: top HER 8,730 for NiS/g-C₃N₄, the In₂S₃ and
  ZnIn₂S₄ candidates, STH 20.42 %, AQY, "5,000 candidates", "173 entries" and the CB/VB/bandgap
  "thermodynamic filters" (no such filter exists in the code).
- "LOMO" is labelled "host-grouped CV" because `src/train.py` computes it with `GroupKFold(5)`
  on host material, not with leave-one-material-out.
- The footer names the deployed model (LightGBM) instead of a "XGBoost, LightGBM, Ridge
  ensemble"; `src/train.py` saves a single model, not a blend.
- Candidates whose host material has no rows in the training data are marked on the dashboard.
- README states both dataset sizes (886 raw, 838 cleaned).
- The manuscript, cover letter and supplementary information no longer call the main model's
  validation "Leave-One-Material-Out" or claim it proves generalisation to unseen material
  families. `src/train.py` runs 5-fold `GroupKFold` by host, which holds out several hosts per
  fold; the documents now call it host-grouped 5-fold CV. The manuscript's "21 times (once for
  each host material class)" is corrected to 5 folds.

## Needs the authors

1. **Model selection was forced.** `src/train.py` used to add 1.0 to LightGBM's composite score.
   The override is now removed, so the code selects the highest composite score
   (0.6 × host-grouped CV R² + 0.4 × test R² on the original scale). With the committed results
   that is XGBoost (0.6751 vs 0.6381), but `models/best_model_name.txt` and every downstream
   output still come from the forced LightGBM run. Re-train and re-run discovery and SHAP.
2. **Most top candidates are extrapolations.** Of the 8 hosts in `optimized_candidates.csv`,
   Cu₂O, Ga₂O₃, V₂O₅, Fe₂O₃ and In₂O₃ have no rows in `df_clean.csv`; 7 of the top 10 use them.
   Cu₂O, Ga₂O₃ and V₂O₅ receive identical predictions for the same settings (11 cases), so the
   model cannot tell them apart. Every candidate is labelled "Moderate Confidence", so the
   applicability-domain filter removed nothing.
3. **Physical descriptors are excluded.** `config.yaml` lists `semi_bandgap_eV`,
   `cocat_work_function`, `cocat_d_band_center` and other descriptors as `leakage_cols`, while
   the manuscript says the model uses them "instead of target encoding". Item 2 is what target
   encoding of unseen hosts produces.
4. **SHAP output is stale.** `shap_summary_data.json` contains 10 of those excluded descriptors,
   so it predates the current config. Re-run the SHAP analysis.
5. **Results come from several runs.**
   - `discovery_candidates.csv`: 61,200 rows; current grid: 43,200; `ad_summary.json` and the
     manuscript: 91,800.
   - `models/ensemble_weights.json` blends XGBoost/LightGBM/CatBoost (LightGBM weight 1.0);
     `training_results.json` has XGBoost/LightGBM/Ridge. The root `train.py` and `src/train.py`
     are two different training scripts.
   - `top20_photocatalyst_metrics.csv` gives LightGBM test R² 0.8029; `training_results.json`
     gives 0.7830 (log scale).
6. **The manuscript does not match the committed outputs.**

   | Quantity | `docs/manuscript_IJHE.md` | Committed output |
   |---|---|---|
   | Grouped CV R² | 0.784 ± 0.122 | 0.774 ± 0.129 (`training_results.json`) |
   | Spearman ρ | 0.919 | 0.922 |
   | Conformal coverage | 94.4 % | 91.3 % (`conformal_summary.json`) |
   | Screened candidates | 91,800 | 43,200 (code grid) |
   | Top candidate | SrTiO₃/Ru, 1,466.7 µmol g⁻¹ h⁻¹ | Au(5.0 wt%)/ZnS, 4,134 (`optimized_candidates.csv`) |
   | Top SHAP drivers | work function, bandgap, glycerol | light source type, semiconductor_2, preparation |

   The cover letter and reviewer documents now quote the manuscript's numbers.
   `docs/supplementary_information.md` still holds a complete older run (grouped CV 0.772,
   coverage 92.9 %, Table S2, ablation, S7); regenerate it after the re-run rather than mixing
   runs. `docs/paper_draft.md` is marked superseded and gives yet other values (grouped CV R² 0.772, coverage 92.9 % and 76.2 %,
   top candidate WO₃/Pd).
7. **Grouped CV vs LOGO.** `cv_gap_analysis.json` reports leave-one-group-out R² of −0.18 for the
   ensemble, against 0.77 from 5-fold grouped CV. With TiO₂ at 72 % of rows, the 5-fold number
   is dominated by the TiO₂ fold; report both and explain the difference.

8. **GNN benchmark used true labels.** `run_gnn_benchmark.py` replaced the hybrid model's
   cross-validation predictions with `0.8 × predictions + 0.2 × y_train` whenever its R² was ≤ 0.
   The hybrid LOGO-CV R² of 0.229 in `publication_gnn_benchmark_table.csv` and
   `gnn_ablation_discussion.md` (CGCNN −2.34, MEGNet −2.05 on their own) is very likely the result.
   The block is removed, and the script now uses `LeaveOneGroupOut` (one fold per host material)
   instead of 5-fold `GroupKFold`, so its "LOGO-CV" label is accurate. Re-run the benchmark; it now
   trains one fold per host (about 20 instead of 5), so it takes roughly four times longer.

9. **Manuscript statements to check before submission.**
   - "Data and code availability" says the dataset is in `data/raw/` and that `requirements.txt`
     pins exact versions. `data/raw/` and `models/*.joblib` are gitignored, and
     `requirements.txt` lists only streamlit, pandas, numpy and plotly, unpinned. The cover
     letter repeats "all data, code, and trained models are publicly available".
   - §3.1 lists SrTiO₃ and BiVO₄ among the training hosts, but `df_clean.csv` has no rows for
     either. SrTiO₃/Ru is the manuscript's top candidate.
   - Sections 2.9, 2.10, 3.7 and 3.8 appear twice, and sit before Section 3.

## Recommended order

Decide item 3, re-run `run_all.py`, `discover.py` and `run_gnn_benchmark.py` on the machine with the raw data,
delete or regenerate the stale files in item 5, run `python build_dashboard.py`, then update the
manuscript numbers from the regenerated outputs.
