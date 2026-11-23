# Data consistency report

Last checked 2026-11-23 against the files in this repository. `python build_dashboard.py --check`
reproduces the automated checks on the pipeline outputs.

## What was fixed

**Code**
- `src/train.py` selects the model with the highest composite score. It used to add 1.0 to
  LightGBM's score so LightGBM always won.
- `run_gnn_benchmark.py` no longer blends 20 % of the true labels into the hybrid model's
  cross-validation predictions when its R² is ≤ 0, and it now uses `LeaveOneGroupOut`, so its
  "LOGO-CV" label is accurate.
- `build_dashboard.py` writes every dashboard number from the pipeline outputs and checks them
  against each other. `index.html` holds no hand-typed values.
- `requirements.txt` lists all pipeline dependencies (it held only the dashboard's four).

**Dashboard**
- Removed values with no source in the pipeline (8,730 for NiS/g-C₃N₄, the In₂S₃ and ZnIn₂S₄
  candidates, STH 20.42 %, AQY, "5,000 candidates", "173 entries", the CB/VB/bandgap filters).
- Labels say "host-grouped CV" and name the deployed model; candidates whose host has no
  training rows are marked.

**Manuscript (`docs/manuscript_IJHE.md`)**
- The main validation is described as host-grouped 5-fold CV (it is `GroupKFold(5)`), not
  "Leave-One-Material-Out", and no longer claimed to prove generalisation to unseen families.
- §2.6: the library has 15 host materials, not 14 (15 × 17 × 6 × 5 × 3 × 4 = 91,800; NiFe₂O₄ was
  missing from the list).
- §3.1: host shares and mean HER recomputed from `df_clean.csv` (CdS 2.6 % not 6 %, g-C₃N₄ 2.5 %
  not 5 %, mean 11,300 not 6,854); SrTiO₃ and BiVO₄, which have no rows, are no longer listed
  as training hosts.
- §3.5 and Conclusions: no longer claim degenerate predictions were resolved; Table 3 rows 7–10
  are identical predictions for four co-catalysts.
- Sections 2.9, 2.10, 3.7 and 3.8 appeared twice, and 3.7–3.10 sat inside Methods. Duplicates
  removed, 2.9/2.10 renumbered 2.7/2.8, 3.7–3.10 moved after 3.6. Fixed broken `\text` LaTeX.
- References: removed the duplicated [14]–[24] block and the duplicate Goldsmith entry ([12],
  wrong journal); later entries renumbered.
- Data availability no longer says the raw dataset and models are in the repository or that
  versions are pinned. It contains an **[AUTHORS: …]** placeholder to complete.

**Cover letter, supplementary, reviewer documents**
- All quote the manuscript's numbers (grouped CV 0.784 ± 0.122, ρ 0.919, coverage 94.4 %,
  SrTiO₃/Ru 1,466.7, AD threshold 88.14 = µ + 1σ, 15 hosts).
- Supplementary Tables S2, S3, S3.1–S3.3, S6 and S7 now match the manuscript run; cells the
  manuscript does not report are "—", and Table S4 (ablation) is marked as from an earlier run.
- Reviewer documents: corrected the meaning of Spearman ρ and the interval width, removed claims
  the repository does not support, and added author notes where a reviewer could catch a problem.
- `docs/paper_draft.md` is marked as superseded.

## Still open

These need either a pipeline re-run on the machine that holds `data/raw/` and the trained models,
or an author decision. Neither is in the repository.

1. **Re-run the pipeline.** The committed outputs come from several different runs, none of which
   is the run the manuscript describes:
   - `training_results.json` (grouped CV 0.774, coverage 91.3 % in `conformal_summary.json`),
   - `optimized_candidates.csv` (43,200-candidate grid, top Au/ZnS),
   - `discovery_candidates.csv` (61,200 rows, 2 light types),
   - `ad_summary.json` (91,800, matches the manuscript),
   - `shap_summary_data.json` (hand-coded descriptors),
   - `publication_gnn_benchmark_table.csv` (hybrid 0.229, produced with the label blending).

   Current code differs from the manuscript run in three ways: matminer descriptors replace the
   hand-coded ones (commit f411503), the grid in `src/discovery/candidate_generation.py` has 12
   hosts and 15 co-catalysts, and model selection is no longer forced. After the re-run, update the
   manuscript, supplementary and cover letter from the new outputs, run `python build_dashboard.py`,
   and remove the "Needs a re-run" notes in `gnn_ablation_discussion.md`.
2. **Data availability.** Decide whether to publish the raw dataset and trained models (for
   example on Zenodo) and fill the placeholders in the manuscript and cover letter.
3. **Pin dependency versions** with `pip freeze` in the environment that produces the final results.
4. **Strict leave-one-group-out R² is negative** (−0.18 in `cv_gap_analysis.json`). Decide whether
   to report it; the reviewer documents flag it.
5. **Unverified literature claims.** The comparison table (Goldsmith et al. 2018: ~600 samples,
   R² 0.71 on log TOF; Li et al. 2021: ~500 samples, R² 0.74) and the supplementary's "81 papers"
   vs the manuscript's "over 90 peer-reviewed articles" need checking against the sources.
6. **Uncited references.** The body cites only [5], [7] and [12]; most reference entries are never
   cited in the text.
