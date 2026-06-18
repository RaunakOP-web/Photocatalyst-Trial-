# Dead Code Elimination Report

This report documents the files and functions identified as dead or obsolete code that have been removed or archived.

| Deleted File/Function | Reason for Deletion | Potential Risks |
| --- | --- | --- |
| `temp_repo/` (directory) | Identical duplicate copy of the repository. | None. |
| `archive_v2/` (directory) | Obsolete code (`preprocess_v2.py`, `train_v2.py`) from previous runs. | None. |
| `src/tune_weights.py` | Leftover weights optimization experiment. | None. |
| `src/update_ensemble.py` | Obsolete blend weighting helper script. | None. |
| `src/compute_spearman.py` | Single-use correlation check script. | None. |
| `dump_shap_data.py` | Replaced by direct outputs of evaluate/SHAP scripts. | None. |
| `run_all.py` | Main runner to be replaced by standardized root entrypoints. | None. |
| `run_all_publication.py` | Monolithic script to be replaced by simple root entrypoints. | None. |
| `automatminer*.log` (files) | Auto-generated training artifacts. | None. |
