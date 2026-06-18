import pytest
import pandas as pd
import numpy as np
from src.discovery.candidate_generation import generate_candidate_grid
from src.discovery.ranking import rank_candidates

def test_generate_candidate_grid():
    feature_cols = [
        "semi_bandgap_eV", "cocat_work_function", "co_catalyst_wt_pct",
        "glycerol_concentration_v_pct", "catalyst_loading_g_L", "light_power_W"
    ]
    medians = pd.Series({
        "semi_bandgap_eV": 3.2,
        "cocat_work_function": 5.65,
        "co_catalyst_wt_pct": 1.0,
        "glycerol_concentration_v_pct": 10.0,
        "catalyst_loading_g_L": 1.0,
        "light_power_W": 300.0
    })
    
    cand_df, records = generate_candidate_grid(feature_cols, medians, encoder=None)
    assert len(cand_df) > 0
    assert len(records) > 0
    assert "semi_bandgap_eV" in cand_df.columns

def test_rank_candidates():
    records = [
        {"host_material": "tio2", "co_catalyst": "pt"},
        {"host_material": "zno", "co_catalyst": "ni"}
    ]
    cand_df = pd.DataFrame({
        "semi_bandgap_eV": [3.2, 3.3]
    })
    predictions = np.array([2.0, 1.5])
    uq_intervals = (np.array([1.5, 1.0]), np.array([2.5, 2.0]))
    X_train = pd.DataFrame({
        "semi_bandgap_eV": [3.2, 3.2, 3.2]
    })
    
    ranked = rank_candidates(cand_df, records, predictions, uq_intervals, X_train, kappa=1.0)
    assert len(ranked) == 2
    assert "ucb_log" in ranked.columns
    assert "novelty_score" in ranked.columns
    # Check that highest UCB is ranked first
    assert ranked.loc[0, "ucb_log"] >= ranked.loc[1, "ucb_log"]
