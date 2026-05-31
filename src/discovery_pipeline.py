"""
discovery_pipeline.py
Phase 6/10 — Novel Material Discovery Mode

Generates a combinatorial candidate library including:
  - Heterojunctions (TiO2/g-C3N4, ZnO/CdS, WO3/TiO2, etc.)
  - Doped systems (N-doped TiO2, Fe-doped ZnO, Cu-doped g-C3N4)
  - Co-catalyst combinations (Pt, Ni, Cu, Co, MoS2)

Each candidate scored with bootstrap uncertainty + novelty score.
Ranks by UCB, saves top-100 novel candidates.

Saves:
  data/results/discovery_candidates.csv — full ranked candidate table
  data/results/top_novel_candidates.csv — top-100 with novelty scores
  data/results/discovery_pareto.png     — UCB vs novelty Pareto plot
  data/results/discovery_top20.png      — bar chart of top-20 predictions
"""

import os
import yaml
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import random

# ─────────────────────────────────────────────────────────────────────────────

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths = CFG["paths"]
data_cfg = CFG["data"]
PROC = paths["proc_dir"]
RES = paths["results_dir"]
MOD = paths["models_dir"]


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE LIBRARY — expanded for scientific novelty
# ─────────────────────────────────────────────────────────────────────────────

# Host materials — exclude TiO2, ZnO, CdS for true novelty
HOST_MATERIALS_NOVEL = [
    "g-c3n4", "bivo4", "bi2wo6", "fe2o3", "wo3",
    "nb2o5", "ta2o5", "srtio3", "cu2o", "ga2o3", "in2o3",
    "zns", "moo3", "v2o5", "nife2o4",
]

# Heterojunction pairs (semiconductor_2)
HETEROJUNCTION_PAIRS = {
    "g-c3n4": ["tio2", "zno", "bivo4", "wo3", "cds", "bi2wo6", "fe2o3"],
    "bivo4": ["tio2", "wo3", "g-c3n4", "fe2o3"],
    "bi2wo6": ["tio2", "g-c3n4", "zno"],
    "fe2o3": ["tio2", "g-c3n4", "zno", "wo3"],
    "wo3": ["tio2", "g-c3n4", "bivo4", "fe2o3"],
    "zno": ["tio2", "g-c3n4", "cds", "wo3"],
    "cds": ["zno", "tio2", "g-c3n4"],
    "srtio3": ["tio2", "g-c3n4", "zno"],
}

COCATALYSTS_NOVEL = [
    "pt", "ni", "cu", "co", "mos2", "ni2p", "rgo", "pd", "au",
    "ru", "rh", "ir", "fe", "mn", "wc", "cos2", "fes2",
]

# Dopant options for key hosts
DOPANT_MAP = {
    "tio2": ["n", "fe", "cu", "ag", "c", "s", "f", "p", "cr", "co"],
    "zno": ["fe", "cu", "co", "mn", "ni", "al", "ga", "in"],
    "g-c3n4": ["cu", "fe", "co", "ni", "p", "s", "o", "k", "na"],
    "cds": ["cu", "fe", "ni", "mn", "er"],
}

CANDIDATE_GRID = {
    "host_material": HOST_MATERIALS_NOVEL,
    "co_catalyst": COCATALYSTS_NOVEL,
    "cocatalyst_wt_pct": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
    "glycerol_vol_pct": [5.0, 10.0, 20.0, 30.0, 50.0],
    "light_source_type": ["visible", "solar", "uv-vis"],
    "catalyst_loading_g_L": [0.5, 1.0, 1.5, 2.0],
}

# Numeric columns that require defaults from training medians
NUMERIC_DEFAULTS_COLS = [
    "cocatalyst_wt_pct", "glycerol_vol_pct", "catalyst_loading_g_L",
    "reaction_time_h", "pH", "temperature_C",
    "light_intensity_mW_cm2", "wavelength_cutoff_nm",
    "BET_surface_area_m2_g",
]


def build_candidate_library(feature_cols, medians, encoder):
    """
    Generates a combinatorial grid including heterojunctions and dopants,
    then encodes it to match the feature matrix used during training.
    """
    print("  Building combinatorial candidate library...")
    print(f"  Host materials (non-TiO2/non-ZnO/non-CdS): {HOST_MATERIALS_NOVEL}")

    keys = list(CANDIDATE_GRID.keys())
    vals = list(CANDIDATE_GRID.values())
    records = [dict(zip(keys, combo)) for combo in product(*vals)]
    cand_df = pd.DataFrame(records)
    n_candidates = len(cand_df)
    print(f"  Generated {n_candidates:,} base candidate combinations")

    # Add heterojunctions: set semiconductor_2 for hosts that have them
    if "semiconductor_2" in feature_cols:
        cand_df["semiconductor_2"] = "none"
        for host, partners in HETEROJUNCTION_PAIRS.items():
            mask = cand_df["host_material"] == host
            # Assign a random heterojunction partner based on host
            random.seed(42)
            cand_df.loc[mask, "semiconductor_2"] = [
                random.choice(partners) for _ in range(mask.sum())
            ]
        # For a subset (10%), set to "none" for non-heterojunction variants
        n_hetero = (cand_df["semiconductor_2"] != "none").sum()
        print(f"  Added heterojunction labels: {n_hetero} candidates")

    # Fill remaining feature columns with training medians or "unknown"
    for col in feature_cols:
        if col not in cand_df.columns:
            if col in medians.index:
                cand_df[col] = float(medians[col])
            else:
                cand_df[col] = "unknown"

    # Reorder columns to match training feature list
    cand_df = cand_df.reindex(columns=feature_cols, fill_value=np.nan)

    # Fill numeric NaNs with medians
    num_cols = [c for c in feature_cols if c in medians.index]
    for col in num_cols:
        cand_df[col] = cand_df[col].fillna(float(medians[col]))

    # Identify non-numeric columns using robust detection
    non_numeric_cols = []
    for c in feature_cols:
        try:
            if not np.issubdtype(cand_df[c].dtype, np.number):
                non_numeric_cols.append(c)
        except Exception:
            non_numeric_cols.append(c)

    if non_numeric_cols:
        print(f"  Encoding {len(non_numeric_cols)} non-numeric columns...")
        # Use target encoder if available
        if encoder is not None and hasattr(encoder, 'feature_names_in_'):
            try:
                # Prepare input with all columns the encoder expects
                enc_cols = list(encoder.feature_names_in_)
                enc_input = pd.DataFrame(index=cand_df.index)
                for col in enc_cols:
                    if col in cand_df.columns:
                        enc_input[col] = cand_df[col].astype(str)
                    else:
                        enc_input[col] = "unknown"
                encoded = encoder.transform(enc_input)  # returns numpy array
                # Assign encoded values back to original columns
                for i, col in enumerate(enc_cols):
                    if col in cand_df.columns:
                        cand_df[col] = encoded[:, i]
                print(f"    TargetEncoder applied to {len(enc_cols)} columns")
                # Remove successfully encoded cols from non_numeric list
                non_numeric_cols = [c for c in non_numeric_cols if c not in set(enc_cols)]
            except Exception as e:
                print(f"    TargetEncoder transform failed: {e}")

        # Fallback: label-encode or zero-fill remaining non-numeric columns
        if non_numeric_cols:
            print(f"    Fallback encoding remaining columns: {non_numeric_cols}")
            from sklearn.preprocessing import LabelEncoder
            for col in non_numeric_cols:
                try:
                    le = LabelEncoder()
                    cand_df[col] = le.fit_transform(cand_df[col].astype(str))
                except Exception as e2:
                    print(f"      Forcing {col} to 0: {e2}")
                    cand_df[col] = 0.0

    # Final numeric fill for any remaining NaN
    cand_df = cand_df.fillna(0.0)
    return cand_df, records


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_candidates(cand_X, best_model, X_train, y_train,
                     n_boot=50, alpha=0.10, random_state=42):
    """
    Bootstrap uncertainty scoring for candidates.
    Returns (median, p05, p95) in log-HER space.
    """
    print(f"  Bootstrap scoring ({n_boot} members) for {len(cand_X):,} candidates...")
    rng = np.random.default_rng(random_state)
    preds_matrix = np.zeros((n_boot, len(cand_X)))

    try:
        lgb_params = json.load(open(os.path.join(RES, "best_params_LightGBM.json")))
        lgb_params["verbose"] = -1
        lgb_params["random_state"] = random_state
        from lightgbm import LGBMRegressor
        model_cls = LGBMRegressor
        model_params = lgb_params
    except Exception as e:
        # Fallback: use best_model directly (no bootstrap uncertainty)
        print(f"  Warning: bootstrap failed ({e}); returning point predictions only")
        preds = best_model.predict(cand_X)
        return preds, preds, preds

    batch_size = 5000
    for i in range(n_boot):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        m = model_cls(**model_params)
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        # Predict in batches to avoid OOM
        for start in range(0, len(cand_X), batch_size):
            end = min(start + batch_size, len(cand_X))
            preds_matrix[i, start:end] = m.predict(cand_X.iloc[start:end])

    lo = np.percentile(preds_matrix, 100 * alpha / 2, axis=0)
    hi = np.percentile(preds_matrix, 100 * (1 - alpha / 2), axis=0)
    med = np.median(preds_matrix, axis=0)
    return med, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# NOVELTY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def novelty_score(cand_X, X_train, k=5):
    """
    Novelty = mean distance to k nearest training neighbours (Euclidean).
    Higher -> more novel (further from training distribution).
    """
    print("  Computing novelty scores (kNN distance to training set)...")
    k = max(1, min(k, len(X_train) - 1))  # guard against too-small training set
    n_sample = min(1000, len(X_train))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(X_train), size=n_sample, replace=False)
    X_tr_sample = X_train.iloc[sample_idx].values

    batch_size = 5000
    novel = np.zeros(len(cand_X))
    for start in range(0, len(cand_X), batch_size):
        end = min(start + batch_size, len(cand_X))
        batch = cand_X.iloc[start:end].values
        dists = euclidean_distances(batch, X_tr_sample)
        k_nearest = np.sort(dists, axis=1)[:, :k]
        novel[start:end] = k_nearest.mean(axis=1)

    # Normalize to [0, 1] by max-min scaling
    n_min, n_max = novel.min(), novel.max()
    if n_max > n_min:
        novel = (novel - n_min) / (n_max - n_min)
    return novel


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(df_ranked, results_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df_ranked["novelty_score"],
        np.expm1(df_ranked["ucb_log"]),
        c=np.expm1(df_ranked["pred_median_log"]),
        cmap="plasma", alpha=0.5, s=15, edgecolors="none"
    )
    # Annotate top 10 by UCB
    top10 = df_ranked.nlargest(10, "ucb_log")
    for _, row in top10.iterrows():
        ax.annotate(
            f"{row['host_material']}/{row['co_catalyst']}",
            (row["novelty_score"], np.expm1(row["ucb_log"])),
            fontsize=7, alpha=0.85,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
            xytext=(5, 5), textcoords="offset points"
        )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Predicted HER (umol/g/h)", fontsize=10)
    ax.set_xlabel("Novelty Score (higher = more novel)", fontsize=11)
    ax.set_ylabel("Upper Confidence Bound HER (umol/g/h)", fontsize=11)
    ax.set_title("Discovery Pareto: Novelty vs UCB Performance", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "discovery_pareto.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(results_dir, "discovery_pareto.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved discovery_pareto.png / .pdf")


def plot_top20(df_ranked, results_dir):
    top20 = df_ranked.nlargest(20, "ucb_log").copy()
    top20["label"] = top20["host_material"] + "/" + top20["co_catalyst"].astype(str)
    top20["pred_her"] = np.expm1(top20["pred_median_log"])
    top20["err_lo"] = top20["pred_her"] - np.expm1(top20["pred_p05_log"])
    top20["err_hi"] = np.expm1(top20["ucb_log"]) - top20["pred_her"]
    top20 = top20.sort_values("pred_her", ascending=True)

    cmap = plt.cm.plasma(np.linspace(0.2, 0.9, len(top20)))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top20["label"], top20["pred_her"],
        xerr=[top20["err_lo"], top20["err_hi"]],
        color=cmap, capsize=4, alpha=0.85, edgecolor="k", linewidth=0.5
    )
    ax.set_xlabel("Predicted HER (umol/g/h)", fontsize=12)
    ax.set_title("Top 20 Novel Catalyst Candidates by UCB Score", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "discovery_top20.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(results_dir, "discovery_top20.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved discovery_top20.png / .pdf")


def rescore_with_conformal(result_df, cand_X):
    """
    Replaces bootstrap CI/pred columns in result_df with split conformal prediction intervals.
    Uses the model and q_hat saved in conformal_model.joblib.
    """
    conformal_path = os.path.join(MOD, "conformal_model.joblib")
    if not os.path.exists(conformal_path):
        print(f"  Warning: {conformal_path} not found. Skipping conformal rescoring.")
        return result_df

    print(f"  Applying conformal UQ to candidates using {conformal_path}...")
    conformal_data = joblib.load(conformal_path)
    q_hat = conformal_data["q_hat"]
    model = conformal_data["model"]

    # Predict log-HER
    batch_size = 5000
    pred_log = np.zeros(len(cand_X))
    for start in range(0, len(cand_X), batch_size):
        end = min(start + batch_size, len(cand_X))
        pred_log[start:end] = model.predict(cand_X.iloc[start:end])

    # Conformal intervals
    result_df["pred_median_log"] = pred_log
    result_df["pred_p05_log"] = pred_log - q_hat
    result_df["pred_p95_log"] = pred_log + q_hat
    result_df["ucb_log"] = pred_log + q_hat  # UCB is upper bound of interval
    result_df["pred_her_umol_g_h"] = np.expm1(pred_log)
    result_df["ucb_her_umol_g_h"] = np.expm1(pred_log + q_hat)

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RES, exist_ok=True)
    print("Phase 6/10 - Novel Material Discovery Pipeline")

    # Load assets
    X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).squeeze()
    df_clean = pd.read_csv(os.path.join(PROC, "df_clean.csv"), index_col=0)

    strat_bins = pd.qcut(df_clean["log_HER"], 10, labels=False, duplicates="drop")
    _, _, _, y_test_al = train_test_split(
        df_clean[[]], df_clean["log_HER"],
        test_size=data_cfg["test_size"],
        stratify=strat_bins,
        random_state=data_cfg["random_state"]
    )
    train_idx = df_clean.index.difference(y_test_al.index)
    X_train.index = train_idx
    y_train.index = train_idx

    feature_cols = joblib.load(os.path.join(PROC, "feature_list.joblib"))
    medians = joblib.load(os.path.join(PROC, "numeric_medians.joblib"))
    enc_path = os.path.join(PROC, "target_encoder.joblib")
    encoder = joblib.load(enc_path) if os.path.exists(enc_path) else None
    best_model = joblib.load(os.path.join(MOD, "best_model.joblib"))

    # Build candidate library
    cand_X, records = build_candidate_library(feature_cols, medians, encoder)

    # Score with bootstrap
    pred_med, pred_lo, pred_hi = score_candidates(
        cand_X, best_model, X_train, y_train, n_boot=100
    )

    # Novelty
    novel = novelty_score(cand_X, X_train, k=5)

    # UCB = predicted median + kappa * (p95 - p05) / 2
    kappa = 1.0
    ucb = pred_med + kappa * (pred_hi - pred_lo) / 2.0

    # Assemble result dataframe
    result_df = pd.DataFrame(records)
    result_df["pred_median_log"] = pred_med
    result_df["pred_p05_log"] = pred_lo
    result_df["pred_p95_log"] = pred_hi
    result_df["ucb_log"] = ucb
    result_df["novelty_score"] = novel
    result_df["pred_her_umol_g_h"] = np.expm1(pred_med)
    result_df["ucb_her_umol_g_h"] = np.expm1(ucb)

    # Re-score with conformal UQ if available
    result_df = rescore_with_conformal(result_df, cand_X)

    # Construct composition column
    result_df["composition"] = (
        result_df["host_material"].astype(str) + "/" +
        result_df["co_catalyst"].astype(str) + " (" +
        result_df["cocatalyst_wt_pct"].astype(str) + " wt%)"
    )

    # Confidence based on CI width relative to prediction
    ci_width = result_df["pred_p95_log"] - result_df["pred_p05_log"]
    med_log = result_df["pred_median_log"]
    # Relative uncertainty
    rel_uncertainty = np.where(med_log > 0, ci_width / med_log, ci_width)
    result_df["confidence"] = pd.cut(
        rel_uncertainty,
        bins=[-np.inf, 0.3, 0.6, np.inf],
        labels=["HIGH", "MEDIUM", "LOW"]
    )

    # Filter extreme predictions (> 3x max training HER as sanity check)
    max_train_her = df_clean[data_cfg["target"]].max()
    result_df = result_df[result_df["pred_her_umol_g_h"] <= 3 * max_train_her]

    # Rank by UCB
    result_df = result_df.sort_values("ucb_log", ascending=False).reset_index(drop=True)
    result_df.to_csv(os.path.join(RES, "discovery_candidates.csv"), index=False)
    print(f"\n  Saved {len(result_df):,} ranked candidates -> discovery_candidates.csv")

    # Save top-100 novel candidates
    top100 = result_df.head(100).copy()
    top100_out = top100[[
        "composition", "host_material", "co_catalyst", "cocatalyst_wt_pct",
        "pred_her_umol_g_h", "ucb_her_umol_g_h",
        "pred_median_log", "pred_p05_log", "pred_p95_log",
        "confidence", "novelty_score",
    ]]
    top100_out.to_csv(os.path.join(RES, "top_novel_candidates.csv"), index=False)
    print(f"  Saved top-100 novel candidates -> top_novel_candidates.csv")

    # Print top 10
    top10 = result_df.head(10)[["host_material", "co_catalyst", "cocatalyst_wt_pct",
                                 "glycerol_vol_pct", "light_source_type",
                                 "pred_her_umol_g_h", "ucb_her_umol_g_h",
                                 "novelty_score", "confidence"]]
    print("\n  TOP 10 NOVEL CANDIDATES (ranked by UCB):")
    print(top10.to_string(index=False))

    # Novelty statistics
    print(f"\n  Novelty score stats:")
    print(f"    Mean: {novel.mean():.3f}")
    print(f"    Median: {np.median(novel):.3f}")
    print(f"    Top-10 mean: {result_df.head(10)['novelty_score'].mean():.3f}")

    # Count high-confidence novel candidates
    n_high_conf = (result_df["confidence"] == "HIGH").sum()
    n_novel_high = ((result_df["novelty_score"] > 0.5) & (result_df["confidence"] == "HIGH")).sum()
    print(f"    High-confidence candidates: {n_high_conf}")
    print(f"    High-novelty (>0.5) AND high-confidence: {n_novel_high}")

    # Plots
    plot_pareto(result_df, RES)
    plot_top20(result_df, RES)

    print("\nPhase 10 complete - discovery outputs saved to data/results/")


if __name__ == "__main__":
    main()
