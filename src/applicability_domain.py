"""
applicability_domain.py
Applicability domain (AD) check using k-NN distance in feature space.

Points far from training data (high distance) are outside the AD.
  Low  score = inside AD = trustworthy prediction
  High score = outside AD = extrapolation = treat with caution

AD threshold = mean + 2 * std of within-training k-NN distances.

Also generates a 2D PCA projection coloured by AD score (Fig 9).

Saves:
  data/results/ad_summary.json         -- threshold + test coverage
  data/results/top_20_candidates.csv   -- top 20 with AD labels
  data/results/figures/fig_applicability_domain.png/pdf/svg
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC_DIR    = "data/processed"
RESULTS_DIR = "data/results"
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
K = 5  # use 5 nearest neighbors


def compute_ad(X_train, X_query, k=K):
    """
    Compute applicability domain score using k-NN distance.
    Returns: mean distance to k nearest training neighbors.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_query)
    return distances.mean(axis=1)


def encode_discovery_candidates(disc, feature_cols, X_train):
    """
    Map the discovery candidate raw columns into the encoded feature space
    used by the model (all-numeric, same columns as X_train).
    
    The discovery CSV has raw string columns (host_material, co_catalyst, etc.)
    while X_train has target-encoded versions. We use the target encoder if
    available, otherwise label-encode + fill missing columns with medians.
    """
    # Try to load target encoder
    enc_path = os.path.join(PROC_DIR, "target_encoder.joblib")
    encoder = joblib.load(enc_path) if os.path.exists(enc_path) else None
    medians = joblib.load(os.path.join(PROC_DIR, "numeric_medians.joblib"))
    
    # Build a dataframe matching the feature columns
    result = pd.DataFrame(index=disc.index)
    
    # Map discovery columns to feature columns
    col_map = {
        "cocatalyst_wt_pct": "co_catalyst_wt_pct",
        "glycerol_vol_pct": "glycerol_concentration_v_pct",
    }
    
    for col in feature_cols:
        src_col = col
        # Check for column name mappings
        for disc_name, feat_name in col_map.items():
            if col == feat_name and disc_name in disc.columns:
                src_col = disc_name
                break
        
        if src_col in disc.columns:
            result[col] = disc[src_col]
        elif col in medians.index:
            result[col] = float(medians[col])
        else:
            result[col] = 0.0
    
    # Identify non-numeric columns that need encoding
    non_numeric = []
    for col in feature_cols:
        try:
            result[col] = pd.to_numeric(result[col])
        except (ValueError, TypeError):
            non_numeric.append(col)
    
    # Encode non-numeric columns
    if non_numeric and encoder is not None and hasattr(encoder, 'feature_names_in_'):
        try:
            enc_cols = list(encoder.feature_names_in_)
            enc_input = pd.DataFrame(index=result.index)
            for col in enc_cols:
                if col in result.columns:
                    enc_input[col] = result[col].astype(str)
                else:
                    enc_input[col] = "unknown"
            encoded = encoder.transform(enc_input)
            for i, col in enumerate(enc_cols):
                if col in result.columns:
                    result[col] = encoded[:, i]
            non_numeric = [c for c in non_numeric if c not in set(enc_cols)]
        except Exception as e:
            print(f"    TargetEncoder failed: {e}")
    
    # Fallback: label-encode remaining non-numeric columns  
    for col in non_numeric:
        try:
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].astype(str))
        except Exception:
            result[col] = 0.0
    
    result = result.fillna(0.0)
    return result[feature_cols].values


def run_ad_check():
    print("=" * 60)
    print("APPLICABILITY DOMAIN CHECK")
    print("=" * 60)

    X_train = pd.read_csv(f"{PROC_DIR}/X_train.csv")
    X_test  = pd.read_csv(f"{PROC_DIR}/X_test.csv")
    feature_cols = X_train.columns.tolist()

    X_train_vals = X_train.values
    X_test_vals  = X_test.values

    # Compute AD threshold from training data itself (leave-one-out style)
    nn = NearestNeighbors(n_neighbors=K + 1, metric="euclidean")
    nn.fit(X_train_vals)
    train_dists, _ = nn.kneighbors(X_train_vals)
    # Exclude self (index 0), use remaining K neighbors
    train_ad = train_dists[:, 1:].mean(axis=1)

    # AD threshold = mean + 2 * std of training distances
    ad_threshold = train_ad.mean() + 2 * train_ad.std()
    print(f"  AD threshold: {ad_threshold:.4f}")
    print(f"  Training AD mean: {train_ad.mean():.4f}, std: {train_ad.std():.4f}")

    # Score test set
    test_ad = compute_ad(X_train_vals, X_test_vals)
    within_ad_test = (test_ad <= ad_threshold).sum()
    print(f"  Test set within AD: {within_ad_test}/{len(X_test)} "
          f"({within_ad_test / len(X_test) * 100:.1f}%)")

    # Score virtual screening candidates
    disc_path = f"{RESULTS_DIR}/discovery_candidates.csv"
    disc = None
    sort_col = "pred_her_umol_g_h"
    
    if os.path.exists(disc_path):
        disc = pd.read_csv(disc_path)
        print(f"  Encoding {len(disc):,} discovery candidates into feature space...")
        
        # Encode discovery candidates into the same feature space as X_train
        X_disc = encode_discovery_candidates(disc, feature_cols, X_train)
        
        # Process in batches to avoid OOM
        batch_size = 10000
        disc_ad = np.zeros(len(X_disc))
        for start in range(0, len(X_disc), batch_size):
            end = min(start + batch_size, len(X_disc))
            disc_ad[start:end] = compute_ad(X_train_vals, X_disc[start:end])
            if (start // batch_size) % 5 == 0 and start > 0:
                print(f"    Processed {end:,}/{len(X_disc):,} candidates...")

        disc["ad_score"]  = disc_ad
        disc["within_ad"] = disc_ad <= ad_threshold
        disc["ad_label"]  = disc["within_ad"].map(
            {True: "within_AD", False: "outside_AD"}
        )
        disc.to_csv(disc_path, index=False)

        within = disc["within_ad"].sum()
        outside = (~disc["within_ad"]).sum()
        print(f"  Discovery candidates within AD:  {within:,}")
        print(f"  Discovery candidates outside AD: {outside:,}")

        # Determine sort column
        if sort_col not in disc.columns:
            sort_col = "ucb_her_umol_g_h" if "ucb_her_umol_g_h" in disc.columns else "ucb_log"

        # Show top candidate AD status
        top = disc.sort_values(sort_col, ascending=False).head(10)
        display_cols = ["host_material", "co_catalyst", sort_col, "ad_score", "ad_label"]
        display_cols = [c for c in display_cols if c in top.columns]
        print("\n  Top 10 candidates with AD status:")
        print(top[display_cols].to_string(index=False))

        # Save top 20 with AD labels
        top20 = disc.sort_values(sort_col, ascending=False).head(20)
        top20.to_csv(f"{RESULTS_DIR}/top_20_candidates.csv", index=False)
        print(f"\n  Saved top_20_candidates.csv with AD labels")

    # Save summary
    summary = {
        "ad_threshold": round(float(ad_threshold), 4),
        "k_neighbors": K,
        "training_ad_mean": round(float(train_ad.mean()), 4),
        "training_ad_std": round(float(train_ad.std()), 4),
        "test_within_ad_count": int(within_ad_test),
        "test_total": len(X_test),
        "test_within_ad_pct": round(float(within_ad_test / len(X_test)) * 100, 1),
    }
    if disc is not None:
        summary["discovery_within_ad_count"] = int(disc["within_ad"].sum())
        summary["discovery_outside_ad_count"] = int((~disc["within_ad"]).sum())
        summary["discovery_total"] = len(disc)
        # WO3/Pd specific status
        wo3_pd = disc[(disc["host_material"] == "wo3") & (disc["co_catalyst"] == "pd")]
        if len(wo3_pd) > 0:
            best_wo3_pd = wo3_pd.sort_values(sort_col, ascending=False).iloc[0]
            summary["wo3_pd_ad_score"] = round(float(best_wo3_pd["ad_score"]), 4)
            summary["wo3_pd_ad_label"] = str(best_wo3_pd["ad_label"])
            summary["wo3_pd_pred_her"] = round(float(best_wo3_pd.get(sort_col, 0)), 1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/ad_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved ad_summary.json")

    # ── Generate AD Figure ─────────────────────────────────────────────
    print("\n  Generating applicability domain PCA figure...")
    os.makedirs(FIG_DIR, exist_ok=True)

    # PCA on training data
    pca = PCA(n_components=2, random_state=42)
    train_2d = pca.fit_transform(X_train_vals)
    test_2d  = pca.transform(X_test_vals)

    fig, ax = plt.subplots(figsize=(7.2, 5.5))

    # Training points in grey
    ax.scatter(train_2d[:, 0], train_2d[:, 1],
               c="lightgrey", s=8, alpha=0.4, edgecolors="none",
               label=f"Training (n={len(X_train)})", zorder=1, rasterized=True)

    # Test points coloured by AD score
    sc = ax.scatter(test_2d[:, 0], test_2d[:, 1],
                    c=test_ad, cmap="RdYlGn_r", s=25, alpha=0.8,
                    edgecolors="k", linewidths=0.3,
                    label=f"Test (n={len(X_test)})", zorder=2,
                    vmin=0, vmax=ad_threshold * 1.5)

    # Top 20 candidates as red diamonds (only if we successfully encoded them)
    if disc is not None:
        try:
            top20_disc = disc.sort_values(sort_col, ascending=False).head(20)
            X_top20_enc = encode_discovery_candidates(top20_disc, feature_cols, X_train)
            top20_2d = pca.transform(X_top20_enc)
            ax.scatter(top20_2d[:, 0], top20_2d[:, 1],
                       c="red", marker="D", s=50, alpha=0.9,
                       edgecolors="k", linewidths=0.6,
                       label="Top-20 candidates", zorder=3)

            # Annotate top 5
            for i in range(min(5, len(top20_disc))):
                row = top20_disc.iloc[i]
                lbl = f"{row.get('host_material', '?')}/{row.get('co_catalyst', '?')}"
                ax.annotate(lbl, (top20_2d[i, 0], top20_2d[i, 1]),
                            fontsize=7, alpha=0.85,
                            xytext=(8, 5), textcoords="offset points",
                            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        except Exception as e:
            print(f"  Warning: Could not plot top-20 candidates on PCA: {e}")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("AD score (k-NN distance)", fontsize=9)

    # AD threshold annotation
    ax.text(0.03, 0.95, f"AD threshold = {ad_threshold:.2f}",
            transform=ax.transAxes, fontsize=8, color="green",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.8))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Applicability Domain: PCA Projection with k-NN Distance Scoring",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.2)

    for ext in ["png", "pdf", "svg"]:
        dpi = 300 if ext == "png" else None
        fig.savefig(os.path.join(FIG_DIR, f"fig_applicability_domain.{ext}"),
                    dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_applicability_domain.png / .pdf / .svg")

    return ad_threshold


if __name__ == "__main__":
    run_ad_check()
