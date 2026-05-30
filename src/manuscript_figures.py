"""
manuscript_figures.py
Phase 11 — Publication-Grade Manuscript Figure Generation

Generates all figures at 300 DPI (PNG) + vector (PDF/SVG) for journal submission.
All figures follow Nature/ACS journal formatting:
  - Arial/Helvetica font, 8–10 pt axis labels, tight layout
  - Panel labels (a, b, c, ...) in bold
  - Colour-blind friendly palettes (Wong 2011)

Outputs go to data/results/figures/ directory.

Figures:
  Fig 1: Dataset overview (material distribution + HER distribution)
  Fig 2: Model comparison table (as formatted bar chart)
  Fig 3: Actual vs predicted (best model, log scale)
  Fig 4: SHAP importance + beeswarm (2-panel)
  Fig 5: LOMO-CV generalization analysis
  Fig 6: Discovery top-20 candidates
  Fig 7: Uncertainty calibration
  Fig 8: Ablation study
"""

import os
import json
import yaml
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL STYLE SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Wong (2011) colour-blind safe palette
WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermillion": "#D55E00",
    "pink":    "#CC79A7",
}

PALETTE_8 = list(WONG.values())

def set_journal_style(font_size=9):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def panel_label(ax, label, x=-0.15, y=1.05, fontsize=11):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="right")


def save_fig(fig, name, fig_dir):
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(fig_dir, f"{name}.{ext}")
        dpi = 300 if ext == "png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf / .svg")


# ─────────────────────────────────────────────────────────────────────────────

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths = CFG["paths"]
data_cfg = CFG["data"]
PROC = paths["proc_dir"]
RES = paths["results_dir"]
MOD = paths["models_dir"]
FIG_DIR = os.path.join(RES, "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    set_journal_style(font_size=9)

    # ── Load common data ──────────────────────────────────────────────────────
    df_clean = pd.read_csv(os.path.join(PROC, "df_clean.csv"), index_col=0)
    X_test   = pd.read_csv(os.path.join(PROC, "X_test.csv"))
    y_test   = pd.read_csv(os.path.join(PROC, "y_test.csv")).squeeze()
    X_train  = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    y_train  = pd.read_csv(os.path.join(PROC, "y_train.csv")).squeeze()

    from sklearn.model_selection import train_test_split
    strat_bins = pd.qcut(df_clean["log_HER"], 10, labels=False, duplicates="drop")
    _, _, _, y_test_al = train_test_split(
        df_clean[[]], df_clean["log_HER"],
        test_size=data_cfg["test_size"],
        stratify=strat_bins,
        random_state=data_cfg["random_state"]
    )
    test_idx  = y_test_al.index
    train_idx = df_clean.index.difference(test_idx)
    X_test.index  = test_idx
    y_test.index  = test_idx
    X_train.index = train_idx
    y_train.index = train_idx

    best_model = joblib.load(os.path.join(MOD, "best_model.joblib"))
    with open(os.path.join(MOD, "best_model_name.txt")) as fh:
        best_name = fh.read().strip()
    with open(os.path.join(RES, "training_results.json")) as fh:
        metrics = json.load(fh)

    preds = best_model.predict(X_test)
    residuals = y_test.values - preds

    # ════════════════════════════════════════════════════════════════════════
    # FIG 1 — Dataset Overview
    # ════════════════════════════════════════════════════════════════════════
    print("Generating Fig 1 — Dataset Overview...")
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.4)

    # (a) Material distribution
    ax_a = fig.add_subplot(gs[0, 0])
    mat_counts = df_clean["host_material"].value_counts().head(12)
    colours = [WONG["blue"] if "tio2" in m.lower() else WONG["sky"] for m in mat_counts.index]
    bars = ax_a.barh(mat_counts.index[::-1], mat_counts.values[::-1],
                     color=colours[::-1], edgecolor="k", linewidth=0.4, alpha=0.9)
    ax_a.set_xlabel("Number of experiments")
    ax_a.set_title("Host material distribution")
    panel_label(ax_a, "a")

    # (b) HER distribution
    ax_b = fig.add_subplot(gs[0, 1])
    her_vals = df_clean[data_cfg["target"]].dropna()
    ax_b.hist(np.log10(her_vals + 1), bins=30, color=WONG["orange"],
              edgecolor="k", linewidth=0.3, alpha=0.85)
    ax_b.set_xlabel("log₁₀(HER + 1)  [µmol g⁻¹ h⁻¹]")
    ax_b.set_ylabel("Count")
    ax_b.set_title("HER distribution (n = {:,})".format(len(her_vals)))
    panel_label(ax_b, "b")

    save_fig(fig, "fig1_dataset_overview", FIG_DIR)

    # ════════════════════════════════════════════════════════════════════════
    # FIG 2 — Model Comparison
    # ════════════════════════════════════════════════════════════════════════
    print("Generating Fig 2 — Model Comparison...")
    metric_keys = ["CV_R2_mean", "LOMO_CV_R2_mean", "Test_R2_log", "Test_R2_original"]
    model_names = list(metrics.keys())
    x = np.arange(len(metric_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    colours_m = [WONG["blue"], WONG["orange"], WONG["green"]]
    for i, (mname, col) in enumerate(zip(model_names, colours_m)):
        vals = [metrics[mname].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=mname, color=col,
                      alpha=0.85, edgecolor="k", linewidth=0.4)
        # Error bars for CV metrics
        stds = [metrics[mname].get("CV_R2_std", 0), metrics[mname].get("LOMO_CV_R2_std", 0), 0, 0]
        ax.errorbar(x + i * width, vals, yerr=stds, fmt="none",
                    ecolor="k", elinewidth=0.8, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["5-fold CV R²", "LOMO-CV R²", "Test R² (log)", "Test R² (orig)"],
                       rotation=15, ha="right")
    ax.set_ylabel("R² score")
    ax.set_title("Model performance comparison (best: {})".format(best_name))
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="gray", linestyle=":", lw=0.8, alpha=0.7)
    panel_label(ax, "a", x=-0.08)
    save_fig(fig, "fig2_model_comparison", FIG_DIR)

    # ════════════════════════════════════════════════════════════════════════
    # FIG 3 — Actual vs Predicted + Residuals
    # ════════════════════════════════════════════════════════════════════════
    print("Generating Fig 3 — Actual vs Predicted...")
    host_test = df_clean.loc[test_idx, "host_material"].fillna("unknown")
    unique_mats = host_test.unique()
    mat_colour_map = {m: PALETTE_8[i % 8] for i, m in enumerate(unique_mats)}
    point_colours = host_test.map(mat_colour_map).values

    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    fig = plt.figure(figsize=(7.2, 3.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(y_test, preds, c=point_colours, alpha=0.6, s=18,
                 edgecolors="none", rasterized=True)
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    ax_a.plot([mn, mx], [mn, mx], "k--", lw=1.0)
    ax_a.set_xlabel("Actual log(HER+1)")
    ax_a.set_ylabel("Predicted log(HER+1)")
    ax_a.set_title(f"Actual vs predicted ({best_name})\nR²={r2:.3f}, MAE={mae:.3f}")
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(preds, residuals, c=point_colours, alpha=0.5, s=14,
                 edgecolors="none", rasterized=True)
    ax_b.axhline(0, color="k", lw=0.8, linestyle="--")
    ax_b.set_xlabel("Predicted log(HER+1)")
    ax_b.set_ylabel("Residual")
    ax_b.set_title("Residual plot")
    panel_label(ax_b, "b")

    save_fig(fig, "fig3_actual_vs_predicted", FIG_DIR)

    # ════════════════════════════════════════════════════════════════════════
    # FIG 4 — SHAP Analysis
    # ════════════════════════════════════════════════════════════════════════
    print("Generating Fig 4 — SHAP Analysis...")
    try:
        import shap
        if best_name in ("XGBoost", "LightGBM"):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

            fig = plt.figure(figsize=(7.2, 7.0))
            gs = gridspec.GridSpec(2, 1, hspace=0.5)

            # (a) SHAP importance bar
            ax_a = fig.add_subplot(gs[0, 0])
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False,
                              max_display=12, color=WONG["blue"])
            ax_a.set_title("SHAP feature importance")
            panel_label(ax_a, "a")

            # (b) SHAP beeswarm
            ax_b = fig.add_subplot(gs[1, 0])
            shap.summary_plot(shap_values, X_test, show=False, max_display=12)
            ax_b.set_title("SHAP beeswarm")
            panel_label(ax_b, "b")

            save_fig(fig, "fig4_shap_analysis", FIG_DIR)
        else:
            print("  Skipping SHAP fig (best model not tree-based)")
    except Exception as e:
        print(f"  SHAP figure skipped: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # FIG 5 — Per-Material LOMO-CV Performance
    # ════════════════════════════════════════════════════════════════════════
    print("Generating Fig 5 — Per-material performance...")
    mat_metrics_path = os.path.join(RES, "per_material_metrics.csv")
    if os.path.exists(mat_metrics_path):
        mat_df = pd.read_csv(mat_metrics_path)
        if not mat_df.empty and len(mat_df) >= 2:
            mat_df = mat_df.sort_values("R2_log", ascending=True)
            fig, ax = plt.subplots(figsize=(7.2, max(3.5, len(mat_df) * 0.55)))
            colours = [WONG["green"] if r >= 0.7 else WONG["orange"] if r >= 0.4
                       else WONG["vermillion"] for r in mat_df["R2_log"]]
            ax.barh(mat_df["host_material"], mat_df["R2_log"],
                    color=colours, edgecolor="k", linewidth=0.4, alpha=0.85)
            ax.axvline(0, color="k", lw=0.8)
            ax.axvline(0.7, color="gray", linestyle=":", lw=0.8, alpha=0.7)
            ax.set_xlabel("R² (log scale, test set)")
            ax.set_title("Per-material prediction performance")
            panel_label(ax, "a")
            save_fig(fig, "fig5_per_material_performance", FIG_DIR)

    # ════════════════════════════════════════════════════════════════════════
    # FIG 6 — Discovery Top-20
    # ════════════════════════════════════════════════════════════════════════
    print("Generating Fig 6 — Discovery candidates...")
    disc_path = os.path.join(RES, "discovery_candidates.csv")
    if os.path.exists(disc_path):
        disc_df = pd.read_csv(disc_path)
        top20 = disc_df.nlargest(20, "ucb_log").copy()
        top20["label"] = top20["host_material"] + "/" + top20["co_catalyst"].astype(str)
        top20["pred_her"] = np.expm1(top20["pred_median_log"])
        top20["ucb_her"]  = np.expm1(top20["ucb_log"])
        top20["p05_her"]  = np.expm1(top20["pred_p05_log"])
        top20 = top20.sort_values("pred_her", ascending=True)

        n = len(top20)
        cmap = matplotlib.cm.plasma(np.linspace(0.15, 0.85, n))
        fig, ax = plt.subplots(figsize=(7.2, max(4.0, n * 0.38)))
        ax.barh(top20["label"], top20["pred_her"],
                xerr=[top20["pred_her"] - top20["p05_her"],
                      top20["ucb_her"] - top20["pred_her"]],
                color=cmap, alpha=0.85, capsize=3, edgecolor="k", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Predicted HER  [µmol g⁻¹ h⁻¹]")
        ax.set_title("Top-20 novel catalyst candidates (UCB ranking)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        panel_label(ax, "a")
        save_fig(fig, "fig6_discovery_candidates", FIG_DIR)

    # ════════════════════════════════════════════════════════════════════════
    # FIG 7 — Calibration Curve (if available)
    # ════════════════════════════════════════════════════════════════════════
    uq_path = os.path.join(RES, "uncertainty_summary.json")
    if os.path.exists(uq_path):
        import shutil
        src = os.path.join(RES, "calibration_curve.png")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIG_DIR, "fig7_calibration_curve.png"))
            print("  Copied calibration_curve.png -> fig7_calibration_curve.png")

    # ════════════════════════════════════════════════════════════════════════
    # FIG 8 — Ablation Study (regenerate from scratch)
    # ════════════════════════════════════════════════════════════════════════
    abl_path = os.path.join(RES, "ablation_results.csv")
    if os.path.exists(abl_path):
        df_abl = pd.read_csv(abl_path)
        if not df_abl.empty and "delta_r2" in df_abl.columns:
            # Check format — new style (model column) vs old style (ablation column)
            if "model" in df_abl.columns:
                baseline_mask = df_abl["model"].str.contains("baseline", case=False)
                df_plot = df_abl[~baseline_mask].copy()
                plot_models = df_plot["model"].tolist()
                deltas = df_plot["delta_r2"].values
            elif "ablation" in df_abl.columns:
                baseline_mask = df_abl["ablation"].str.contains("baseline", case=False)
                df_plot = df_abl[~baseline_mask].copy()
                plot_models = df_plot["ablation"].tolist()
                deltas = df_plot["delta_r2"].values
            else:
                plot_models = []
                deltas = []

            if len(plot_models) > 0:
                colours = []
                for d in deltas:
                    if d < -0.05:
                        colours.append("#d62728")
                    elif d < -0.01:
                        colours.append("#ff7f0e")
                    elif d > 0.01:
                        colours.append("#2ca02c")
                    else:
                        colours.append("#7f7f7f")

                fig, ax = plt.subplots(figsize=(6.5, max(3.0, len(plot_models) * 0.55)))
                short_names = [m.replace("Model ", "").replace(":", "\n") if len(m) > 35
                              else m for m in plot_models]
                bars = ax.barh(short_names, deltas, color=colours, alpha=0.85,
                               edgecolor="k", linewidth=0.4)
                ax.axvline(0, color="k", lw=1.0)
                ax.set_xlabel("Delta LOMO-CV R² vs. baseline", fontsize=10)
                ax.set_title("Ablation study: feature group contribution",
                             fontsize=11, fontweight="bold")
                ax.grid(True, axis="x", linestyle="--", alpha=0.35)
                for bar, d_val in zip(bars, deltas):
                    offset = 0.003
                    if d_val >= 0:
                        ax.text(d_val + offset, bar.get_y() + bar.get_height() / 2,
                                f"{d_val:+.4f}", va="center", ha="left", fontsize=8)
                    else:
                        ax.text(d_val - offset, bar.get_y() + bar.get_height() / 2,
                                f"{d_val:+.4f}", va="center", ha="right", fontsize=8)
                panel_label(ax, "a")
                plt.tight_layout()
                save_fig(fig, "fig8_ablation", FIG_DIR)
            else:
                # Fallback: copy pre-generated plot
                import shutil
                for ext in ["png", "pdf", "svg"]:
                    src = os.path.join(RES, f"ablation_plot.{ext}")
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(FIG_DIR, f"fig8_ablation.{ext}"))
                        print(f"  Copied {src} -> fig8_ablation.{ext}")
                        break

    print(f"\nPhase 11 complete. All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
