"""
dashboard.py
Interactive results dashboard for the glycerol photocatalyst
HER prediction project.

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import joblib

st.set_page_config(
    page_title="Photocatalyst HER Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROC_DIR    = "data/processed"
RESULTS_DIR = "data/results"
MODELS_DIR  = "models"
FIGS_DIR    = os.path.join(RESULTS_DIR, "figures")

PALETTE = {
    "TiO2":    "#1D9E75",
    "BiVO4":   "#378ADD",
    "ZnO":     "#BA7517",
    "Fe2O3":   "#D85A30",
    "SrTiO3":  "#8B5CF6",
    "CdS":     "#EC4899",
    "g-C3N4":  "#6B7280",
    "CeO2":    "#F59E0B",
    "WO3":     "#0EA5E9",
    "other":   "#9CA3AF",
}

# ── HELPERS ────────────────────────────────────────────────────────

@st.cache_data
def load_training_results():
    path = os.path.join(RESULTS_DIR, "training_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_clean_data():
    path = os.path.join(PROC_DIR, "df_clean.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)

@st.cache_data
def load_candidates():
    path = os.path.join(RESULTS_DIR, "discovery_candidates.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_conformal():
    path = os.path.join(RESULTS_DIR, "conformal_summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_conformal_intervals():
    path = os.path.join(RESULTS_DIR, "conformal_intervals.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_ad_summary():
    path = os.path.join(RESULTS_DIR, "ad_summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def material_color(name):
    for key in PALETTE:
        if key.lower() in str(name).lower():
            return PALETTE[key]
    return PALETTE["other"]

# ── SIDEBAR ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚗️ Photocatalyst HER")
    st.markdown("**ML-guided screening for glycerol photoreforming**")
    st.divider()

    metrics = load_training_results()
    if metrics:
        best = "LightGBM" if "LightGBM" in metrics else list(metrics.keys())[0]
        m = metrics[best]
        st.metric("Best model", best)
        st.metric("LOMO-CV R²",
                  f"{m.get('LOMO_CV_R2_mean', m.get('CV_R2_mean', 0)):.4f}",
                  f"±{m.get('LOMO_CV_R2_std', m.get('CV_R2_std', 0)):.4f}")
        st.metric("Test R² (log)", f"{m.get('Test_R2_log', 0):.4f}")
        st.metric("Spearman ρ",
                  f"{m.get('Spearman_rho_log', m.get('Spearman_rho', 0)):.4f}")

    conf = load_conformal()
    if conf:
        cov = conf.get("empirical_coverage", 0)
        color = "normal" if cov >= 0.90 else "inverse"
        st.metric("Conformal coverage",
                  f"{cov*100:.1f}%",
                  "✓ meets 90% target" if cov >= 0.90 else "✗ below target",
                  delta_color=color)

    st.divider()
    st.caption("Repo: RaunakOP-web/Photocatalyst-Trial-")

# ── TABS ───────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset",
    "🤖 Model performance",
    "🔍 Feature importance",
    "🧪 Virtual screening",
    "🎯 Predict new catalyst",
    "📄 Publication figures",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════

with tab1:
    st.header("Dataset overview")
    df = load_clean_data()

    if df.empty:
        st.warning("df_clean.csv not found in data/processed/")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total experiments", f"{len(df):,}")
    c2.metric("Semiconductors", df["host_material"].nunique()
              if "host_material" in df.columns else "—")
    c3.metric("HER range",
              f"{df['HER_std_umol_g_h'].min():.0f}–"
              f"{df['HER_std_umol_g_h'].max():,.0f} µmol/g/h"
              if "HER_std_umol_g_h" in df.columns else "—")
    c4.metric("Median HER",
              f"{df['HER_std_umol_g_h'].median():,.0f} µmol/g/h"
              if "HER_std_umol_g_h" in df.columns else "—")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Experiments by semiconductor")
        if "host_material" in df.columns:
            counts = df["host_material"].value_counts().reset_index()
            counts.columns = ["material", "count"]
            counts["color"] = counts["material"].apply(material_color)
            fig = px.bar(counts, x="count", y="material",
                         orientation="h", color="material",
                         color_discrete_map={r.material: r.color
                                             for _, r in counts.iterrows()},
                         labels={"count": "Experiments", "material": ""},
                         height=420)
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("HER distribution (log scale)")
        if "HER_std_umol_g_h" in df.columns:
            fig = px.histogram(
                df, x="HER_std_umol_g_h", nbins=50,
                log_x=True,
                labels={"HER_std_umol_g_h": "HER (µmol/g/h)", "count": "Experiments"},
                color_discrete_sequence=["#1D9E75"], height=420
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("HER by semiconductor — distribution")
    if "host_material" in df.columns and "HER_std_umol_g_h" in df.columns:
        top_mats = df["host_material"].value_counts().head(10).index.tolist()
        df_top = df[df["host_material"].isin(top_mats)].copy()
        fig = px.box(df_top, x="host_material", y="HER_std_umol_g_h",
                     log_y=True, color="host_material",
                     color_discrete_map={m: material_color(m) for m in top_mats},
                     labels={"host_material": "Semiconductor",
                             "HER_std_umol_g_h": "HER (µmol/g/h, log scale)"},
                     height=380)
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Co-catalyst loading vs HER")
    if all(c in df.columns for c in
           ["co_catalyst_wt_pct","HER_std_umol_g_h","co_catalyst"]):
        df_cc = df[df["co_catalyst_wt_pct"].notna() &
                   df["HER_std_umol_g_h"].notna()].copy()
        top_cc = df_cc["co_catalyst"].value_counts().head(8).index.tolist()
        df_cc = df_cc[df_cc["co_catalyst"].isin(top_cc)]
        fig = px.scatter(df_cc,
                         x="co_catalyst_wt_pct",
                         y="HER_std_umol_g_h",
                         color="co_catalyst",
                         log_y=True,
                         trendline="lowess",
                         labels={"co_catalyst_wt_pct": "Co-catalyst wt%",
                                 "HER_std_umol_g_h": "HER (µmol/g/h, log)",
                                 "co_catalyst": "Co-catalyst"},
                         height=420)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════

with tab2:
    st.header("Model performance")
    metrics = load_training_results()
    conf    = load_conformal()
    ci_df   = load_conformal_intervals()

    if not metrics:
        st.warning("training_results.json not found.")
    else:
        model_names = list(metrics.keys())
        lomo_means  = [metrics[m].get("LOMO_CV_R2_mean",
                        metrics[m].get("CV_R2_mean", 0)) for m in model_names]
        lomo_stds   = [metrics[m].get("LOMO_CV_R2_std",
                        metrics[m].get("CV_R2_std", 0))  for m in model_names]
        test_r2_log = [metrics[m].get("Test_R2_log", 0)  for m in model_names]
        test_r2_ori = [metrics[m].get("Test_R2_original",
                        metrics[m].get("R2_original", 0)) for m in model_names]
        mae_vals    = [metrics[m].get("Test_MAE_umol_g_h",
                        metrics[m].get("MAE_umol_g_h", 0)) for m in model_names]
        spearman    = [metrics[m].get("Spearman_rho_log",
                        metrics[m].get("Spearman_rho", 0)) for m in model_names]

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("LOMO-CV R² by model")
            fig = go.Figure()
            colors_bar = ["#1D9E75" if n == "LightGBM"
                          else "#378ADD" if n == "XGBoost"
                          else "#9CA3AF" for n in model_names]
            fig.add_trace(go.Bar(
                x=model_names, y=lomo_means,
                error_y=dict(type="data", array=lomo_stds, visible=True),
                marker_color=colors_bar,
                text=[f"{v:.4f}" for v in lomo_means],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis=dict(range=[0, 1.05], title="LOMO-CV R²"),
                plot_bgcolor="rgba(0,0,0,0)", height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("All metrics comparison")
            metrics_df = pd.DataFrame({
                "Model":         model_names,
                "LOMO-CV R²":    [f"{v:.4f}" for v in lomo_means],
                "Test R² (log)": [f"{v:.4f}" for v in test_r2_log],
                "Test R² (orig)":[f"{v:.4f}" for v in test_r2_ori],
                "MAE (µmol/g/h)":[f"{v:,.0f}" for v in mae_vals],
                "Spearman ρ":    [f"{v:.4f}" for v in spearman],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.divider()

        if not ci_df.empty:
            st.subheader("Actual vs predicted — test set (log scale)")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(ci_df,
                                 x="y_true_log", y="y_pred_log",
                                 color="covered",
                                 color_discrete_map={True: "#1D9E75",
                                                     False: "#D85A30"},
                                 labels={"y_true_log":  "Actual log(HER+1)",
                                         "y_pred_log":  "Predicted log(HER+1)",
                                         "covered":     "Within 90% CI"},
                                 opacity=0.7, height=380)
                mn = min(ci_df["y_true_log"].min(), ci_df["y_pred_log"].min())
                mx = max(ci_df["y_true_log"].max(), ci_df["y_pred_log"].max())
                fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                              line=dict(color="red", dash="dash", width=1.5))
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Conformal interval analysis")
                if conf:
                    cov = conf.get("empirical_coverage", 0)
                    st.metric("Empirical coverage",
                              f"{cov*100:.1f}%",
                              f"Target: 90%")
                    st.metric("Mean interval width (log)",
                              f"{conf.get('mean_width_log', 0):.3f}")
                    st.metric("Mean interval width (HER)",
                              f"{conf.get('mean_width_her_umol', 0):,.0f} µmol/g/h")
                    st.metric("Calibration samples",
                              f"{conf.get('n_calibration', 0)}")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=cov * 100,
                        title={"text": "Coverage (%)"},
                        gauge={
                            "axis":  {"range": [0, 100]},
                            "bar":   {"color": "#1D9E75"},
                            "steps": [
                                {"range": [0, 90],  "color": "#FEF3C7"},
                                {"range": [90, 100],"color": "#D1FAE5"},
                            ],
                            "threshold": {
                                "line":  {"color": "#D85A30", "width": 3},
                                "thickness": 0.75, "value": 90
                            },
                        },
                        number={"suffix": "%", "font": {"size": 32}},
                    ))
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════

with tab3:
    st.header("Feature importance — SHAP analysis")

    shap_png = os.path.join(FIGS_DIR, "fig2_shap_importance.png")
    bee_png  = os.path.join(FIGS_DIR, "fig3_shap_beeswarm.png")
    dep_png  = os.path.join(FIGS_DIR, "fig4_shap_dependence.png")
    pdp_png  = os.path.join(FIGS_DIR, "fig5_partial_dependence.png")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Global feature importance (|SHAP|)")
        if os.path.exists(shap_png):
            st.image(shap_png, use_column_width=True)
        else:
            st.info("Run manuscript_figures.py to generate this figure.")

    with c2:
        st.subheader("Feature effect directions (beeswarm)")
        if os.path.exists(bee_png):
            st.image(bee_png, use_column_width=True)
        else:
            st.info("Run manuscript_figures.py to generate this figure.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("SHAP dependence plots")
        if os.path.exists(dep_png):
            st.image(dep_png, use_column_width=True)
        else:
            st.info("Run manuscript_figures.py to generate this figure.")

    with c2:
        st.subheader("Partial dependence plots")
        if os.path.exists(pdp_png):
            st.image(pdp_png, use_column_width=True)
        else:
            st.info("Run manuscript_figures.py to generate this figure.")

    st.subheader("Physical feature interpretation")
    interpretation = {
        "cocat_work_function":      "Co-catalyst work function (eV) — controls Schottky barrier height with semiconductor. Higher = better electron trapping.",
        "semi_bandgap_eV":          "Semiconductor bandgap (eV) — determines light absorption range. Lower = more visible light absorbed.",
        "cocat_d_band_center":      "Co-catalyst d-band center (eV) — controls H adsorption strength via Sabatier principle. Optimal ≈ −1.5 eV.",
        "co_catalyst_wt_pct":       "Co-catalyst loading (wt%) — trade-off between active sites and light blocking. Typically optimal at 0.5–2 wt%.",
        "glycerol_concentration_std":"Glycerol concentration (mol/L) — sacrificial donor availability. Higher = more holes quenched but mass transfer limited.",
        "semi_electron_affinity_eV":"Semiconductor electron affinity (eV) — band alignment with co-catalyst. Drives charge separation direction.",
        "cocat_electronegativity":  "Co-catalyst electronegativity — influences metal–support interaction and CO tolerance.",
        "wavelength_cutoff_nm":     "Light filter cutoff (nm) — determines UV vs visible irradiation regime.",
    }
    for feat, desc in interpretation.items():
        with st.expander(f"**{feat}**"):
            st.write(desc)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — VIRTUAL SCREENING
# ══════════════════════════════════════════════════════════════════

with tab4:
    st.header("Virtual screening results")
    disc = load_candidates()
    ad   = load_ad_summary()

    if disc.empty:
        st.warning("discovery_candidates.csv not found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total screened", f"{len(disc):,}")
        c2.metric("Within AD", f"{disc['within_ad'].sum():,}"
                  if "within_ad" in disc.columns else "—")
        c3.metric("Top predicted HER",
                  f"{disc['pred_her_umol_g_h'].max():,.0f} µmol/g/h"
                  if "pred_her_umol_g_h" in disc.columns else "—")
        top_row = disc.sort_values("pred_her_umol_g_h",
                                   ascending=False).iloc[0]
        c4.metric("Top candidate",
                  f"{top_row.get('host_material','?')}/"
                  f"{top_row.get('co_catalyst','?')}")

        st.divider()

        st.subheader("Filter and explore candidates")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        semis = sorted(disc["host_material"].dropna().unique().tolist()) \
                if "host_material" in disc.columns else []
        cats  = sorted(disc["co_catalyst"].dropna().unique().tolist()) \
                if "co_catalyst"  in disc.columns else []
        lights= sorted(disc["light_type"].dropna().unique().tolist()) \
                if "light_type"   in disc.columns else []

        sel_semi  = col_f1.multiselect("Semiconductor", semis,
                                        default=semis[:5] if semis else [])
        sel_cat   = col_f2.multiselect("Co-catalyst",   cats,
                                        default=cats[:6]  if cats  else [])
        sel_light = col_f3.multiselect("Light type",    lights,
                                        default=lights)
        only_ad   = col_f4.checkbox("Only within-AD candidates", value=True)

        df_f = disc.copy()
        if sel_semi  and "host_material" in df_f.columns:
            df_f = df_f[df_f["host_material"].isin(sel_semi)]
        if sel_cat   and "co_catalyst"   in df_f.columns:
            df_f = df_f[df_f["co_catalyst"].isin(sel_cat)]
        if sel_light and "light_type"    in df_f.columns:
            df_f = df_f[df_f["light_type"].isin(sel_light)]
        if only_ad   and "within_ad"     in df_f.columns:
            df_f = df_f[df_f["within_ad"] == True]

        df_top = df_f.sort_values("pred_her_umol_g_h",
                                   ascending=False).head(30)

        st.subheader(f"Top 30 candidates (filtered, {len(df_f):,} total)")
        fig = px.bar(
            df_top.head(20),
            x="pred_her_umol_g_h",
            y=df_top.head(20).apply(
                lambda r: f"{r.get('host_material','?')}/"
                          f"{r.get('co_catalyst','?')} "
                          f"({r.get('co_catalyst_wt_pct','?')}wt%)", axis=1),
            orientation="h",
            color="host_material" if "host_material" in df_top.columns else None,
            error_x=df_top.head(20).apply(
                lambda r: r.get("pred_upper_her", r.get("pred_her_umol_g_h", 0))
                          - r.get("pred_her_umol_g_h", 0), axis=1)
                if "pred_upper_her" in df_top.columns else None,
            labels={"pred_her_umol_g_h": "Predicted HER (µmol/g/h)", "y": ""},
            height=540,
        )
        fig.update_layout(showlegend=True, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predicted HER vs AD score — all filtered candidates")
        fig2 = px.scatter(
            df_f.sample(min(2000, len(df_f))),
            x="ad_score" if "ad_score" in df_f.columns else df_f.columns[0],
            y="pred_her_umol_g_h",
            color="host_material" if "host_material" in df_f.columns else None,
            log_y=True,
            hover_data=["co_catalyst", "co_catalyst_wt_pct"]
                       if "co_catalyst" in df_f.columns else None,
            labels={"ad_score": "AD score (lower = safer prediction)",
                    "pred_her_umol_g_h": "Predicted HER (µmol/g/h, log)",
                    "host_material": "Semiconductor"},
            height=420, opacity=0.6,
        )
        if ad:
            fig2.add_vline(x=ad.get("ad_threshold", 130),
                           line_dash="dash", line_color="red",
                           annotation_text="AD threshold",
                           annotation_position="top right")
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Full filtered table (download below)")
        show_cols = [c for c in ["host_material","co_catalyst",
                                  "co_catalyst_wt_pct",
                                  "glycerol_concentration_std",
                                  "light_type","pred_her_umol_g_h",
                                  "ad_score","ad_label","within_ad"]
                     if c in df_top.columns]
        st.dataframe(df_top[show_cols], use_container_width=True,
                     hide_index=True)
        st.download_button(
            "⬇ Download filtered candidates (CSV)",
            df_f.to_csv(index=False).encode(),
            file_name="filtered_candidates.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT NEW CATALYST
# ══════════════════════════════════════════════════════════════════

with tab5:
    st.header("Predict HER for a new catalyst")
    st.info("Enter your catalyst conditions and get an instant HER prediction "
            "with a 90% conformal confidence interval.")

    try:
        from src.material_features import (
            SEMICONDUCTOR_PROPS, COCATALYST_PROPS,
            add_physical_features
        )
        semi_options  = sorted(SEMICONDUCTOR_PROPS.keys())
        cocat_options = sorted(COCATALYST_PROPS.keys())
        model_loaded  = True
    except ImportError:
        st.warning("Could not import material_features. "
                   "Run from the repo root directory.")
        model_loaded = False

    if model_loaded:
        c1, c2, c3 = st.columns(3)
        semi  = c1.selectbox("Semiconductor",   semi_options,
                              index=semi_options.index("TiO2")
                              if "TiO2" in semi_options else 0)
        cocat = c2.selectbox("Co-catalyst",     cocat_options,
                              index=cocat_options.index("Pt")
                              if "Pt"   in cocat_options else 0)
        wt_pct= c3.slider("Co-catalyst wt%", 0.1, 5.0, 1.0, 0.1)

        c4, c5, c6 = st.columns(3)
        glyc  = c4.slider("Glycerol concentration (mol/L)", 0.1, 5.0, 1.37, 0.05)
        load  = c5.slider("Catalyst loading (mg)", 10, 500, 50, 10)
        vol   = c6.slider("Reaction volume (mL)", 10, 500, 100, 10)

        c7, c8 = st.columns(2)
        light = c7.selectbox("Light type", ["UV", "Visible"])
        wl    = c8.slider("Wavelength cutoff (nm)", 300, 500, 420, 5)

        if st.button("🔮 Predict HER", type="primary"):
            try:
                feature_list = joblib.load(
                    os.path.join(PROC_DIR, "feature_list.joblib"))
                conf_pkg     = joblib.load(
                    os.path.join(MODELS_DIR, "conformal_model.joblib"))
                model  = conf_pkg["model"]
                q_hat  = conf_pkg["q_hat"]

                input_row = {
                    "host_material":            semi,
                    "co_catalyst":              cocat,
                    "co_catalyst_wt_pct":       wt_pct,
                    "semiconductor_2":          "unknown",
                    "glycerol_concentration_std": glyc,
                    "catalyst_loading_mg":      load,
                    "reaction_volume_mL":       vol,
                    "temperature_C":            25,
                    "pH":                       None,
                    "light_power_W":            300,
                    "wavelength_cutoff_nm":     wl,
                    "is_xe_lamp":               1 if light == "Visible" else 0,
                    "is_hg_lamp":               1 if light == "UV" else 0,
                    "is_led":                   0,
                    "is_uv":                    1 if light == "UV" else 0,
                    "is_visible_light":         1 if light == "Visible" else 0,
                    "is_solar_simulator":       0,
                }
                input_df = pd.DataFrame([input_row])
                input_df = add_physical_features(input_df)

                X = input_df[[f for f in feature_list
                               if f in input_df.columns]].copy()
                missing_f = [f for f in feature_list if f not in input_df.columns]
                for mf in missing_f:
                    X[mf] = 0.0
                X = X[feature_list]

                log_pred = model.predict(X)[0]
                her_pred = np.expm1(log_pred)
                her_lo   = np.expm1(log_pred - q_hat)
                her_hi   = np.expm1(log_pred + q_hat)

                st.success(f"**Predicted HER: {her_pred:,.0f} µmol/g/h**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted HER", f"{her_pred:,.0f} µmol/g/h")
                m2.metric("90% CI lower",  f"{her_lo:,.0f} µmol/g/h")
                m3.metric("90% CI upper",  f"{her_hi:,.0f} µmol/g/h")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"{semi}/{cocat}"],
                    y=[her_pred],
                    error_y=dict(type="data",
                                 array=[her_hi - her_pred],
                                 arrayminus=[her_pred - her_lo],
                                 visible=True),
                    marker_color="#1D9E75",
                    width=0.3,
                ))
                fig.update_layout(
                    yaxis_title="Predicted HER (µmol/g/h)",
                    height=320,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure all pipeline scripts have been run first.")


# ══════════════════════════════════════════════════════════════════
# TAB 6 — PUBLICATION FIGURES
# ══════════════════════════════════════════════════════════════════

with tab6:
    st.header("Publication figures")
    st.caption("All figures are saved as PNG (300 DPI), PDF, and SVG "
               "in data/results/figures/")

    if not os.path.exists(FIGS_DIR):
        st.warning("Figures directory not found. "
                   "Run src/manuscript_figures.py first.")
    else:
        png_files = sorted(
            [f for f in os.listdir(FIGS_DIR) if f.endswith(".png")]
        )
        if not png_files:
            st.warning("No PNG figures found.")
        else:
            cols_per_row = 2
            for i in range(0, len(png_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(png_files):
                        fname = png_files[i + j]
                        fpath = os.path.join(FIGS_DIR, fname)
                        fig_num = fname.split("_")[0] if "_" in fname else fname
                        col.subheader(fig_num.replace("fig","Figure "))
                        col.image(fpath, use_column_width=True,
                                  caption=fname)
                        with open(fpath, "rb") as f:
                            col.download_button(
                                f"⬇ Download {fname}",
                                f.read(),
                                file_name=fname,
                                mime="image/png",
                                key=f"dl_{fname}",
                            )
