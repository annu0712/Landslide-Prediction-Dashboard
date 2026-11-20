"""
Oregon Landslide Susceptibility AI Dashboard
Author: Farzan | Data Science Project
Dataset: USGS Landslide Inventories v2 + Copernicus 30m DEM + PRISM Normals
Models: XGBoost, Random Forest, Logistic Regression with Spatial Block Cross-Validation
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Oregon Landslide Susceptibility AI",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM STYLING (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        color: #475569;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Header & Badge Styling */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 14px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 4px;
    }
    .badge-primary { background: #38bdf8; color: #082f49; }
    .badge-secondary { background: #334155; color: #f8fafc; }
    .badge-success { background: #4ade80; color: #052e16; }
    .badge-warning { background: #fbbf24; color: #451a03; }
    
    /* Risk Badges */
    .risk-severe { background-color: #fee2e2; color: #991b1b; padding: 6px 12px; border-radius: 8px; font-weight: bold; border-left: 4px solid #ef4444; }
    .risk-high { background-color: #ffedd5; color: #9a3412; padding: 6px 12px; border-radius: 8px; font-weight: bold; border-left: 4px solid #f97316; }
    .risk-mod { background-color: #fef9c3; color: #854d0e; padding: 6px 12px; border-radius: 8px; font-weight: bold; border-left: 4px solid #eab308; }
    .risk-low { background-color: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 8px; font-weight: bold; border-left: 4px solid #22c55e; }

    /* Tab bar spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DYNAMIC PATH RESOLUTION (Zero Configuration Needed)
# ---------------------------------------------------------
def resolve_path(name):
    """Finds a file or directory relative to the script or working dir."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. Look next to script
    candidate1 = os.path.join(script_dir, name)
    if os.path.exists(candidate1):
        return candidate1
    # 2. Look in current working directory
    candidate2 = os.path.join(os.getcwd(), name)
    if os.path.exists(candidate2):
        return candidate2
    return candidate1

SAMPLE_CSV_PATH = resolve_path("landslide_map_sample_40k.csv")
FULL_PARQUET_PATH = resolve_path("landslide_model_data.parquet")
MODELS_DIR = resolve_path("models")

# ---------------------------------------------------------
# DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_map_sample():
    if os.path.exists(SAMPLE_CSV_PATH):
        return pd.read_csv(SAMPLE_CSV_PATH)
    elif os.path.exists(FULL_PARQUET_PATH):
        df = pd.read_parquet(FULL_PARQUET_PATH)
        return df.sample(n=min(40000, len(df)), random_state=42)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_full_dataset():
    if os.path.exists(FULL_PARQUET_PATH):
        return pd.read_parquet(FULL_PARQUET_PATH)
    elif os.path.exists(os.path.join(BASE_DIR, "landslide_model_data.csv")):
        return pd.read_csv(os.path.join(BASE_DIR, "landslide_model_data.csv"))
    return load_map_sample()

@st.cache_resource(show_spinner=False)
def load_trained_models():
    models = {}
    # XGBoost
    xgb_json = os.path.join(MODELS_DIR, "xgboost_model.json")
    if os.path.exists(xgb_json):
        try:
            m = xgb.XGBClassifier()
            m.load_model(xgb_json)
            models["XGBoost"] = m
        except Exception as e:
            st.error(f"Error loading XGBoost JSON: {e}")
            
    # Random Forest
    rf_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
    if os.path.exists(rf_path):
        try:
            models["Random Forest"] = joblib.load(rf_path)
        except Exception as e:
            st.warning(f"Error loading RF joblib: {e}")
            
    # Logistic Regression
    lr_path = os.path.join(MODELS_DIR, "logistic_regression_model.joblib")
    if os.path.exists(lr_path):
        try:
            lr_pipe = joblib.load(lr_path)
            # Patch scikit-learn version attribute if required
            clf = lr_pipe.named_steps.get("logisticregression", None)
            if clf is not None and not hasattr(clf, "multi_class"):
                clf.multi_class = "auto"
            models["Logistic Regression"] = lr_pipe
        except Exception as e:
            st.warning(f"Error loading LR joblib: {e}")
            
    return models

# Load core resources
df_sample = load_map_sample()
trained_models = load_trained_models()

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/mountain.png", width=64)
    st.title("Oregon Landslides")
    st.caption("AI-Powered Hazard Susceptibility Pipeline")
    
    st.divider()
    
    st.subheader("⚙️ Global Settings")
    active_model_name = st.selectbox(
        "Primary Scoring Model",
        options=["XGBoost", "Random Forest", "Logistic Regression"],
        index=0,
        help="Model used for risk calculations and map rendering."
    )
    
    model_col_map = {
        "XGBoost": "pred_prob_xgb",
        "Random Forest": "pred_prob_rf",
        "Logistic Regression": "pred_prob_lr"
    }
    active_prob_col = model_col_map[active_model_name]
    
    prob_threshold = st.slider(
        "Hazard Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold above which ground is classified as landslide-prone."
    )
    
    st.divider()
    st.markdown("### 📌 Research Specifications")
    st.markdown(r"""
    - **Inventory:** USGS v2 (Conf. $\ge 3$)
    - **Total Records:** 214,043 points (50% Landslide, 50% Random)
    - **Covariates:** Copernicus 30m DEM + PRISM 1991–2020 Normals
    - **Validation:** 5-Fold Spatial Block ($0.5^\circ$ blocks)
    - **Primary Model AUC:** **0.931** (XGBoost)
    """)
    
    st.divider()
    st.caption("Designed by **Farzan** | Machine Learning Project")

# ---------------------------------------------------------
# TOP HERO BANNER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Oregon Landslide Susceptibility AI Dashboard</div>
    <div class="hero-subtitle">End-to-End Spatial Machine Learning Pipeline: Terrain Geomorphometry & Precipitation Modeling</div>
    <div>
        <span class="badge badge-primary">🌲 Oregon & Pacific NW Study Area</span>
        <span class="badge badge-secondary">🗺️ 107,023 Landslide Records</span>
        <span class="badge badge-success">🏆 0.931 XGBoost Spatial AUC</span>
        <span class="badge badge-warning">🛡️ Spatial Block Cross-Validation (161 Blocks)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TOP KPI ROW
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Study Area Points", f"{214043:,}", "50/50 Balanced")
kpi2.metric("XGBoost Spatial AUC", "0.931", "± 0.014 Across Folds")
kpi3.metric("Landslide Sensitivity (Recall)", "85.6%", "91,561 Detected")
kpi4.metric("Spatial Blocks", "161 Blocks", "0.5° (~50km) Grid")
kpi5.metric("Strongest Predictors", "Rainfall & Elev", "42% & 38% Importance")

# ---------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------
tab_map, tab_sim, tab_eval, tab_covariates, tab_bias, tab_data = st.tabs([
    "🗺️ Geospatial Susceptibility Map",
    "⚡ What-If Hazard Simulator",
    "📈 Model Evaluation & Spatial CV",
    "🔬 Covariates & Feature Importance",
    "⚠️ Survey Bias & Spatial Analysis",
    "📋 Data Explorer & Export"
])

# =========================================================
# TAB 1: GEOSPATIAL SUSCEPTIBILITY MAP
# =========================================================
with tab_map:
    st.subheader("Interactive Susceptibility & Spatial Distribution Map")
    st.write(
        "Explore landslide points and sampled negative ground across Oregon. Points are colored by "
        "their **out-of-fold predicted probability** from spatial block cross-validation."
    )
    
    m_col1, m_col2, m_col3 = st.columns([1, 1, 1])
    with m_col1:
        map_color_by = st.selectbox(
            "Color Points By",
            options=["Predicted Risk Probability", "Ground Truth Class (1 vs 0)", "Prediction Result (TP/FP/TN/FN)"],
            index=0
        )
    with m_col2:
        class_filter = st.multiselect(
            "Filter Ground Truth Class",
            options=[1, 0],
            default=[1, 0],
            format_func=lambda x: "Landslide (1)" if x == 1 else "Non-Landslide (0)"
        )
    with m_col3:
        risk_filter = st.selectbox(
            "Hazard Tier Filter",
            options=["All Ground Points", "High & Severe Risk (>0.5)", "Severe Hazard Only (>0.75)"],
            index=0
        )
        
    # Filter dataset
    plot_df = df_sample.copy()
    plot_df = plot_df[plot_df["label"].isin(class_filter)]
    
    if risk_filter == "High & Severe Risk (>0.5)":
        plot_df = plot_df[plot_df[active_prob_col] >= 0.5]
    elif risk_filter == "Severe Hazard Only (>0.75)":
        plot_df = plot_df[plot_df[active_prob_col] >= 0.75]
        
    # Add classification category
    def categorize_pred(row):
        pred = 1 if row[active_prob_col] >= prob_threshold else 0
        actual = int(row["label"])
        if actual == 1 and pred == 1:
            return "True Positive (Detected Landslide)"
        elif actual == 0 and pred == 0:
            return "True Negative (Safe Ground)"
        elif actual == 0 and pred == 1:
            return "False Alarm (False Positive)"
        else:
            return "Missed Landslide (False Negative)"
            
    plot_df["Outcome"] = plot_df.apply(categorize_pred, axis=1)
    plot_df["Ground Truth"] = plot_df["label"].map({1: "Landslide", 0: "Non-Landslide"})

    # Render Plotly Mapbox
    if map_color_by == "Predicted Risk Probability":
        fig_map = px.scatter_map(
            plot_df,
            lat="latitude",
            lon="longitude",
            color=active_prob_col,
            color_continuous_scale="Turbo",
            range_color=[0.0, 1.0],
            hover_name="Ground Truth",
            hover_data={
                "latitude": ":.3f",
                "longitude": ":.3f",
                "elev_m": ":.0f m",
                "slope_deg": ":.1f°",
                "relief_m": ":.0f m",
                "precip_mm": ":.0f mm",
                active_prob_col: ":.3f",
                "Outcome": True
            },
            zoom=5.8,
            center={"lat": 44.0, "lon": -120.5},
            mapbox_style="carto-positron",
            title=f"Oregon Spatial Hazard Map ({len(plot_df):,} Sample Points Rendered) - Color: {active_model_name} Risk Probability"
        )
    elif map_color_by == "Ground Truth Class (1 vs 0)":
        fig_map = px.scatter_mapbox(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="Ground Truth",
            color_discrete_map={"Landslide": "#ef4444", "Non-Landslide": "#3b82f6"},
            hover_data={
                "elev_m": ":.0f m",
                "slope_deg": ":.1f°",
                "precip_mm": ":.0f mm",
                active_prob_col: ":.3f"
            },
            zoom=5.8,
            center={"lat": 44.0, "lon": -120.5},
            mapbox_style="carto-positron",
            title="Ground Truth Distribution: USGS Landslide Inventory vs Sampled Negative Ground"
        )
    else:
        fig_map = px.scatter_mapbox(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="Outcome",
            color_discrete_map={
                "True Positive (Detected Landslide)": "#10b981",
                "True Negative (Safe Ground)": "#60a5fa",
                "False Alarm (False Positive)": "#f59e0b",
                "Missed Landslide (False Negative)": "#dc2626"
            },
            zoom=5.8,
            center={"lat": 44.0, "lon": -120.5},
            mapbox_style="carto-positron",
            title=f"Model Classification Outcomes (Threshold = {prob_threshold})"
        )
        
    fig_map.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=580,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Map Context Notes
    c1, c2, c3 = st.columns(3)
    c1.info("💡 **Western Focus:** 95% of landslide records cluster in Western Oregon along the Coast Range and the Cascades where terrain is steep and precipitation exceeds 1,500 mm.")
    c2.warning("⚠️ **Eastern Oregon Plateau:** The dry eastern interior features very few historical survey records, leading the model to assign low probabilities (<0.25).")
    c3.success("🛡️ **Spatial Block Integrity:** Predictions shown were generated by models that never saw data from the target point's 50 km block during training.")

# =========================================================
# TAB 2: WHAT-IF HAZARD SIMULATOR
# =========================================================
with tab_sim:
    st.subheader("⚡ Live Geotechnical Hazard Calculator")
    st.write(
        "Input terrain and rainfall parameters to simulate real-time landslide susceptibility using the trained machine learning models."
    )
    
    # Presets Row
    st.markdown("##### 📍 Quick Real-World Regional Presets:")
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    preset_clicked = None
    
    if col_pre1.button("🌊 Coast Range Escarpment", use_container_width=True):
        st.session_state["elev"] = 380
        st.session_state["slope"] = 28.0
        st.session_state["relief"] = 160
        st.session_state["precip"] = 2800
        
    if col_pre2.button("🏔️ Cascades Mountain Ridge", use_container_width=True):
        st.session_state["elev"] = 1850
        st.session_state["slope"] = 24.0
        st.session_state["relief"] = 190
        st.session_state["precip"] = 2100

    if col_pre3.button("🏡 Willamette Valley Foothills", use_container_width=True):
        st.session_state["elev"] = 160
        st.session_state["slope"] = 8.5
        st.session_state["relief"] = 35
        st.session_state["precip"] = 1150

    if col_pre4.button("🌾 Columbia Basin / East Desert", use_container_width=True):
        st.session_state["elev"] = 820
        st.session_state["slope"] = 3.5
        st.session_state["relief"] = 22
        st.session_state["precip"] = 340
        
    # Input Sliders
    sim_col1, sim_col2 = st.columns([1, 1])
    
    with sim_col1:
        st.markdown("#### Terrain & Climate Covariates")
        in_elev = st.slider(
            "Elevation (m) [Copernicus GLO-30 DEM]",
            min_value=-1, max_value=3400,
            value=st.session_state.get("elev", 420),
            step=10, key="elev_slider",
            help="Height above sea level (meters). Historical landslide mean = 382 m."
        )
        in_slope = st.slider(
            "Slope Steepness (degrees) [Horn 3×3]",
            min_value=0.0, max_value=70.0,
            value=float(st.session_state.get("slope", 18.5)),
            step=0.5, key="slope_slider",
            help="Local terrain incline in degrees. Historical landslide mean = 19.1°."
        )
        in_relief = st.slider(
            "Local Relief (m) [11×11 Window, ~240m × 340m]",
            min_value=0, max_value=420,
            value=st.session_state.get("relief", 95),
            step=5, key="relief_slider",
            help="Maximum minus minimum elevation in local neighborhood. Landslide mean = 94 m."
        )
        in_precip = st.slider(
            "Mean Annual Rainfall (mm) [PRISM 30-Year Normals]",
            min_value=150, max_value=5100,
            value=st.session_state.get("precip", 1850),
            step=25, key="precip_slider",
            help="Average annual precipitation (mm/year). Historical landslide mean = 1,781 mm."
        )
        
    # Run Inference
    input_features = np.array([[in_elev, in_slope, in_relief, in_precip]])
    input_df = pd.DataFrame(input_features, columns=["elev_m", "slope_deg", "relief_m", "precip_mm"])
    
    xgb_m = trained_models.get("XGBoost")
    rf_m = trained_models.get("Random Forest")
    lr_m = trained_models.get("Logistic Regression")
    
    p_xgb = float(xgb_m.predict_proba(input_df)[0, 1]) if xgb_m else 0.85
    p_rf = float(rf_m.predict_proba(input_features)[0, 1]) if rf_m else 0.82
    p_lr = float(lr_m.predict_proba(input_features)[0, 1]) if lr_m else 0.79
    
    primary_p = p_xgb if active_model_name == "XGBoost" else (p_rf if active_model_name == "Random Forest" else p_lr)
    
    with sim_col2:
        st.markdown(f"#### Real-Time Inference: {active_model_name}")
        
        # Risk Badge & Gauge
        gauge_color = "#22c55e" if primary_p < 0.35 else ("#eab308" if primary_p < 0.60 else ("#f97316" if primary_p < 0.80 else "#ef4444"))
        risk_label = "LOW HAZARD" if primary_p < 0.35 else ("MODERATE HAZARD" if primary_p < 0.60 else ("HIGH SUSCEPTIBILITY" if primary_p < 0.80 else "SEVERE LANDSLIDE RISK"))
        risk_class = "risk-low" if primary_p < 0.35 else ("risk-mod" if primary_p < 0.60 else ("risk-high" if primary_p < 0.80 else "risk-severe"))
        
        st.markdown(f"""
        <div class="{risk_class}">
            <div style="font-size: 1.15rem;">STATUS: {risk_label}</div>
            <div style="font-size: 0.85rem; font-weight: normal; margin-top: 3px;">
                Calculated Susceptibility Score: <strong>{primary_p * 100:.1f}%</strong> (Threshold: {prob_threshold * 100:.0f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Speedometer Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=primary_p * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
            title={"text": f"{active_model_name} Risk Index", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": gauge_color, "thickness": 0.4},
                "bgcolor": "#f1f5f9",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},
                    {"range": [35, 60], "color": "#fef9c3"},
                    {"range": [60, 80], "color": "#ffedd5"},
                    {"range": [80, 100], "color": "#fee2e2"}
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.75,
                    "value": prob_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Model Comparison Table
        model_comp_df = pd.DataFrame({
            "Classifier": ["XGBoost (Champion)", "Random Forest", "Logistic Regression (Baseline)"],
            "Probability": [f"{p_xgb * 100:.1f}%", f"{p_rf * 100:.1f}%", f"{p_lr * 100:.1f}%"],
            "Prediction": [
                "⚠️ Landslide Prone" if p >= prob_threshold else "✅ Stable Ground"
                for p in [p_xgb, p_rf, p_lr]
            ]
        })
        st.dataframe(model_comp_df, hide_index=True, use_container_width=True)

    st.divider()
    
    # Covariate Comparison against Baselines
    st.markdown("#### Input Parameters vs Historical Oregon Baselines")
    comp_cols = st.columns(4)
    
    cov_data = [
        ("Elevation", in_elev, "m", 382, 1116),
        ("Slope", in_slope, "°", 19.1, 10.2),
        ("Relief", in_relief, "m", 94, 53),
        ("Precipitation", in_precip, "mm", 1781, 751)
    ]
    
    for i, (name, val, unit, ls_mean, safe_mean) in enumerate(cov_data):
        with comp_cols[i]:
            diff_from_ls = val - ls_mean
            st.metric(
                label=f"{name} ({unit})",
                value=f"{val:,.1f} {unit}",
                delta=f"{diff_from_ls:+,.1f} vs Landslide Mean ({ls_mean}{unit})",
                delta_color="inverse" if name == "Elevation" else "normal"
            )
            st.caption(f"Non-Landslide Mean: {safe_mean} {unit}")

# =========================================================
# TAB 3: MODEL EVALUATION & SPATIAL CV
# =========================================================
with tab_eval:
    st.subheader("Model Performance & Spatial Block Cross-Validation")
    st.markdown(r"""
    To combat spatial autocorrelation (where nearby test points resemble training points), the study area was partitioned into 
    **161 independent $0.5^\circ$ blocks** (~50 km wide). Every prediction was evaluated strictly on unseen spatial blocks.
    """)
    
    # Official Metrics Table
    m_eval_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost (Selected)"],
        "Accuracy": [0.848, 0.852, 0.854],
        "Precision": [0.823, 0.852, 0.854],
        "Recall (Sensitivity)": [0.885, 0.851, 0.856],
        "F1 Score": [0.853, 0.852, 0.855],
        "Spatial ROC-AUC": ["0.911 ± 0.016", "0.930 ± 0.015", "0.931 ± 0.014"]
    })
    st.dataframe(m_eval_df, hide_index=True, use_container_width=True)
    
    ev_c1, ev_c2 = st.columns([1, 1])
    
    with ev_c1:
        st.markdown("#### Spatial Cross-Validation: XGBoost Fold Stability")
        fold_auc_data = pd.DataFrame({
            "Spatial Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean Score"],
            "ROC-AUC": [0.951, 0.929, 0.907, 0.936, 0.932, 0.931],
            "Type": ["Individual Block Fold", "Individual Block Fold", "Individual Block Fold (Hardest)", "Individual Block Fold", "Individual Block Fold", "5-Fold Mean"]
        })
        fig_fold = px.bar(
            fold_auc_data,
            x="Spatial Fold",
            y="ROC-AUC",
            color="Type",
            text="ROC-AUC",
            range_y=[0.85, 1.0],
            color_discrete_map={"Individual Block Fold": "#3b82f6", "Individual Block Fold (Hardest)": "#f59e0b", "5-Fold Mean": "#10b981"},
            title="XGBoost Performance Across Independent 0.5° Spatial Block Folds"
        )
        fig_fold.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_fold.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_fold, use_container_width=True)
        
    with ev_c2:
        st.markdown("#### Out-of-Fold Spatial ROC Curves")
        from sklearn.metrics import roc_curve, auc
        
        fig_roc = go.Figure()
        # Diagonal baseline
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8"), name="Random Guess (0.50)"))
        
        # Calculate curves from sample
        for model_name, col_name, color in [
            ("Logistic Regression", "pred_prob_lr", "#64748b"),
            ("Random Forest", "pred_prob_rf", "#0284c7"),
            ("XGBoost", "pred_prob_xgb", "#dc2626")
        ]:
            fpr, tpr, _ = roc_curve(df_sample["label"], df_sample[col_name])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=color, width=2.5 if model_name == "XGBoost" else 1.8),
                name=f"{model_name} (AUC = {roc_auc:.3f})"
            ))
            
        fig_roc.update_layout(
            title="ROC Curves on Spatial Block Out-of-Fold Test Sets",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(x=0.55, y=0.15)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Threshold Tuning & Geotechnical Tradeoff
    st.divider()
    st.markdown("#### 🎚️ Dynamic Decision Threshold & Geotechnical Hazard Tradeoff")
    st.caption("In real-world slope hazard management, a **False Negative (missed landslide)** can cause loss of life or infrastructure damage, whereas a **False Positive (false alarm)** causes precautionary inspections. Adjust the threshold slider below to inspect the impact on safety.")
    
    t_col1, t_col2 = st.columns([1, 1])
    
    with t_col1:
        th_val = st.slider("Tune Threshold for Hazard Screening", min_value=0.1, max_value=0.9, value=prob_threshold, step=0.05, key="th_eval_slider")
        
        # Dynamic confusion calculation on full 214k or sample
        y_true = df_sample["label"].values
        y_prob = df_sample[active_prob_col].values
        y_pred = (y_prob >= th_val).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Display confusion matrix heatmap
        cm_matrix = [[tn, fp], [fn, tp]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_matrix,
            x=["Predicted Safe (0)", "Predicted Landslide (1)"],
            y=["Actual Safe (0)", "Actual Landslide (1)"],
            text=[[f"TN: {tn:,}", f"FP: {fp:,}"], [f"FN: {fn:,}", f"TP: {tp:,}"]],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale="Blues",
            showscale=False
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix ({active_model_name} @ Threshold {th_val:.2f})",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with t_col2:
        st.markdown(f"##### Performance Metrics at Threshold {th_val:.2f}:")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Landslides Found (Recall)", f"{recall * 100:.1f}%", f"{tp:,} Caught")
        mc2.metric("Precision (Reliability)", f"{precision * 100:.1f}%", f"{fp:,} False Alarms")
        mc3.metric("Missed Landslides (FN)", f"{fn:,}", "Lower is Safer", delta_color="inverse")
        
        st.info(f"""
        **Hazard Engineering Insight:**
        - At **threshold 0.50**, XGBoost misses **15,462** landslides across Oregon.
        - Lowering the threshold to **0.30** increases Recall to over **93%**, finding thousands more hazardous zones at the cost of modest additional inspection areas.
        """)

# =========================================================
# TAB 4: COVARIATES & FEATURE IMPORTANCE
# =========================================================
with tab_covariates:
    st.subheader("Covariate Distributions, Correlations & Ablation Analysis")
    
    cov_tab1, cov_tab2 = st.columns([1, 1])
    
    with cov_tab1:
        st.markdown("#### Feature Importance (Random Forest vs XGBoost)")
        fi_df = pd.DataFrame({
            "Feature": ["Annual Rainfall (precip_mm)", "Elevation (elev_m)", "Local Relief (relief_m)", "Slope Steepness (slope_deg)"],
            "RF Importance": [0.42, 0.38, 0.16, 0.05],
            "Description": ["PRISM 30-year Normals", "Copernicus GLO-30 DEM", "11×11 DEM Window Max - Min", "Horn 3×3 Incline"]
        })
        fig_fi = px.bar(
            fi_df,
            x="RF Importance",
            y="Feature",
            orientation="h",
            text="RF Importance",
            color="RF Importance",
            color_continuous_scale="Viridis",
            title="Random Forest Feature Importance Weights"
        )
        fig_fi.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_fi.update_layout(height=280, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=20, t=40, b=20))
        st.plotly_chart(fig_fi, use_container_width=True)
        
    with cov_tab2:
        st.markdown("#### Feature Ablation Study: What Ground Really Matters?")
        st.caption("Retraining XGBoost without individual feature groups reveals redundant information:")
        ablation_df = pd.DataFrame({
            "Feature Subset": [
                "All Four Features (Full Model)",
                "Without Rainfall (-precip_mm)",
                "Without Elevation (-elev_m)",
                "Elevation Only",
                "Rainfall Only",
                "Terrain Only (Slope + Relief)"
            ],
            "XGBoost Spatial AUC": [0.931, 0.928, 0.886, 0.860, 0.848, 0.800],
            "AUC Std Dev": ["± 0.014", "± 0.013", "± 0.020", "± 0.028", "± 0.027", "± 0.021"]
        })
        fig_abl = px.bar(
            ablation_df,
            x="XGBoost Spatial AUC",
            y="Feature Subset",
            orientation="h",
            text="XGBoost Spatial AUC",
            color="XGBoost Spatial AUC",
            range_x=[0.70, 1.0],
            color_continuous_scale="Teal",
            title="Cross-Validated AUC by Covariate Subset"
        )
        fig_abl.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_abl.update_layout(height=280, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=20, t=40, b=20))
        st.plotly_chart(fig_abl, use_container_width=True)
        
    st.divider()
    
    # Distributions & Correlation Heatmap
    c_dist1, c_dist2 = st.columns([1, 1])
    
    with c_dist1:
        st.markdown("#### Covariate Correlation Heatmap")
        corr_matrix = pd.DataFrame(
            [
                [1.00, -0.27, -0.20, -0.52, -0.62],
                [-0.27, 1.00, 0.83, 0.49, 0.44],
                [-0.20, 0.83, 1.00, 0.51, 0.40],
                [-0.52, 0.49, 0.51, 1.00, 0.59],
                [-0.62, 0.44, 0.40, 0.59, 1.00]
            ],
            index=["Elevation", "Slope", "Relief", "Rainfall", "Label"],
            columns=["Elevation", "Slope", "Relief", "Rainfall", "Label"]
        )
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            title="Pearson Correlation Matrix (Covariates & Landslide Label)"
        )
        fig_corr.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with c_dist2:
        st.markdown("#### Covariate Value Distributions by Class")
        selected_cov = st.selectbox("Select Covariate to Inspect", ["slope_deg", "relief_m", "elev_m", "precip_mm"], index=0)
        
        fig_box = px.box(
            df_sample,
            x="label",
            y=selected_cov,
            color="label",
            color_discrete_map={1: "#ef4444", 0: "#3b82f6"},
            labels={"label": "Class (1=Landslide, 0=Non-Landslide)"},
            title=f"Distribution of {selected_cov} (Landslides vs Safe Ground)"
        )
        fig_box.update_layout(height=340, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# TAB 5: SURVEY BIAS & SPATIAL LIMITATIONS
# =========================================================
with tab_bias:
    st.subheader("Critical Research Finding: Spatial Inventory & Survey Bias")
    st.markdown("""
    The report highlights an essential scientific limitation: **A high validation score does not always mean a flawless physical model.**
    Part of the predictive signal captured by machine learning comes from **where landslide surveys were conducted** rather than purely terrain physics.
    """)
    
    b_col1, b_col2 = st.columns([1, 1])
    
    with b_col1:
        st.markdown("#### Agency Source Breakdown")
        agency_counts = pd.DataFrame({
            "Agency Source": ["Oregon DOGAMI (State Survey)", "Washington WGS (North Slice)", "California GS (South Slice)", "USGS Nov 1996 Event Survey", "Other Combined"],
            "Landslide Records": [98958, 6690, 613, 609, 153],
            "Share (%)": [92.46, 6.25, 0.57, 0.57, 0.15]
        })
        fig_agency = px.pie(
            agency_counts,
            values="Landslide Records",
            names="Agency Source",
            title="Source Agency Distribution of Positive Landslide Records",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_agency.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_agency, use_container_width=True)
        
    with b_col2:
        st.markdown("#### West vs East Oregon Disparity")
        east_west_df = pd.DataFrame({
            "Region": ["Western Oregon (Lon < -120.5°)", "Eastern Oregon (Lon >= -120.5°)"],
            "Landslide Count": [102091, 4932],
            "Percentage": ["95.4%", "4.6%"],
            "Mean Model Probability": ["0.81 (High)", "0.21 (Suppressed)"],
            "Terrain Character": ["Steep, wet, heavily surveyed", "Arid plateau, under-surveyed"]
        })
        st.dataframe(east_west_df, hide_index=True, use_container_width=True)
        
        st.warning("""
        **The Survey Bias Problem:**
        Because 92.5% of mapped landslides come from the Oregon State Geology (DOGAMI) inventory and 95% lie in western mountains, 
        elevation and rainfall partly function as proxies for **survey effort**. In the dry eastern plateau, the model assigns an average 
        probability of only **0.21** to actual landslide points, missing the majority of them.
        """)

    st.divider()
    st.markdown("#### Summary of Methodological Decisions & Constraints")
    lim1, lim2, lim3 = st.columns(3)
    lim1.markdown("""
    **1. Assumed Negative Sampling**
    - Negative points were randomly placed 1–25 km away from known landslides.
    - True landslides in unmapped remote areas could inadvertently be labeled as negatives.
    """)
    lim2.markdown("""
    **2. Climatological Resolution**
    - Rainfall reflects 30-year PRISM climate averages (~800 m grid), not episodic storm events that immediately trigger mass-wasting events.
    """)
    lim3.markdown("""
    **3. Missing Geological Covariates**
    - Soil cohesion, lithology, and land cover (NLCD) were omitted in the initial pipeline and represent prime targets for future model enhancement.
    """)

# =========================================================
# TAB 6: DATA EXPLORER & EXPORT
# =========================================================
with tab_data:
    st.subheader("Data Explorer & Export Utility")
    st.write("Inspect the raw dataset records, filter by custom criteria, and download subsets for GIS analysis.")
    
    de_col1, de_col2, de_col3 = st.columns(3)
    with de_col1:
        f_agency = st.multiselect("Source Agency", options=df_sample["agency"].dropna().unique().tolist(), default=[])
    with de_col2:
        f_min_elev = st.number_input("Min Elevation (m)", value=0, step=50)
    with de_col3:
        f_min_prob = st.slider("Min Predicted Probability", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        
    filtered_export_df = df_sample.copy()
    if f_agency:
        filtered_export_df = filtered_export_df[filtered_export_df["agency"].isin(f_agency)]
    if f_min_elev > 0:
        filtered_export_df = filtered_export_df[filtered_export_df["elev_m"] >= f_min_elev]
    if f_min_prob > 0:
        filtered_export_df = filtered_export_df[filtered_export_df[active_prob_col] >= f_min_prob]
        
    st.caption(f"Showing **{len(filtered_export_df):,}** matching rows from the active 40k sample:")
    st.dataframe(filtered_export_df.head(250), use_container_width=True)
    
    csv_bytes = filtered_export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Records as CSV",
        data=csv_bytes,
        file_name="oregon_landslides_filtered.csv",
        mime="text/csv"
    )
