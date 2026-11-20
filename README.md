# Oregon Landslide Susceptibility AI Dashboard

A GIS-integrated machine learning web application for landslide susceptibility assessment across Oregon, USA. Based on the data science research report *"Landslide Prediction Using Machine Learning on Terrain and Rainfall Data: Oregon, USA"* by **Farzan**.

---

## 🏔️ Overview

This dashboard translates an end-to-end spatial machine learning pipeline into a decision-support tool for geotechnical engineers, environmental researchers, and regional planners. It models landslide susceptibility from open terrain and climatological data:

- **Landslide Inventory:** USGS Landslide Inventories across the United States (v2), filtered to high-confidence events ($\ge 3$).
- **Negative Sampling:** Controlled distance-buffer random sampling ($1\text{ km} \le d \le 25\text{ km}$) preserving survey-area integrity (50/50 balanced dataset, 214,043 points).
- **Terrain & Climate Covariates:**
  - Elevation ($m$) from **Copernicus GLO-30 DEM** (OpenTopography)
  - Slope steepness ($^\circ$) via **Horn 3×3 method**
  - Local relief ($m$) in an **$11 \times 11$ cell neighborhood** (~$240\text{ m} \times 340\text{ m}$)
  - Mean annual rainfall ($mm/\text{year}$) from **PRISM 1991–2020 30-year Normals** (~800 m grid)
- **Validation:** **5-Fold Spatial Block Cross-Validation** ($0.5^\circ$ grid blocks, ~50 km wide) to eliminate optimistic spatial autocorrelation bias.
- **Top Classifier:** **XGBoost** achieving an out-of-fold spatial **ROC-AUC of 0.931 ± 0.014** and finding **85.6%** of all recorded landslides.

---

## 🚀 How to Launch the Dashboard

Open your terminal or PowerShell and run:

```bash
cd "C:\Users\farza\.gemini\antigravity\scratch\landslide_prediction_dashboard"
streamlit run app.py
```

The application will automatically open in your browser at:
`http://localhost:8501`

---

## 📊 Dashboard Modules & Features

### 1. 🗺️ Geospatial Susceptibility Map
- Interactive high-performance Mapbox/CartoDB visualization of Oregon.
- Color code points by **Predicted Hazard Probability** ($0.0 - 1.0$), **Ground Truth Class** ($1 \text{ vs } 0$), or **Classification Outcome** (True Positive, True Negative, False Positive, False Negative).
- Filter by ground truth class and hazard severity tiers.
- Inspect terrain features, rainfall, and risk levels with hover tooltips.

### 2. ⚡ Live What-If Hazard Simulator & Calculator
- Real-time geotechnical hazard calculation powered by the trained **XGBoost**, **Random Forest**, and **Logistic Regression** models.
- Interactive sliders for **Elevation**, **Slope**, **Relief**, and **Annual Rainfall**.
- **1-Click Regional Presets:**
  - 🌊 *Coast Range Escarpment* (High Hazard)
  - 🏔️ *Cascades Mountain Ridge* (Elevated Hazard)
  - 🏡 *Willamette Valley Foothills* (Low Hazard)
  - 🌾 *Columbia Basin / East Desert* (Negligible Hazard)
- Speedometer risk gauge with color-coded safety classifications (`Low`, `Moderate`, `High`, `Severe`).
- Dynamic baseline comparator benchmarking user inputs against historic Oregon landslide averages.

### 3. 📈 Model Evaluation & Spatial Block CV
- Cross-validation scorecard comparing Logistic Regression, Random Forest, and XGBoost.
- Out-of-fold ROC curves with interactive hover AUC inspection.
- XGBoost stability breakdown across independent spatial block folds.
- **Dynamic Decision Threshold Slider:** Live confusion matrix and cost-benefit analysis demonstrating why lowering threshold from $0.50$ to $0.30$ captures $>93\%$ of landslides to protect critical infrastructure.

### 4. 🔬 Covariates, Feature Importance & Ablation
- Feature importance comparison showing why rainfall ($0.42$) and elevation ($0.38$) dominate tree decisions.
- **Ablation Study Chart:** Visualizing why removing rainfall drops AUC by only $0.003$ when elevation is present, and why terrain geomorphology alone (slope + relief) achieves $\text{AUC } 0.800$.
- Interactive Pearson correlation heatmap and covariate distribution boxplots.

### 5. ⚠️ Survey Bias & Spatial Limitations
- Critical exploration of inventory bias:
  - **92.5%** of landslide points originate from a single state survey (Oregon DOGAMI).
  - **95%** of records are concentrated in western Oregon.
  - Mean predicted probability drops from **0.81** in the west to **0.21** in the arid eastern plateau.
- Analysis of assumed negative sampling and the temporal limitations of 30-year rainfall normals.

### 6. 📋 Data Explorer & Export
- Searchable, sortable tabular explorer.
- Filter records by agency, elevation range, and predicted probability.
- One-click CSV export for downstream GIS integration (QGIS, ArcGIS).

---

## 📂 Project Directory Structure

```
landslide_prediction_dashboard/
├── app.py                         # Complete interactive Streamlit dashboard
├── README.md                      # Documentation and architecture guide
├── landslide_map_sample_40k.csv   # 40,000-row representative sample for instant map rendering
├── landslide_model_data.parquet   # Full 214,043-row dataset with out-of-fold predictions
├── landslide_model_data.csv       # CSV copy of the full dataset
└── models/
    ├── xgboost_model.json         # Portable trained XGBoost model
    ├── xgboost_model.joblib       # Joblib serialization
    ├── random_forest_model.joblib # Trained Random Forest model (Listing 4 parameters)
    └── logistic_regression_model.joblib # Scaler + Logistic Regression pipeline
```
