LANDSLIDE PREDICTION - OREGON: DATA + MODEL PACKAGE
===================================================
Project report: "Landslide Prediction Using Machine Learning on Terrain and Rainfall Data: Oregon, USA"
Study area: lon -124.6 to -116.4, lat 41.9 to 46.3 (Oregon plus small slices of WA, ID, CA)

FILES
-----
landslide_model_data.csv / .parquet   full dataset, 214,043 rows (same content, parquet is smaller)
landslide_map_sample_40k.csv          random 40,000-row sample for fast map rendering
models/xgboost_model.json             XGBoost (portable format: XGBClassifier().load_model(...))
models/xgboost_model.joblib           same model, joblib
models/random_forest_model.joblib     random forest (Listing 4 parameters)
models/logistic_regression_model.joblib  scaler + logistic regression pipeline

Models were trained on ALL rows with the four features in this order:
    ["elev_m", "slope_deg", "relief_m", "precip_mm"]
They are meant for the what-if predictor only. Do not use them to score the dataset rows
(they saw those rows). Use the pred_prob_* columns instead.

COLUMNS
-------
latitude, longitude   float   point location (WGS84); study area lat 41.9-46.3, lon -124.6 to -116.5
elev_m                float   Copernicus GLO-30 elevation, m (-0.8 to 3,353)
slope_deg             float   slope, Horn method on 3x3 cells, degrees (0 to 70.9)
relief_m              float   max minus min elevation in an 11x11 cell window, m (0 to 426)
precip_mm             float   PRISM 1991-2020 mean annual precipitation, mm/year (169 to 5,106)
label                 int     1 = landslide (107,023), 0 = random non-landslide point (107,020)
pred_prob_xgb         float   out-of-fold landslide probability, XGBoost
pred_prob_rf          float   out-of-fold landslide probability, random forest
pred_prob_lr          float   out-of-fold landslide probability, logistic regression
block_id              string  0.5 degree spatial block (161 blocks), "<lat_block>_<lon_block>"
fold                  int     spatial CV fold 1-5 (GroupKFold by block_id)
confidence            float   USGS confidence score (3, 5 or 8) - landslide rows only, empty for random points
agency                string  source inventory (e.g. OR DOGAMI, WA WGS); "random" for the sampled non-landslide points
ecoregion             string  level-2 ecoregion from the inventory - landslide rows only

OUT-OF-FOLD PREDICTIONS
-----------------------
Each row's pred_prob_* value was made by a model that did NOT see that row's 0.5 degree block
(5-fold spatial cross-validation, parameters in Listing 4 of the report). They can be plotted
directly as a risk map. Landslide rows 98,958 of 107,023 come from agency "OR DOGAMI", and about
95% lie in western Oregon, so the map partly shows where surveys were done.

DATA SOURCES
------------
Landslides: USGS "Landslide Inventories across the United States", doi 10.5066/P9FZUX6N (confidence >= 3 only)
Elevation:  Copernicus GLO-30 DEM via OpenTopography, doi 10.5069/G9028PQB
Rainfall:   PRISM Climate Group, Oregon State University, 1991-2020 normals

CAUTION
-------
The two classes are balanced 50/50 by design, so probabilities are relative risk scores, not the
real chance of a landslide.
