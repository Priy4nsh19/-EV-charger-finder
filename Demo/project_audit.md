# Project Audit — EV Charging Forecasting + Station Recommendation

## 1. Actual Dataset

| Property | Value |
|---|---|
| **Filename** | [`ev_charging_station_data.csv`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/ev_charging_station_data.csv) |
| **Size** | 339 MB |
| **Rows** | 1,317,750 |
| **Columns** | 33 |
| **Rows per station** | 8,785 (perfectly balanced panel) |

### Column Names and Types

| Column | Type | Notes |
|---|---|---|
| `timestamp` | str → datetime | 30-minute intervals |
| `station_id` | str | `EV00001` – `EV00114` (150 unique) |
| `station_name` | str | e.g. "ChargePoint - Los Angeles #1" |
| `network` | str | 8 unique networks |
| `city` | str | 15 cities |
| `state` | str | |
| `latitude` | float64 | ✅ Available |
| `longitude` | float64 | ✅ Available |
| `location_type` | str | e.g. Shopping Center |
| `charger_type` | str | e.g. DC Fast Charge |
| `power_output_kw` | float64 | |
| `amenities_nearby` | str | *Not in simulated pipeline* |
| `ports_total` | int64 | |
| `ports_available` | int64 | |
| `ports_occupied` | int64 | |
| `ports_out_of_service` | int64 | |
| `utilization_rate` | float64 | Range [0.02, 0.98] |
| `station_status` | str | `operational`, `offline`, `partial_outage`, `under_maintenance` |
| `estimated_wait_time_mins` | int64 | *Not in simulated pipeline* |
| `avg_session_duration_mins` | int64 | |
| `current_price` | float64 | |
| `pricing_type` | str | |
| `temperature_f` | float64 | |
| `precipitation_mm` | float64 | |
| `weather_condition` | str | |
| `gas_price_per_gallon` | float64 | |
| `traffic_congestion_index` | int64 | |
| `local_event` | str | |
| `is_weekend` | bool | |
| `is_peak_hour` | bool | |
| `hour_of_day` | int64 | |
| `day_of_week` | int64 | |
| `month` | int64 | *Not in simulated pipeline* |

### Missing Values

**Zero missing values across all 33 columns.** This is unusual for real-world data and suggests the dataset is a high-quality synthetic/Kaggle-style dataset, but the user designates it as the "actual" dataset for this project.

### Summary Statistics

| Property | Value |
|---|---|
| **Unique stations** | 150 |
| **Unique networks** | 8 (Blink, ChargePoint, EVCS, EVgo, Electrify America, Shell Recharge, Tesla Supercharger, Volta) |
| **Cities** | 15 across 12 states |
| **Timestamp range** | 2025-07-01 00:00 → 2025-12-31 00:00 |
| **Time granularity** | 30 minutes |
| **Utilization range** | [0.02, 0.98] |
| **Station status values** | operational, offline, partial_outage, under_maintenance |
| **Lat/Lon** | ✅ Available for all stations |

---

## 2. Current Simulated Pipeline

### Model: [`model.pkl`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/model.pkl)

| Property | Value |
|---|---|
| Type | LGBMRegressor |
| Features | 37 (listed in [`features.json`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/features.json)) |
| Best iteration | 132 / 800 |
| Objective | Regression (default = L2) |
| Key hyperparams | num_leaves=63, lr=0.1, subsample=0.8, colsample=0.8 |

### Feature List (from `features.json`)

Static features: `network`, `latitude`, `longitude`, `location_type`, `charger_type`, `power_output_kw`, `ports_total`

Current-time features: `ports_available`, `ports_occupied`, `ports_out_of_service`, `utilization_rate`, `station_status`, `avg_session_duration_mins`, `current_price`, `pricing_type`, `temperature_f`, `precipitation_mm`, `weather_condition`, `gas_price_per_gallon`, `traffic_congestion_index`, `local_event`, `is_weekend`, `is_peak_hour`, `hour_of_day`, `day_of_week`

Lag features: `ports_available_lag1/2/4/48/336`, `utilization_rate_lag1/2/4/48/336`

Rolling features: `util_roll6`, `util_roll48`

### Target Variable

**`next_util`** — utilization rate approximately 1 hour ahead (2 steps at 30-min granularity).

### Conformal Prediction (from `features.json`)

- `q_global`: 0.1344 (global nonconformity quantile)
- `q_hour`: 24 hour-specific quantiles
- These produce `lo` and `hi` columns in `demo_table.parquet`

### Demo Table: [`demo_table.parquet`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/demo_table.parquet)

- Shape: 100,650 rows × 18 columns
- Time range: 2025-12-17 → 2025-12-30 (test period, last ~2 weeks)
- Contains: station metadata + `utilization_rate`, `next_util`, `pred`, `lo`, `hi`
- **150 stations** — but the CSV only has station IDs EV00001–EV00114 (114 IDs). The demo may have used a different generation run.

---

## 3. Simulated / Demo Artifacts (to preserve)

| File | Type | Status |
|---|---|---|
| `model.pkl` | LightGBM model | Simulated-data trained |
| `features.json` | Feature list + conformal quantiles | Simulated |
| `demo_table.parquet` | Test predictions + intervals | Simulated |
| `results_main.csv` | Congestion classification metrics (ROC/PR/F1/Brier) | Simulated |
| `results_taskA_utilization.csv` | Regression metrics (MAE/RMSE/R²) | Simulated |
| `results_taskB_congestion.csv` | Congestion classification metrics | Simulated |
| `results_cross_network.csv` | Leave-one-network-out MAE | Simulated |
| `cross_network.png` | Cross-network plot | Simulated |
| `interval_plot.png` | Conformal interval plot | Simulated |
| `feature_importance.png` | Feature importance (split) | Simulated |
| `feature_importance_util.png` | Feature importance (utilization) | Simulated |

---

## 4. Compatibility Analysis

### ✅ Fully Compatible

| Experiment | Feasibility |
|---|---|
| **Utilization forecasting (Task A)** | ✅ `utilization_rate` exists, 30-min granularity supports 1h-ahead target |
| **Congestion classification (Task B)** | ✅ Can threshold `utilization_rate` for congestion |
| **Cross-network evaluation** | ✅ 8 distinct networks available |
| **Conformal prediction intervals** | ✅ Sufficient chronological calibration data |
| **Station recommender** | ✅ Station IDs, lat/lon, utilization, status all available |

### ⚠️ Notes

- The actual CSV has **2 extra columns** not used by the old pipeline: `amenities_nearby`, `estimated_wait_time_mins`, `month`
- The old pipeline's lag numbering (lag1=30min, lag2=1h, lag4=2h, lag48=24h, lag336=7d) aligns with 30-min intervals ✅
- The "1 hour ahead" target = shift by 2 rows (each row = 30 min) ✅
- All 8 networks have ≥7 stations → cross-network leave-one-out is viable ✅

---

## 5. Real-Data Artifacts to Create

| File | Directory |
|---|---|
| `model_real.pkl` | `real_data_results/` |
| `features_real.json` | `real_data_results/` |
| `predictions_real.parquet` | `real_data_results/` |
| `results_main_real.csv` | `real_data_results/` |
| `results_cross_network_real.csv` | `real_data_results/` |
| `results_recommender_real.csv` | `real_data_results/` |
| `interval_results_real.csv` | `real_data_results/` |
| `cross_network_real.png` | `real_data_results/` |
| `interval_plot_real.png` | `real_data_results/` |
| `feature_importance_real.png` | `real_data_results/` |
| `forecast_vs_actual_real.png` | `real_data_results/` |
