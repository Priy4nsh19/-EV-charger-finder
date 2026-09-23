# Reproducing the Real-Data EV Charging Experiment

## 1. Dataset Overview

| Property | Value |
|---|---|
| Source file | `ev_charging_station_data.csv` |
| Total observations | 1,317,750 rows |
| Columns | 33 features |
| Stations | 150 unique charging locations across 15 US metropolitan areas |
| Operating Networks | 8 (Blink, ChargePoint, EVCS, EVgo, Electrify America, Shell Recharge, Tesla Supercharger, Volta) |
| Time range | July 1, 2025 00:00:00 – December 31, 2025 23:30:00 (6 full months) |
| Sampling interval | 30 minutes (8,785 timestamps per station; perfectly balanced panel) |
| Target variable | `next_util` (utilization rate shifted -2 intervals, i.e., 1 hour ahead) |

> [!WARNING]
> ### Dataset Limitation & Provenance Notice
> The evaluation dataset `ev_charging_station_data.csv` is a Kaggle-provided, highly structured panel with zero missing values, perfectly uniform 30-minute intervals, and regularized spatial/temporal distributions. These characteristics indicate synthetic or semi-synthetic generation. Therefore, all findings presented herein demonstrate methodological validity and pipeline feasibility on this benchmark dataset, and should **not** be interpreted as field-proven deployment performance on real operator telemetry without prospective trial validation.

---

## 2. Experimental Methodology & Validation

### 2.1 Target Construction
- The primary prediction target is 1-hour-ahead station utilization:
  $$\text{next\_util}_t = \text{utilization\_rate}_{t + 2}$$
- Computed strictly per station using `utilization_rate.shift(-2)` on 30-minute sampled data.
- The future target is **never** exposed to feature engineering or model training.

### 2.2 Feature Construction (37 Features Total)
To prevent lookahead bias and target leakage:
- **Static Station Metadata (7):** `network`, `latitude`, `longitude`, `location_type`, `charger_type`, `power_output_kw`, `ports_total`.
- **Current Observation Features (18):** `ports_available`, `ports_occupied`, `ports_out_of_service`, `utilization_rate`, `station_status`, `avg_session_duration_mins`, `current_price`, `pricing_type`, `temperature_f`, `precipitation_mm`, `weather_condition`, `gas_price_per_gallon`, `traffic_congestion_index`, `local_event`, `is_weekend`, `is_peak_hour`, `hour_of_day`, `day_of_week`.
- **Lag Features (10):** strictly past observations per station:
  - `lag1`: $t - 30\text{ min}$ (`shift(1)`)
  - `lag2`: $t - 1\text{ hour}$ (`shift(2)`)
  - `lag4`: $t - 2\text{ hours}$ (`shift(4)`)
  - `lag48`: $t - 24\text{ hours}$ (`shift(48)`)
  - `lag336`: $t - 7\text{ days}$ (`shift(336)`)
  Computed for both `utilization_rate` and `ports_available`.
- **Rolling Aggregations (2):** past rolling means using `shift(1)` to ensure the current step is not included:
  - `util_roll6`: 3-hour moving average (`shift(1).rolling(6).mean()`)
  - `util_roll48`: 24-hour moving average (`shift(1).rolling(48).mean()`)

### 2.3 Chronological Train / Validation / Test Split
To reflect real-world deployment, random shuffling is strictly prohibited. The panel is split chronologically:

| Split | Time Window | Rows | Percentage | Primary Purpose |
|---|---|---|---|---|
| **Train** | Jul 1, 2025 – Oct 31, 2025 | 874,500 | 66.4% | Model training |
| **Validation / Calibration** | Nov 1, 2025 – Nov 30, 2025 | 216,000 | 16.4% | Early stopping & conformal residual calibration |
| **Test** | Dec 1, 2025 – Dec 31, 2025 | 215,850 | 16.4% | Final held-out evaluation & recommender replay |

### 2.4 Congestion Threshold Definition
- The primary congestion threshold is fixed at:
  $$\text{utilization\_rate} \ge 0.90$$
- In the test split (December 2025), exactly **11.69%** of observations exceed this 0.90 threshold.
- *Note:* The exploratory quantile value of 0.818 observed in early EDA was intentionally excluded from formal benchmark tables to preserve consistency with the project's standard 0.90 congestion definition.

### 2.5 Cross-Network Generalization Methodology
- Leave-one-network-out cross-validation was conducted across all 8 networks:
  For each operator $N \in \{1 \dots 8\}$, a LightGBM model was trained on the remaining 7 operators and evaluated on $N$.
- MAE on seen networks, MAE on the held-out network, and the generalization gap ($\Delta \text{MAE} = \text{MAE}_{\text{held-out}} - \text{MAE}_{\text{seen}}$) were logged.
- **Methodology Caveat:** Leave-one-network-out evaluation produced a mean generalization gap of 0.0012 MAE. The network feature was retained during this experiment; therefore, the result should be interpreted as transfer to held-out network identities within the provided benchmark rather than as a completely operator-agnostic model.

### 2.6 Conformal Prediction Methodology
- Nonconformity scores were computed as the absolute prediction residuals on the November validation/calibration set. For the 90% target coverage level, the empirical 90th percentile of the calibration residuals was used to construct the prediction intervals.
- Target coverage: $\mathbf{90\%}$ ($\alpha = 0.10$).
- Evaluated as:
  1. **Global Conformal:** Single empirical 90th percentile threshold across all validation residuals ($q = 0.1250$).
  2. **Hour-wise Conformal:** 24 separate hourly empirical 90th percentile thresholds ($q_h, h \in [0, 23]$).
  3. **Per-Network Evaluation:** Coverage validated across each of the 8 individual network operators.

### 2.7 Station Recommender Replay Methodology
- Replay test simulating 2,000 user requests sampled at random timestamps during the December test period.
- Candidate set: online working stations within $50\text{ km}$ ($K=8$ nearest candidates). Offline stations are excluded.
- Evaluates 5 policies:
  1. **Nearest Station:** baseline minimizing physical travel distance.
  2. **Lowest Current Utilization:** reactive greedy baseline using current status.
  3. **Lowest Predicted Utilization:** proactive point forecast at $t+1\text{h}$.
  4. **Lowest Predicted Upper Bound:** risk-averse policy selecting lowest $90\%$ conformal upper bound ($hi$).
  5. **Random Selection:** baseline randomly picking from valid candidates.
- Primary evaluation is reported for pure policies ($\lambda = 0.0$) and with distance penalty ($\lambda = 0.02$).

---

## 3. Real-Data Experimental Results

### 3.1 Task A: 1-Hour-Ahead Utilization Forecasting (Dec 2025 Test Set)

| Model | MAE | RMSE | $R^2$ | Description |
|---|---|---|---|---|
| **Persistence (Current Util)** | 0.0999 | 0.1394 | 0.8058 | Uses current $t$ utilization as forecast for $t+1\text{h}$ |
| **Hour-of-Week Average** | 0.0656 | 0.0955 | 0.9088 | Historical station-specific mean for (day of week, hour) |
| **LightGBM Regressor** | **0.0554** | **0.0770** | **0.9407** | Gradient boosting with 37 engineered features |

> **Key Finding:** LightGBM achieves a **44.5% MAE reduction** over naive persistence and a **15.5% MAE reduction** over the strong cyclical hour-of-week baseline.

### 3.2 Task B: Congestion Classification ($\text{util} \ge 0.90$)

| Model | ROC-AUC | PR-AUC | $F_1$ Score |
|---|---|---|---|
| **Persistence ($\text{util}_t \ge 0.90$)** | 0.9236 | 0.5636 | 0.5813 |
| **LightGBM ($\hat{y}_{t+1\text{h}} \ge 0.90$)** | **0.9703** | **0.8070** | **0.5836** |

> **Key Finding:** LightGBM substantially improves ranking-based congestion detection, increasing ROC-AUC from 0.9236 to 0.9703 and PR-AUC from 0.5636 to 0.8070. At the fixed 0.90 congestion threshold, however, F1 changes only marginally from 0.5813 to 0.5836.

### 3.3 Conformal Prediction Coverage ($90\%$ Target)

| Method | Target Coverage | Empirical Coverage | Avg. Interval Width |
|---|---|---|---|
| **Global Conformal** | 90.0% | **90.19%** | 0.2499 |
| **Hour-wise Conformal** | 90.0% | **90.16%** | **0.2215** |

> **Key Finding:** Hour-wise conformal calibration successfully reduces the average interval width by **11.4%** (from 0.250 to 0.222) while preserving the exact 90% target coverage (90.16%).

#### Conformal Coverage by Operating Network:
| Network | Target Coverage | Empirical Coverage | Avg. Interval Width | Test Samples |
|---|---|---|---|---|
| **Blink** | 90.0% | 89.31% | 0.2216 | 30,219 |
| **ChargePoint** | 90.0% | 89.48% | 0.2227 | 24,463 |
| **EVCS** | 90.0% | 90.49% | 0.2253 | 23,024 |
| **EVgo** | 90.0% | 90.06% | 0.2175 | 11,512 |
| **Electrify America** | 90.0% | 90.71% | 0.2212 | 34,536 |
| **Shell Recharge** | 90.0% | 89.93% | 0.2175 | 30,219 |
| **Tesla Supercharger** | 90.0% | 90.59% | 0.2224 | 30,219 |
| **Volta** | 90.0% | 90.49% | 0.2223 | 31,658 |

### 3.4 Cross-Network Generalization (Leave-One-Network-Out)

| Held-Out Network | Seen MAE | Held-Out MAE | Baseline Persistence MAE | Generalization Gap ($\Delta$) |
|---|---|---|---|---|
| **Blink** | 0.0560 | 0.0603 | 0.1044 | +0.0042 |
| **ChargePoint** | 0.0563 | 0.0565 | 0.0996 | +0.0001 |
| **EVCS** | 0.0565 | 0.0566 | 0.0985 | +0.0000 |
| **EVgo** | 0.0563 | 0.0565 | 0.0969 | +0.0002 |
| **Electrify America** | 0.0568 | 0.0564 | 0.0973 | -0.0004 |
| **Shell Recharge** | 0.0571 | 0.0549 | 0.0971 | -0.0022 |
| **Tesla Supercharger** | 0.0562 | 0.0588 | 0.1015 | +0.0027 |
| **Volta** | 0.0559 | 0.0606 | 0.1017 | +0.0047 |
| **Aggregate Mean** | **0.0564** | **0.0576** | **0.0996** | **+0.0012** |

> **Key Finding:** The cross-network generalization gap is minimal (+0.0012 MAE on average). Even when evaluated on an entirely unseen charging operator, the model achieves **0.0576 MAE**, outperforming persistence (0.0996) by **42.2%**.  
> *Methodology Note:* The network feature was retained during this experiment; therefore, the result should be interpreted as transfer to held-out network identities within the provided benchmark rather than as a completely operator-agnostic model.

### 3.5 Station Recommender Replay Evaluation (2,000 Trips)

#### Primary Benchmark: Pure Policies ($\lambda = 0.0$, No Distance Penalty)
| Recommendation Policy | Mean Arrival Util | % Congested ($\ge 0.90$) | Avg. Travel Distance | Decisions |
|---|---|---|---|---|
| **Nearest Station** | 0.4479 | 11.70% | **5.41 km** | 2,000 |
| **Lowest Current Util** | 0.2111 | 2.45% | 14.41 km | 2,000 |
| **Lowest Predicted Util** | **0.2004** | **1.45%** | 14.39 km | 2,000 |
| **Lowest Upper Bound (Conformal)** | 0.2010 | 1.75% | 14.29 km | 2,000 |
| **Random Baseline** | 0.4331 | 11.15% | 14.77 km | 2,000 |

#### Distance-Penalized Comparison ($\lambda = 0.02$, $0.02 \times \text{km}/10$)
| Recommendation Policy | Mean Arrival Util | % Congested ($\ge 0.90$) | Avg. Travel Distance | Decisions |
|---|---|---|---|---|
| **Nearest Station** | 0.4479 | 11.70% | **5.41 km** | 2,000 |
| **Lowest Current Util** | 0.2115 | 2.40% | 12.57 km | 2,000 |
| **Lowest Predicted Util** | **0.2009** | **1.40%** | **11.94 km** | 2,000 |
| **Lowest Upper Bound (Conformal)** | 0.2013 | 1.60% | 11.91 km | 2,000 |
| **Random Baseline** | 0.4331 | 11.15% | 14.77 km | 2,000 |

> **Congestion vs Distance Trade-off Analysis:**
> - Prediction-based policies lower arrival congestion from **11.70% to 1.40%** (an ~8.4x reduction) and cut mean arrival utilization from **0.448 to 0.201**.
> - However, this requires additional travel: **11.94 km vs 5.41 km** (with $\lambda = 0.02$) or **14.39 km vs 5.41 km** (with $\lambda = 0.0$).
> - Proactive forecasting correctly steers users away from stations that are free right now but predicted to become congested upon arrival.

---

## 4. Execution Commands & Generated Artifacts

### 4.1 Reproducing the Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end real-data pipeline
python real_data_pipeline.py

# 3. Launch the dual-mode Streamlit dashboard
streamlit run app.py
```

### 4.2 Complete Artifacts Generated (`real_data_results/`)

| Filename | Type | Size | Description |
|---|---|---|---|
| `model_real.pkl` | Model | 924 KB | Trained LightGBM regressor on Jul–Oct train set |
| `features_real.json` | Metadata | 1.4 KB | List of 37 features, encoders, and conformal calibration quantiles |
| `predictions_real.parquet` | Dataset | 6.5 MB | December test set with point predictions and conformal intervals |
| `results_main_real.csv` | Metrics | 137 B | Main forecasting metrics (MAE, RMSE, R²) |
| `results_taskA_utilization_real.csv`| Metrics | 137 B | Task A regression evaluation table |
| `results_taskB_congestion_real.csv` | Metrics | 119 B | Task B classification metrics (ROC-AUC, PR-AUC, F1) |
| `results_cross_network_real.csv` | Metrics | 376 B | Leave-one-network-out cross-network evaluation across 8 operators |
| `interval_results_real.csv` | Metrics | 137 B | Conformal global and hour-wise coverage and width |
| `interval_per_network_real.csv` | Metrics | 359 B | Conformal empirical coverage per network |
| `interval_by_hour_real.csv` | Metrics | 703 B | Conformal empirical coverage and width per hour of day |
| `results_recommender_real.csv` | Metrics | 268 B | Replay test metrics for pure policies ($\lambda = 0.0$) |
| `results_recommender_penalty002_real.csv` | Metrics | 266 B | Replay test metrics with distance penalty ($\lambda = 0.02$) |
| `feature_importance_real.png` | Plot | 69 KB | Top 20 feature importances (split & gain) |
| `forecast_timeline_real.png` | Plot | 202 KB | 7-day timeline of predicted vs actual utilization with intervals |
| `forecast_vs_actual_real.png` | Plot | 210 KB | Scatter plot of predicted vs actual utilization |
| `interval_plot_real.png` | Plot | 46 KB | Hour-wise nonconformity quantiles across 24 hours |
| `cross_network_real.png` | Plot | 34 KB | Bar chart comparing seen vs held-out network MAE |

---

## 5. Novelty & Claims Assessment Summary

| Claim / Proposed Novelty | Status | Evidence from Real-Data Evaluation |
|---|---|---|
| **1. 1-Hour-Ahead Machine Learning Forecasting** | **SUPPORTED** | LightGBM achieves 0.0554 MAE (44.5% lower than persistence, 15.5% lower than hour-of-week). |
| **2. Conformal Uncertainty Intervals** | **SUPPORTED** | Achieves 90.16% empirical coverage with 11.4% tighter intervals using hour-wise calibration. Coverage holds across all 8 networks (89.3%–90.7%). |
| **3. Cross-Network Transferability** | **SUPPORTED (within benchmark)** | Generalization gap is +0.0012 MAE across 8 networks (network feature retained; represents transfer to held-out operator identities within benchmark). |
| **4. Proactive Congestion Avoidance via Recommender** | **SUPPORTED (as trade-off)** | Reduces arrival congestion from 11.7% to 1.4%, but requires +6.5 km to +9.0 km travel distance. |
| **5. Claim of Real-World Operational Telemetry** | **NOT SUPPORTED** | Dataset exhibits synthetic properties (zero missing values, balanced panel). Documented as benchmark dataset only. |
