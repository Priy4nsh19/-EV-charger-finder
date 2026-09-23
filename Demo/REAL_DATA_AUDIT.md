# Code-Level Scientific Audit: Real-Data EV Charging Experiment

**Date:** September 24, 2026  
**Audited Directory:** `c:\Users\priya\Desktop\Developer\Projects\EDA\Demo`  
**Audited Pipeline Code:** [`real_data_pipeline.py`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py)  
**Audited Artifacts:** [`real_data_results/`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_results)  

---

## 1. Executive Summary Table

| Check # | Audit Item | Status | Evidence (Code / Data Location) | Scientific Risk / Impact |
|:---|:---|:---:|:---|:---|
| **1** | **Target Construction** | **PASS** | [`real_data_pipeline.py:86,139-141`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L86): `df.sort_values(["station_id", "timestamp"])`, `df.groupby("station_id")["utilization_rate"].shift(-2)`. Excluded from `FEATURE_COLS`. | None. Target is strictly future-step $t+2$ computed independently per station. |
| **2** | **Lag Features** | **PASS** | [`real_data_pipeline.py:121-126`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L121-L126): `LAG_STEPS = [1, 2, 4, 48, 336]`. Grouped by `station_id` with `shift(lag)` for `utilization_rate` and `ports_available`. | None. Perfectly past-only observation steps (30m, 1h, 2h, 24h, 7d). |
| **3** | **Rolling Features** | **PASS** | [`real_data_pipeline.py:130-136`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L130-L136): `s.shift(1).rolling(window, min_periods=1).mean()` grouped by `station_id` for windows 6 (3h) and 48 (24h). | None. Prior `shift(1)` strictly excludes current time observation $t$. |
| **4** | **Chronological Split** | **PASS** | [`real_data_pipeline.py:40-42, 160-172`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L40): Train $\le$ 2025-10-31 23:30; Val 2025-11-01 to 2025-11-30 23:30; Test $>$ 2025-11-30 23:30. Verified assertions check max $<$ min. | None. Pure chronological partition with zero temporal overlap. |
| **5** | **Categorical Encoding** | **WARNING** | [`real_data_pipeline.py:113-116`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L113-L116): `df[col] = df[col].astype("category").cat.codes` executed on entire `df` prior to chronological split. | **Low Risk (Metadata leakage only).** No target or statistical distribution leakage. Note: `weather_condition == 'freezing'` occurs in Val/Test (Nov/Dec) but not Train (Jul-Oct). Full-dataset encoding assigned it a clean code rather than `-1` (unseen). |
| **6** | **Model Training & Test Isolation** | **PASS** | [`real_data_pipeline.py:185-199`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L185-L199): LightGBM fitted on `X_train, y_train`. `eval_set=[(X_val, y_val)]` used exclusively for early stopping (best iteration: 132). `X_test` only used for post-training inference. | None. Test data completely isolated during fitting and model selection. |
| **7** | **Hour-of-Week Baseline** | **PASS** | [`real_data_pipeline.py:207-214`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L207-L214): `hw_avg` table computed strictly on `train_temp`. Fallback to `y_train.mean()`. Test mapping uses lookup only. | None. Zero future or test information used in baseline construction. |
| **8** | **Conformal Prediction** | **PASS** | [`real_data_pipeline.py:334-386`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L334-L386): Residuals $R_i = \|y_i - \hat{y}_i\|$ computed solely on November validation set. Quantiles $q_{\text{global}}=0.1250$ and $q_h$ (24 hourly quantiles) calibrated on Val, applied to Test. | None. Split conformal assumptions strictly honored; independent coverage evaluation. |
| **9** | **Cross-Network Experiment** | **WARNING** | [`real_data_pipeline.py:269-277`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L269-L277): `train_lno = train_all[train_all["network"] != held_out_net]`. Model trains with early stopping on `eval_set=[(X_tr.iloc[:1000], y_tr.iloc[:1000])]`. `test_ho = test_all[test_all["network"] == held_out_net]`. | **Low Risk.** Training strictly isolates the held-out network. However, `network` is included in `features` as a feature column; when evaluating on the held-out network, tree splits on `network` see an unseen integer code. (Tree models handle this via default paths, but it is an imperfect transfer setup). |
| **10** | **Recommender State Isolation** | **PASS** | [`real_data_pipeline.py:445, 453-486`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L445): Recommender selects candidates among `~snap["offline"]`. Scores use `dist_km`, current `utilization_rate`, `pred`, or `hi`. `next_util` is accessed *only* at line 485 (`got[m].append(row["next_util"])`) for evaluation. | None. Zero future target leakage in station selection. |
| **11** | **Recommender Distance Metric** | **PASS** | [`real_data_pipeline.py:429-436, 475`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L429): Spherical Haversine formula implemented with Earth radius $R = 6371.0\text{ km}$. Identical implementation to [`app.py:58-64`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/app.py#L58-L64). | None. Scientifically accurate Great-Circle geographic distance. |
| **12** | **Random Baseline Reproducibility** | **PASS** | [`real_data_pipeline.py:451, 458`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L451): `rng = np.random.default_rng(seed)` with `seed=42`. Random candidate scoring: `rng.random(len(c))`. | None. Fully deterministic and reproducible. |
| **13** | **Lambda ($\lambda$) Provenance** | **PASS** | Audited repository history and codebase. `lambda=0.02` originated exclusively as the interactive slider in [`app.py:150`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/app.py#L150). Pure policies ($\lambda = 0.0$) preserved in `results_recommender_real.csv`; penalty variant isolated in `results_recommender_penalty002_real.csv`. | None. The dual reporting accurately documents the distance trade-off without silently masking the pure policy baseline. |
| **14** | **Dataset Provenance & Integrity** | **PASS** | [`real_data_pipeline.py:29, 81`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L29): Loaded directly from `DATA_CSV = "ev_charging_station_data.csv"`. Model trained on 874,500 training rows of the 1,317,750 row dataset. | None. Confirmed that `demo_table.parquet` (100,650 rows) was not used for model fitting. |
| **15** | **Artifact Separation** | **PASS** | All new outputs write to `OUT_DIR = Path("real_data_results")`. Verified that `simulated_results/` and root simulated files (`model.pkl`, `demo_table.parquet`, `results_main.csv`, etc.) remain identical and untampered. | None. Clean, unambiguous physical directory separation. |
| **16** | **Metric Reproduction & Reporting** | **PASS** | Verified all numbers in `real_data_results/*.csv` against summary tables in `RUN_REAL_DATA.md` and `CLAIMS_REAL_DATA.md`. Every digit matches exactly. | None. No discrepancies, rounding errors, or omitted figures. |

---

## 2. Deep-Dive Code Inspections

### 2.1 Target Construction & Feature Alignment
- **Code Reference:** [`real_data_pipeline.py:86, 139-143`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L86)
```python
df.sort_values(["station_id", "timestamp"], inplace=True)
...
df["next_util"] = df.groupby("station_id")["utilization_rate"].shift(-TARGET_SHIFT)
df.dropna(subset=["next_util"] + lag_cols, inplace=True)
```
- **Audit Findings:**  
  1. Chronological order per station is strictly enforced via `sort_values(["station_id", "timestamp"])`.
  2. `shift(-2)` shifts backwards 2 index positions within each station's panel group, which corresponds to exactly $2 \times 30\text{ min} = 60\text{ min}$ ahead.
  3. `next_util` is explicitly omitted from `FEATURE_COLS` and `all_features` (lines 49-57, line 145). It is supplied solely as `y_train`, `y_val`, and `y_test`.
- **Verdict:** **PASS.**

### 2.2 Lag and Rolling Window Leakage Guard
- **Code Reference:** [`real_data_pipeline.py:121-137`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L121-L137)
```python
# Lags
for lag in LAG_STEPS: # [1, 2, 4, 48, 336]
    for base in ["utilization_rate", "ports_available"]:
        col_name = f"{base}_lag{lag}"
        df[col_name] = df.groupby("station_id")[base].shift(lag)

# Rolling
for col_name, window in ROLL_WINDOWS.items(): # {util_roll6: 6, util_roll48: 48}
    df[col_name] = (
        df.groupby("station_id")["utilization_rate"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
```
- **Audit Findings:**  
  1. All lags have shift index $\ge 1$.
  2. Rolling features explicitly execute `.shift(1)` before applying `.rolling(...)`. At timestamp $t$, the rolling window includes $\{t-1, t-2, \dots\}$, strictly omitting $t$ and all future steps.
  3. The earliest 336 rows per station (7 days) evaluate to `NaN` and are cleanly dropped at line 141.
- **Verdict:** **PASS.**

### 2.3 Categorical Encoding & Distribution Shift
- **Code Reference:** [`real_data_pipeline.py:113-116`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L113-L116)
```python
cat_maps = {}
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype("category")
    cat_maps[col] = dict(enumerate(df[col].cat.categories))
    df[col] = df[col].cat.codes
```
- **Audit Findings:**  
  1. Label encoding was performed over the combined dataset rather than fitting an encoder on Train and transforming Val/Test.
  2. **Impact Analysis:** For 6 of the 7 categorical features (`network`, `station_status`, `local_event`, `pricing_type`, `location_type`, `charger_type`), all categories appear in both Train and Test.
  3. For `weather_condition`, the category `'freezing'` appears only in November (Val) and December (Test). Because encoding was executed globally, `'freezing'` received code 1, whereas fitting purely on Train would have required handling `'freezing'` as an unseen category (`-1` or fallback).
  4. Crucially, label encoding is an arbitrary integer mapping; it does **not** leak target statistics, mean encodings, or future numerical distributions into the train set.
- **Verdict:** **WARNING (Methodological nuance, zero predictive leakage).**

### 2.4 Baseline Implementations
- **Code Reference:** [`real_data_pipeline.py:207-214`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L207-L214)
```python
train_temp = train.copy()
train_temp["hw"] = train_temp["day_of_week"] * 48 + train_temp["hour_of_day"] * 2
hw_avg = train_temp.groupby(["station_id", "hw"])["next_util"].mean()
test_temp = test.copy()
test_temp["hw"] = test_temp["day_of_week"] * 48 + test_temp["hour_of_day"] * 2
test_temp = test_temp.set_index(["station_id", "hw"])
hw_pred = test_temp.index.map(lambda idx: hw_avg.get(idx, y_train.mean()))
```
- **Audit Findings:**  
  1. The hour-of-week baseline groups strictly on `train_temp`.
  2. Mapping on the test set is a pure dictionary lookup (`hw_avg.get(idx, y_train.mean())`).
  3. No test target labels enter the baseline.
- **Verdict:** **PASS.**

### 2.5 Conformal Residual Calibration & Exchangeability
- **Code Reference:** [`real_data_pipeline.py:334-378`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L334-L378)
```python
val_pred = np.clip(model.predict(val[features]), 0, 1)
val_resid = np.abs(val["next_util"].values - val_pred)
q_global = float(np.quantile(val_resid, 1 - CONFORMAL_ALPHA))
```
- **Audit Findings:**  
  1. Validation residuals are calculated exclusively on the November validation set using the model fitted on July–October.
  2. The test predictions and test labels are only evaluated after $q_{\text{global}}$ and $q_{\text{hour}}$ are fixed.
  3. Empirical coverage on test data is 90.19% (global) and 90.16% (hour-wise) against a 90.0% nominal target.
- **Verdict:** **PASS.**

### 2.6 Recommender Evaluation & Lambda Verification
- **Code Reference:** [`real_data_pipeline.py:445-486`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_pipeline.py#L445-L486)
- **Investigation of $\lambda = 0.02$:**  
  - Grep search across the pre-existing project repository found no historical script or notebook utilizing $\lambda = 0.02$.
  - The value $\lambda = 0.02$ appeared strictly in [`app.py:150`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/app.py#L150):
    `lam = st.slider("Distance penalty (utilization points per 10 km)", 0.0, 0.2, 0.02, 0.01)`
  - **Resolution in Output Files:**
    - [`results_recommender_real.csv`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_results/results_recommender_real.csv) was regenerated with pure $\lambda = 0.0$, preserving the exact 5 unpenalized policies.
    - [`results_recommender_penalty002_real.csv`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_results/results_recommender_penalty002_real.csv) was saved alongside it with $\lambda = 0.02$ to document the travel distance trade-off.
- **Verdict:** **PASS.**

---

## 3. Discrepancy & Verification Report

### A. Critical Issues
*None identified.*  
There is no target leakage, no test set contamination, no future lookahead in lags/rolling windows, and no fabrication of metrics.

### B. Non-Critical Issues
1. **Global Categorical Code Assignment:**  
   In `load_and_preprocess()`, `.cat.codes` was executed on the full dataset before chronological splitting. This gave the winter category `'freezing'` (absent in summer/fall training data) a valid integer code (code 1) rather than an out-of-vocabulary `-1` representation. Because tree models split on integer equality/ranges without statistical scaling, this did not introduce target leakage, but in a production inference setting, an explicit `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` should be fitted on train only.
2. **`network` Feature in Leave-One-Network-Out:**  
   In Phase 6, the `network` column remained in `features`. During training on 7 networks, the tree learned splits on the remaining network codes. When evaluated on the held-out network, LightGBM processed an unseen category code via default tree branching. While standard for tree ensembles, dropping `network` entirely from the feature matrix during cross-network transfer yields an even cleaner zero-shot evaluation.

### C. Methodology That Is Verified
1. **Target Construction:** Strictly $t+2$ (1-hour-ahead) computed per station.
2. **Lag Features:** All 10 lag columns use $t - k$ past values exclusively.
3. **Rolling Features:** Rolling 3h and 24h windows strictly shift by 1 prior to rolling mean aggregation.
4. **Chronological Split:** Jul–Oct (Train), Nov (Val/Calibration), Dec (Test) partitions are completely non-overlapping.
5. **Conformal Inference:** Calibrated on November residuals; evaluates to 90.16% empirical coverage on December test.
6. **Recommender Replay:** Station choice uses only information available at $t$; $t+2$ target is strictly used for ex-post evaluation.
7. **Haversine Distance:** Verified mathematically correct ($R = 6371.0\text{ km}$).
8. **Metric Fidelity:** Every figure in [`results_main_real.csv`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_results/results_main_real.csv), [`results_taskB_congestion_real.csv`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_results/results_taskB_congestion_real.csv), and [`results_cross_network_real.csv`](file:///c:/Users/priya/Desktop/Developer/Projects/EDA/Demo/real_data_results/results_cross_network_real.csv) exactly matches reported summaries.

### D. Claims That Are Safe to Make
- ✅ "On this 1.3M-row benchmark dataset, LightGBM forecasts 1-hour-ahead utilization with 0.0554 MAE, reducing error by 44.5% compared to naive persistence (0.0999 MAE) and 15.5% compared to hour-of-week averages (0.0656 MAE)."
- ✅ "Hour-wise split conformal prediction intervals achieve 90.16% empirical coverage at the nominal 90% target level on held-out test data, narrowing interval widths by 11.4% relative to global calibration."
- ✅ "In simulated replay, proactive station recommendation cuts arrival congestion ($\ge 0.90$) from 11.70% (nearest station) down to 1.40%–1.45%."
- ✅ "Cross-network evaluation yields a small generalization gap (+0.0012 MAE), outperforming persistence (0.0996) even on completely held-out operators."

### E. Claims That Should Be Weakened
- ⚠️ **"The recommender is universally superior to nearest-station routing."**  
  *Weakening:* Must be stated as a trade-off. Avoiding congestion requires traveling an additional 6.5 km to 9.0 km on average.
- ⚠️ **"Validated on real-world operational charging telemetry."**  
  *Weakening:* The dataset is a Kaggle-provided, regularized panel with 0 missing values and uniform 30-minute intervals. Describe it as a *large-scale structured benchmark dataset*, not raw operational telemetry.

### F. Recommended Code Changes (For Future Iterations)
No immediate code modifications or reruns are required since the current experiment is scientifically intact. For production hardening, the following small refactor is suggested:
1. In `load_and_preprocess()`, use `sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` fitted solely on `train` to avoid discovering categories in test.
2. In `cross_network_eval()`, exclude the `network` feature column from `features` so tree splits never encounter unseen operator IDs.
