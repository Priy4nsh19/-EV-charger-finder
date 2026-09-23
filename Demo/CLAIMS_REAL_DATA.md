# Claims Assessment — Real-Data EV Charging Experiment

This document formally distinguishes experimentally supported findings from claims that **cannot** be made based on this dataset and evaluation.

---

## 1. SUPPORTED CLAIMS

These claims are rigorously supported by the experimental evidence produced from the evaluation dataset:

### Forecasting Performance (Task A)
- ✅ **Superiority over Baselines:** LightGBM achieves **0.0554 MAE**, **0.0770 RMSE**, and **0.9407 $R^2$** on the held-out December test set (215,850 observations), outperforming naive persistence (**0.0999 MAE**, 44.5% improvement) and the historical hour-of-week baseline (**0.0656 MAE**, 15.5% improvement).
- ✅ **Feature Hierarchy:** Past utilization lag features (`utilization_rate_lag1`, `utilization_rate_lag2`) and rolling averages (`util_roll6`, `util_roll48`) dominate model importance. Contextual features (traffic congestion index, temperature, gas price) provide modest complementary signal.

### Congestion Classification (Task B)
- ✅ **Congestion Detection Performance:** LightGBM substantially improves ranking-based congestion detection, increasing ROC-AUC from 0.9236 to 0.9703 and PR-AUC from 0.5636 to 0.8070. At the fixed 0.90 congestion threshold, however, F1 changes only marginally from 0.5813 to 0.5836.

### Conformal Uncertainty Quantification
- ✅ **Empirical Coverage Validity:** Nonconformity scores were computed as the absolute prediction residuals on the November validation/calibration set. For the 90% target coverage level, the empirical 90th percentile of the calibration residuals was used to construct prediction intervals, achieving **90.19% empirical coverage** globally and **90.16% empirical coverage** hour-wise on the held-out December test set.
- ✅ **Efficiency of Temporal Calibration:** Hour-wise calibration reduces the average interval width from **0.2499 to 0.2215 (an 11.4% tightening)** without compromising coverage.
- ✅ **Cross-Network Calibration Consistency:** Coverage remains robust across all 8 individual networks, spanning from **89.31% (Blink)** to **90.71% (Electrify America)**.

### Cross-Network Generalization
- ✅ **Cross-Network Transfer:** Leave-one-network-out evaluation produced a mean generalization gap of 0.0012 MAE (seen MAE = 0.0564, held-out MAE = 0.0576, held-out persistence MAE = 0.0996). The network feature was retained during this experiment; therefore, the result should be interpreted as transfer to held-out network identities within the provided benchmark rather than as a completely operator-agnostic model.
- ✅ **Unseen Network Performance:** On every held-out network, the model substantially beats persistence (average held-out MAE 0.0576 vs persistence 0.0996, a 42.2% error reduction).

### Station Recommendation & Congestion Mitigation
- ✅ **Congestion Reduction:** In a 2,000-decision replay simulation over the held-out test set, proactive prediction-based recommendation reduces the proportion of arrivals encountering congestion ($\ge 0.90$) from **11.70% (nearest station) down to 1.40%–1.45%** (an ~8-fold reduction).
- ✅ **Quantified Distance Trade-off:** Mitigating congestion requires additional driving distance: users travel an average of **11.94 km** (with distance penalty $\lambda = 0.02$) or **14.39 km** (pure policy $\lambda = 0.0$) compared to **5.41 km** for nearest station. The recommender should be framed as an explicit availability-distance trade-off rather than universally superior.

---

## 2. CLAIMS THAT CANNOT BE MADE

These claims are **NOT** supported and must **NOT** appear in any paper, presentation, or documentation:

### Dataset Provenance & Authenticity
- ❌ **"We validated on real-world operational EV charging telemetry."**  
  *Why:* The dataset is a Kaggle benchmark containing 1,317,750 observations across 150 stations with zero missing values, perfect 30-minute periodicity, and identical sample counts per station. These are hallmarks of synthetic or synthetically regularized data.
  *Correction:* Describe it as a *"large-scale structured benchmark dataset of 150 charging stations across 8 networks."*

### Field Deployment & Operational Guarantees
- ❌ **"The system has been deployed and proven in live production."**  
  *Why:* All evaluations are offline retrospective replaying on the held-out month of December 2025.
  *Correction:* State that *"the methodology was evaluated in retrospective offline replay on held-out temporal partitions."*
- ❌ **"The model handles real-world sensor dropout, missing data, and telemetry latency."**  
  *Why:* No missing values or irregular timestamps exist in this dataset.

### Causal Driving Behavior
- ❌ **"The recommender reduced city-wide EV traffic congestion."**  
  *Why:* The replay test evaluates synthetic driver requests sampled near stations; it does not simulate multi-agent equilibrium or live driver routing.
  *Correction:* State that *"in simulated replay, the recommender steered synthetic trip requests to stations with lower arrival congestion."*

### Broad Geographic Transfer
- ❌ **"The model generalizes to unmapped cities and foreign charging grids."**  
  *Why:* All 15 cities are present throughout the train, validation, and test periods. Spatial out-of-distribution transfer to unseen cities was not evaluated.

---

## 3. RECOMMENDED ACADEMIC REPORT PHRASING

| Topic | Approved Academic Language | Unacceptable Overclaim |
|---|---|---|
| **Data** | "We evaluate our proposed methodology on a 1.3-million-observation structured benchmark dataset covering 150 stations across 8 major US networks." | ~~"We deployed our algorithm across real-world commercial EV networks."~~ |
| **Forecasting** | "The LightGBM model achieved an MAE of 0.0554 on the chronological held-out test period, outperforming naive persistence (0.0999) and hour-of-week historical averages (0.0656)." | ~~"Our model solves the EV charging forecasting problem with near-perfect accuracy."~~ |
| **Uncertainty** | "Hour-wise split conformal prediction achieved 90.16% empirical coverage at the nominal 90% level, tightening interval widths by 11.4% relative to global calibration." | ~~"Conformal prediction guarantees 100% bounds for any charger at any time."~~ |
| **Cross-Network** | "Leave-one-network-out evaluation produced a mean generalization gap of 0.0012 MAE. Because the network feature was retained, this represents transfer to held-out network identities within the benchmark rather than a completely operator-agnostic model." | ~~"The model transfers seamlessly to completely arbitrary charging infrastructure worldwide."~~ |
| **Recommender** | "In offline replay simulation, the proactive recommender reduced arrival congestion from 11.70% to 1.40%, with an associated trade-off of 6.5 km in average travel distance." | ~~"Our recommender is universally superior to existing navigation systems."~~ |
