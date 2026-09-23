"""
Real-Data EV Charging Forecasting Pipeline
===========================================
Phases 3–9: Preprocessing, training, cross-network, conformal, recommender, plots.

Run:
    python real_data_pipeline.py

Outputs → real_data_results/
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────
DATA_CSV   = "ev_charging_station_data.csv"
OUT_DIR    = Path("real_data_results")
OUT_DIR.mkdir(exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────
TARGET_SHIFT   = 2        # 2 × 30 min = 1 hour ahead
LAG_STEPS      = [1, 2, 4, 48, 336]   # 30m, 1h, 2h, 24h, 7d
ROLL_WINDOWS   = {"util_roll6": 6, "util_roll48": 48}   # 3h, 24h
CONGESTION_THR = 0.90
CONFORMAL_ALPHA = 0.10    # 90% coverage

TRAIN_END  = "2025-10-31 23:30:00"
VAL_END    = "2025-11-30 23:30:00"
# test = everything after VAL_END

CATEGORICAL_COLS = [
    "network", "location_type", "charger_type",
    "pricing_type", "weather_condition", "local_event", "station_status",
]

FEATURE_COLS = [
    "network", "latitude", "longitude", "location_type", "charger_type",
    "power_output_kw", "ports_total", "ports_available", "ports_occupied",
    "ports_out_of_service", "utilization_rate", "station_status",
    "avg_session_duration_mins", "current_price", "pricing_type",
    "temperature_f", "precipitation_mm", "weather_condition",
    "gas_price_per_gallon", "traffic_congestion_index", "local_event",
    "is_weekend", "is_peak_hour", "hour_of_day", "day_of_week",
]

LGB_PARAMS = dict(
    boosting_type="gbdt",
    n_estimators=800,
    num_leaves=63,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    subsample_freq=1,
    min_child_samples=20,
    verbose=-1,
    n_jobs=-1,
)


# =====================================================================
# PHASE 3 — PREPROCESSING
# =====================================================================
def load_and_preprocess() -> pd.DataFrame:
    print("=" * 70)
    print("PHASE 3 — LOADING & PREPROCESSING")
    print("=" * 70)

    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # ── Parse timestamps ─────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(["station_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Duplicate check ──────────────────────────────────────────────
    dups = df.duplicated(subset=["station_id", "timestamp"]).sum()
    print(f"Duplicate (station, timestamp) rows: {dups}")
    if dups > 0:
        df.drop_duplicates(subset=["station_id", "timestamp"], keep="first", inplace=True)
        print(f"  → Dropped duplicates. New row count: {len(df):,}")

    # ── Impossible-value check ───────────────────────────────────────
    bad_util = ((df["utilization_rate"] < 0) | (df["utilization_rate"] > 1)).sum()
    bad_ports = (df["ports_available"] < 0).sum()
    print(f"Impossible utilization values: {bad_util}")
    print(f"Negative ports_available: {bad_ports}")

    # ── Timestamp gap check (per station) ────────────────────────────
    gap_count = 0
    for sid, grp in df.groupby("station_id"):
        diffs = grp["timestamp"].diff().dropna()
        unexpected = diffs[diffs != pd.Timedelta(minutes=30)]
        if len(unexpected) > 0:
            gap_count += len(unexpected)
    print(f"Timestamp gaps (!=30 min) across all stations: {gap_count}")

    # ── Save categorical mapping before encoding ─────────────────────
    cat_maps = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
        cat_maps[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes
    print(f"Encoded {len(CATEGORICAL_COLS)} categorical columns")

    # ── Lag features (past-only) ─────────────────────────────────────
    lag_cols = []
    for lag in LAG_STEPS:
        for base in ["utilization_rate", "ports_available"]:
            col_name = f"{base}_lag{lag}"
            df[col_name] = df.groupby("station_id")[base].shift(lag)
            lag_cols.append(col_name)
    print(f"Created {len(lag_cols)} lag features")

    # ── Rolling features (past-only) ─────────────────────────────────
    roll_cols = []
    for col_name, window in ROLL_WINDOWS.items():
        df[col_name] = (
            df.groupby("station_id")["utilization_rate"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        roll_cols.append(col_name)
    print(f"Created {len(roll_cols)} rolling features")

    # ── Target: 1-hour-ahead utilization ─────────────────────────────
    df["next_util"] = df.groupby("station_id")["utilization_rate"].shift(-TARGET_SHIFT)
    before_drop = len(df)
    df.dropna(subset=["next_util"] + lag_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dropped {before_drop - len(df):,} rows with NaN lags/target -> {len(df):,} usable rows")

    all_features = FEATURE_COLS + lag_cols + roll_cols
    print(f"Total features: {len(all_features)}")
    print()

    return df, all_features, cat_maps


# =====================================================================
# PHASE 4 — CHRONOLOGICAL SPLIT
# =====================================================================
def chronological_split(df: pd.DataFrame):
    print("=" * 70)
    print("PHASE 4 — CHRONOLOGICAL SPLIT")
    print("=" * 70)

    train = df[df["timestamp"] <= TRAIN_END].copy()
    val   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)].copy()
    test  = df[df["timestamp"] > VAL_END].copy()

    print(f"Train : {len(train):>10,} rows | {train['timestamp'].min()} -> {train['timestamp'].max()}")
    print(f"Val   : {len(val):>10,} rows | {val['timestamp'].min()} -> {val['timestamp'].max()}")
    print(f"Test  : {len(test):>10,} rows | {test['timestamp'].min()} -> {test['timestamp'].max()}")

    # Leakage guard: verify no overlap
    assert train["timestamp"].max() < val["timestamp"].min(), "Train/val overlap!"
    assert val["timestamp"].max() < test["timestamp"].min(), "Val/test overlap!"
    print("No temporal overlap between splits")
    print()

    return train, val, test


# =====================================================================
# PHASE 5 — TRAIN MODEL + BASELINES
# =====================================================================
def train_model(train, val, test, features):
    print("=" * 70)
    print("PHASE 5 — TRAINING LIGHTGBM MODEL")
    print("=" * 70)

    X_train, y_train = train[features], train["next_util"]
    X_val,   y_val   = val[features],   val["next_util"]
    X_test,  y_test  = test[features],  test["next_util"]

    # ── Train LightGBM ───────────────────────────────────────────────
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )
    print(f"Best iteration: {model.best_iteration_}")

    pred_test = model.predict(X_test)
    pred_test = np.clip(pred_test, 0, 1)

    # ── Baselines ────────────────────────────────────────────────────
    # 1. Persistence: current utilization as prediction
    persist_pred = test["utilization_rate"].values

    # 2. Hour-of-week average (from training set)
    train_temp = train.copy()
    train_temp["hw"] = train_temp["day_of_week"] * 48 + train_temp["hour_of_day"] * 2
    hw_avg = train_temp.groupby(["station_id", "hw"])["next_util"].mean()
    test_temp = test.copy()
    test_temp["hw"] = test_temp["day_of_week"] * 48 + test_temp["hour_of_day"] * 2
    test_temp = test_temp.set_index(["station_id", "hw"])
    hw_pred = test_temp.index.map(lambda idx: hw_avg.get(idx, y_train.mean()))
    hw_pred = np.array(hw_pred, dtype=float)

    # ── Metrics ──────────────────────────────────────────────────────
    results = []
    for name, pred in [
        ("Persistence (current util)", persist_pred),
        ("Hour-of-week avg", hw_pred),
        ("LightGBM", pred_test),
    ]:
        mae  = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)
        results.append({"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)})
        print(f"  {name:35s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "results_main_real.csv", index=False)
    print(f"\n  Saved -> results_main_real.csv")

    # ── Save model ───────────────────────────────────────────────────
    with open(OUT_DIR / "model_real.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved -> model_real.pkl")

    # ── Save predictions ─────────────────────────────────────────────
    pred_df = test[["timestamp", "station_id", "station_name", "network",
                     "city", "state", "latitude", "longitude",
                     "location_type", "charger_type", "power_output_kw",
                     "ports_total", "station_status",
                     "utilization_rate", "next_util"]].copy()
    pred_df["pred"] = pred_test
    pred_df.to_parquet(OUT_DIR / "predictions_real.parquet", index=False)
    print(f"  Saved -> predictions_real.parquet ({len(pred_df):,} rows)")
    print()

    return model, pred_test, y_test.values, test, pred_df


# =====================================================================
# PHASE 6 — CROSS-NETWORK EVALUATION
# =====================================================================
def cross_network_eval(df, features):
    print("=" * 70)
    print("PHASE 6 — CROSS-NETWORK EVALUATION")
    print("=" * 70)

    train_all = df[df["timestamp"] <= TRAIN_END]
    test_all  = df[df["timestamp"] > VAL_END]

    networks = sorted(df["network"].unique())
    print(f"Networks found: {len(networks)}")

    rows = []
    for held_out_net in networks:
        # Train without held-out network
        train_lno = train_all[train_all["network"] != held_out_net]
        X_tr = train_lno[features]
        y_tr = train_lno["next_util"]

        model_lno = lgb.LGBMRegressor(**LGB_PARAMS)
        model_lno.fit(X_tr, y_tr,
                      eval_set=[(X_tr.iloc[:1000], y_tr.iloc[:1000])],
                      eval_metric="mae",
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        # Test on seen networks
        test_seen = test_all[test_all["network"] != held_out_net]
        pred_seen = np.clip(model_lno.predict(test_seen[features]), 0, 1)
        mae_seen = mean_absolute_error(test_seen["next_util"], pred_seen)

        # Test on held-out network
        test_ho = test_all[test_all["network"] == held_out_net]
        pred_ho = np.clip(model_lno.predict(test_ho[features]), 0, 1)
        mae_ho = mean_absolute_error(test_ho["next_util"], pred_ho)

        # Persistence baseline on held-out
        mae_persist = mean_absolute_error(test_ho["next_util"], test_ho["utilization_rate"])

        gap = mae_ho - mae_seen
        rows.append({
            "held_out_network": held_out_net,
            "MAE_seen": round(mae_seen, 4),
            "MAE_held_out": round(mae_ho, 4),
            "MAE_persistence": round(mae_persist, 4),
            "gap": round(gap, 4),
        })
        print(f"  Network {held_out_net:>3}: seen={mae_seen:.4f}  held_out={mae_ho:.4f}  gap={gap:+.4f}")

    cn_df = pd.DataFrame(rows)
    cn_df.to_csv(OUT_DIR / "results_cross_network_real.csv", index=False)
    print(f"\n  Saved -> results_cross_network_real.csv")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cn_df))
    w = 0.30
    ax.bar(x - w/2, cn_df["MAE_seen"], w, label="Seen networks", color="#4c78a8")
    ax.bar(x + w/2, cn_df["MAE_held_out"], w, label="Held-out network", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(cn_df["held_out_network"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("MAE")
    ax.set_title("Cross-Network Generalization (Leave-One-Network-Out) — Real Data")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_network_real.png", dpi=150)
    plt.close(fig)
    print("  Saved -> cross_network_real.png\n")

    return cn_df


# =====================================================================
# PHASE 7 — CONFORMAL PREDICTION INTERVALS
# =====================================================================
def conformal_intervals(model, val, test, features, pred_df):
    print("=" * 70)
    print("PHASE 7 — CONFORMAL PREDICTION INTERVALS")
    print("=" * 70)

    # ── Calibration on validation set ────────────────────────────────
    val_pred = np.clip(model.predict(val[features]), 0, 1)
    val_resid = np.abs(val["next_util"].values - val_pred)

    q_global = float(np.quantile(val_resid, 1 - CONFORMAL_ALPHA))
    print(f"  Global nonconformity quantile (90%): {q_global:.4f}")

    # Hour-specific quantiles
    val_hours = val["hour_of_day"].values
    q_hour = {}
    for h in range(24):
        mask = val_hours == h
        if mask.sum() > 10:
            q_hour[h] = float(np.quantile(val_resid[mask], 1 - CONFORMAL_ALPHA))
        else:
            q_hour[h] = q_global
    print(f"  Computed hour-specific quantiles for {len(q_hour)} hours")

    # ── Save features + quantiles ────────────────────────────────────
    feat_info = {
        "features": features,
        "q_global": q_global,
        "q_hour": {str(k): v for k, v in q_hour.items()},
    }
    with open(OUT_DIR / "features_real.json", "w") as f:
        json.dump(feat_info, f)
    print(f"  Saved -> features_real.json")

    # ── Apply to test set ────────────────────────────────────────────
    test_pred = pred_df["pred"].values
    test_hours = test["hour_of_day"].values
    y_test = pred_df["next_util"].values

    # Global intervals
    lo_g = np.clip(test_pred - q_global, 0, 1)
    hi_g = np.clip(test_pred + q_global, 0, 1)
    cov_g = np.mean((y_test >= lo_g) & (y_test <= hi_g))
    width_g = np.mean(hi_g - lo_g)

    # Hour-wise intervals
    lo_h = np.clip(test_pred - np.array([q_hour[h] for h in test_hours]), 0, 1)
    hi_h = np.clip(test_pred + np.array([q_hour[h] for h in test_hours]), 0, 1)
    cov_h = np.mean((y_test >= lo_h) & (y_test <= hi_h))
    width_h = np.mean(hi_h - lo_h)

    print(f"\n  Global  : coverage={cov_g:.4f}  avg_width={width_g:.4f}")
    print(f"  Hourwise: coverage={cov_h:.4f}  avg_width={width_h:.4f}")

    interval_df = pd.DataFrame([
        {"method": "Global conformal", "target_coverage": 0.90, "empirical_coverage": round(cov_g, 4), "avg_interval_width": round(width_g, 4)},
        {"method": "Hour-wise conformal", "target_coverage": 0.90, "empirical_coverage": round(cov_h, 4), "avg_interval_width": round(width_h, 4)},
    ])
    interval_df.to_csv(OUT_DIR / "interval_results_real.csv", index=False)
    print(f"  Saved -> interval_results_real.csv")

    # ── Add intervals to predictions parquet ─────────────────────────
    pred_df["lo"] = lo_h
    pred_df["hi"] = hi_h
    pred_df.to_parquet(OUT_DIR / "predictions_real.parquet", index=False)
    print(f"  Updated predictions_real.parquet with lo/hi columns")

    # ── Coverage by hour plot ────────────────────────────────────────
    cov_by_hour = []
    for h in range(24):
        mask = test_hours == h
        if mask.sum() > 0:
            c_g = np.mean((y_test[mask] >= lo_g[mask]) & (y_test[mask] <= hi_g[mask]))
            c_h = np.mean((y_test[mask] >= lo_h[mask]) & (y_test[mask] <= hi_h[mask]))
            cov_by_hour.append({"hour": h, "global_coverage": c_g, "hourwise_coverage": c_h})

    cov_hour_df = pd.DataFrame(cov_by_hour)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cov_hour_df["hour"] - 0.2, cov_hour_df["global_coverage"], 0.35,
           label="Global", color="#4c78a8", alpha=0.8)
    ax.bar(cov_hour_df["hour"] + 0.2, cov_hour_df["hourwise_coverage"], 0.35,
           label="Hour-wise", color="#e45756", alpha=0.8)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="Target 90%")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Conformal Prediction Interval Coverage by Hour (Real Data)")
    ax.set_xticks(range(24))
    ax.set_ylim(0.75, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "interval_plot_real.png", dpi=150)
    plt.close(fig)
    print("  Saved -> interval_plot_real.png\n")

    return q_global, q_hour, pred_df


# =====================================================================
# PHASE 8 — STATION RECOMMENDER
# =====================================================================
def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def recommender_eval(pred_df, n_trips=2000, k=8, max_km=50, lam=0.02, seed=42):
    print("=" * 70)
    print("PHASE 8 — STATION RECOMMENDER EVALUATION")
    print("=" * 70)

    df = pred_df.copy()
    # station_status was encoded: alphabetical order -> offline=0, operational=1, partial_outage=2, under_maintenance=3
    df["offline"] = (df["station_status"] == 0)

    snapshots = {t: g for t, g in df.groupby("timestamp")}
    times = list(snapshots.keys())
    stn_locs = df.drop_duplicates("station_id")[["station_id", "latitude", "longitude"]].values

    rng = np.random.default_rng(seed)

    methods = {
        "Nearest": lambda c, lam: c["dist_km"],
        "Lowest current util": lambda c, lam: c["utilization_rate"] + lam * c["dist_km"] / 10,
        "Lowest predicted util": lambda c, lam: c["pred"] + lam * c["dist_km"] / 10,
        "Lowest upper bound": lambda c, lam: c["hi"] + lam * c["dist_km"] / 10,
        "Random": lambda c, lam: rng.random(len(c)),
    }

    got = {m: [] for m in methods}
    dist = {m: [] for m in methods}

    for _ in range(n_trips):
        t = times[rng.integers(len(times))]
        snap = snapshots[t]

        # Random user location near a station
        idx = rng.integers(len(stn_locs))
        user_lat = float(stn_locs[idx, 1]) + rng.uniform(-0.1, 0.1)
        user_lon = float(stn_locs[idx, 2]) + rng.uniform(-0.1, 0.1)

        # Find candidates
        c = snap[~snap["offline"]].copy()
        c["dist_km"] = haversine(user_lat, user_lon, c["latitude"].values, c["longitude"].values)
        c = c[c["dist_km"] <= max_km].nsmallest(k, "dist_km")

        if len(c) < 2:
            continue

        for m, scorer in methods.items():
            score = np.asarray(scorer(c, lam)) + 1e-4 * c["dist_km"].values
            pick_idx = int(np.argmin(score))
            row = c.iloc[pick_idx]
            got[m].append(row["next_util"])
            dist[m].append(row["dist_km"])

    used = len(got["Nearest"])
    print(f"  Evaluated {used} trips (of {n_trips} attempted)")

    if used < 10:
        print("  WARNING: Too few valid trips for reliable evaluation")
        with open(OUT_DIR / "recommender_feasibility.md", "w") as f:
            f.write("# Recommender Feasibility\n\n"
                    f"Only {used} trips had >=2 candidate stations within {max_km} km.\n"
                    "The recommender experiment cannot produce reliable results.\n")
        return None

    rows = []
    for m in methods:
        a = np.array(got[m])
        d = np.array(dist[m])
        rows.append({
            "method": m,
            "mean_arrival_util": round(float(a.mean()), 4),
            "pct_congested_ge90": round(100 * float((a >= CONGESTION_THR).mean()), 2),
            "avg_distance_km": round(float(d.mean()), 2),
            "n_decisions": used,
        })
        print(f"  {m:25s}  util={a.mean():.4f}  congested={100*(a>=CONGESTION_THR).mean():.1f}%  dist={d.mean():.1f}km")

    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(OUT_DIR / "results_recommender_real.csv", index=False)
    print(f"\n  Saved -> results_recommender_real.csv\n")
    return rec_df


# =====================================================================
# PHASE 9 — PLOTS
# =====================================================================
def generate_plots(model, test, features, pred_test, y_test, pred_df):
    print("=" * 70)
    print("PHASE 9 — GENERATING PLOTS")
    print("=" * 70)

    # ── Feature importance ───────────────────────────────────────────
    importances = model.feature_importances_
    feat_names = features
    sorted_idx = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx][::-1], color="#4c78a8")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (split)")
    ax.set_title("Top 20 Feature Importances - Real Data LightGBM")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_real.png", dpi=150)
    plt.close(fig)
    print("  Saved -> feature_importance_real.png")

    # ── Forecast vs Actual (scatter) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    sample_idx = np.random.default_rng(0).choice(len(y_test), min(5000, len(y_test)), replace=False)
    ax.scatter(y_test[sample_idx], pred_test[sample_idx], alpha=0.15, s=5, color="#4c78a8")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("Actual Utilization (+1h)")
    ax.set_ylabel("Predicted Utilization (+1h)")
    ax.set_title("Forecast vs Actual - Real Data")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_vs_actual_real.png", dpi=150)
    plt.close(fig)
    print("  Saved -> forecast_vs_actual_real.png")

    # ── Time-series sample ───────────────────────────────────────────
    sample_station = pred_df["station_id"].unique()[0]
    s = pred_df[pred_df["station_id"] == sample_station].sort_values("timestamp").head(200)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(s["timestamp"], s["next_util"], label="Actual +1h", color="#333", linewidth=1)
    ax.plot(s["timestamp"], s["pred"], label="Predicted +1h", color="#e45756", linewidth=1, alpha=0.8)
    if "lo" in s.columns and "hi" in s.columns:
        ax.fill_between(s["timestamp"], s["lo"], s["hi"], alpha=0.2, color="#e45756", label="90% interval")
    ax.set_xlabel("Time")
    ax.set_ylabel("Utilization")
    ax.set_title(f"Forecast Timeline - Station {sample_station} (Real Data)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_timeline_real.png", dpi=150)
    plt.close(fig)
    print("  Saved -> forecast_timeline_real.png\n")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("\n" + "=" * 70)
    print("  REAL-DATA EV CHARGING FORECASTING PIPELINE")
    print("=" * 70 + "\n")

    # Phase 3
    df, features, cat_maps = load_and_preprocess()

    # Phase 4
    train, val, test = chronological_split(df)

    # Phase 5
    model, pred_test, y_test, test_df, pred_df = train_model(train, val, test, features)

    # Phase 6
    cn_df = cross_network_eval(df, features)

    # Phase 7
    q_global, q_hour, pred_df = conformal_intervals(model, val, test_df, features, pred_df)

    # Phase 8
    rec_df = recommender_eval(pred_df)

    # Phase 9
    generate_plots(model, test_df, features, pred_test, y_test, pred_df)

    # Summary
    print("=" * 70)
    print("PIPELINE COMPLETE - All outputs in real_data_results/")
    print("=" * 70)
    for f in sorted(OUT_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:45s} {size:>12,} bytes")
    print()


if __name__ == "__main__":
    main()
