"""
EV charger finder: "will it be free when I arrive?"

Run (in the folder that contains the data):
    pip install streamlit pandas numpy pyarrow
    streamlit run app.py

Supports two data modes:
    - Real Data: loads from real_data_results/predictions_real.parquet
    - Simulated Demo: loads from simulated_results/demo_table.parquet

Uses precomputed columns:
    pred          = predicted utilization about 1 h ahead
    lo, hi        = 90% conformal interval around pred
    next_util     = what actually happened (used only for the replay test)
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REAL_DATA_PATH = "real_data_results/predictions_real.parquet"
SIM_DATA_PATH  = "simulated_results/demo_table.parquet"
UPPER = "Predicted (90% upper bound)"

st.set_page_config(page_title="EV Charger Finder", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------- data
@st.cache_data
def load(data_path: str) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Handle station_status — may be string (simulated) or encoded int (real)
    if df["station_status"].dtype == object or df["station_status"].dtype.name == "string":
        df["offline"] = df["station_status"].astype(str).str.lower().str.contains("offline")
    else:
        # Encoded: alphabetical order -> offline=0, operational=1, partial_outage=2, under_maintenance=3
        df["offline"] = (df["station_status"] == 0)

    # Decode network if encoded as int (real data pipeline encodes categoricals)
    if df["network"].dtype in [np.int8, np.int16, np.int32, np.int64]:
        # Can't decode without mapping, so use station_name to extract network
        # or just display the code — but let's try to extract from station_name
        pass  # Network codes will display as numbers; acceptable for real data

    return df.reset_index(drop=True)


def get_snapshots(df: pd.DataFrame) -> dict:
    return {t: g for t, g in df.groupby("timestamp")}


# ----------------------------------------------------------------- logic
def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def candidates(snap: pd.DataFrame, lat: float, lon: float, k: int, max_km: float) -> pd.DataFrame:
    """K nearest working stations within max_km of the user."""
    c = snap[~snap["offline"]].copy()
    c["dist_km"] = haversine(lat, lon, c["latitude"].to_numpy(), c["longitude"].to_numpy())
    return c[c["dist_km"] <= max_km].nsmallest(k, "dist_km")


# each method returns a score per candidate; the lowest score is picked
METHODS = {
    "Nearest": lambda c, lam: c["dist_km"],
    "Currently least busy": lambda c, lam: c["utilization_rate"] + lam * c["dist_km"] / 10,
    "Predicted (point)": lambda c, lam: c["pred"] + lam * c["dist_km"] / 10,
    UPPER: lambda c, lam: c["hi"] + lam * c["dist_km"] / 10,
}


def pick(c: pd.DataFrame, method: str, lam: float) -> pd.Series:
    score = METHODS[method](c, lam).to_numpy() + 1e-4 * c["dist_km"].to_numpy()  # tiny tie-break: closer wins
    return c.iloc[int(np.argmin(score))]


def run_eval(df, snaps, n: int, k: int, max_km: float, lam: float, busy: float, seed: int = 0):
    """Replay test: random (time, location) queries; judge each pick by what really happened."""
    rng = np.random.default_rng(seed)
    times = list(snaps.keys())
    stn = df.drop_duplicates("station_id")[["latitude", "longitude"]].to_numpy()
    got = {m: [] for m in METHODS}
    dist = {m: [] for m in METHODS}
    for _ in range(n):
        t = times[rng.integers(len(times))]
        lat, lon = stn[rng.integers(len(stn))] + rng.uniform(-0.1, 0.1, 2)  # user is near a station, not on it
        c = candidates(snaps[t], lat, lon, k, max_km)
        if len(c) < 2:  # nothing to choose between
            continue
        for m in METHODS:
            row = pick(c, m, lam)
            got[m].append(row["next_util"])
            dist[m].append(row["dist_km"])
    used = len(got["Nearest"])
    if used < 2:
        return None, used
    base = np.array(got["Nearest"])
    rows = []
    for m in METHODS:
        a = np.array(got[m])
        d = a - base
        rows.append({
            "method": m,
            "avg utilization on arrival": a.mean(),
            f"% arrivals busy (>= {busy:.0%})": 100 * (a >= busy).mean(),
            "avg distance (km)": float(np.mean(dist[m])),
            "change vs nearest": d.mean(),
            "95% CI (+/-)": 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
        })
    return pd.DataFrame(rows).round(3), used


# ----------------------------------------------------------------- UI
def main():
    # ── Data mode selector ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # Check which data sources are available
        real_available = Path(REAL_DATA_PATH).exists()
        sim_available = Path(SIM_DATA_PATH).exists()

        data_options = []
        if real_available:
            data_options.append("🟢 Real Data")
        if sim_available:
            data_options.append("🔵 Simulated Demo")

        if not data_options:
            st.error("No data files found. Run the pipeline first.")
            return

        data_mode = st.radio("Data Mode", data_options, index=0)
        is_real = "Real" in data_mode

        st.divider()
        k = st.slider("Consider the K nearest stations", 3, 20, 8)
        max_km = st.slider("Max distance (km)", 5, 200, 50)
        lam = st.slider("Distance penalty (utilization points per 10 km)", 0.0, 0.2, 0.02, 0.01)
        st.caption("Utilization is used as a proxy for how hard it is to find a free port. "
                   "Stations that are offline right now are excluded.")

    # ── Header ───────────────────────────────────────────────────────
    st.title("⚡ EV charger finder")

    if is_real:
        st.success("**REAL DATA EXPERIMENT** — Using Kaggle EV charging benchmark dataset — December 2025 test period")
        data_path = REAL_DATA_PATH
    else:
        st.info("**SIMULATED DEMO** — Using simulated/demo data")
        data_path = SIM_DATA_PATH

    st.caption("Will it be free when I arrive? Predictions are for about 1 hour ahead, "
               "with a 90% interval from conformal prediction. Data is replayed from the held-out test period.")

    df = load(data_path)
    snaps = get_snapshots(df)
    times = pd.DatetimeIndex(sorted(df["timestamp"].unique()))

    # Check required columns
    required_cols = {"station_id", "timestamp", "latitude", "longitude",
                     "utilization_rate", "pred", "lo", "hi", "next_util"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    tab1, tab2 = st.tabs(["Find a charger", "Does it beat simple rules? (replay test)"])

    # ---------------- tab 1: live recommender
    with tab1:
        sel = st.slider(
            "Time now",
            min_value=times[0].to_pydatetime(),
            max_value=times[-1].to_pydatetime(),
            value=times[len(times) // 2].to_pydatetime(),
            step=timedelta(minutes=30),
            format="MMM D, HH:mm",
        )
        ts = times[times.get_indexer([pd.Timestamp(sel)], method="nearest")[0]]

        # City selector — only if city/state columns exist
        has_city = "city" in df.columns and "state" in df.columns
        if has_city:
            cities = df.groupby(["city", "state"])[["latitude", "longitude"]].mean().reset_index()
            cities["label"] = cities["city"].astype(str) + ", " + cities["state"].astype(str)
            mode = st.radio("Where are you?", ["Pick a city", "Enter coordinates"], horizontal=True)
            if mode == "Pick a city":
                label = st.selectbox("City", cities["label"])
                r = cities.loc[cities["label"] == label].iloc[0]
                lat, lon = float(r["latitude"]), float(r["longitude"])
            else:
                c1, c2 = st.columns(2)
                lat = c1.number_input("Latitude", value=float(df["latitude"].mean()), format="%.4f")
                lon = c2.number_input("Longitude", value=float(df["longitude"].mean()), format="%.4f")
        else:
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=float(df["latitude"].mean()), format="%.4f")
            lon = c2.number_input("Longitude", value=float(df["longitude"].mean()), format="%.4f")

        c = candidates(snaps[ts], lat, lon, k, max_km)
        if c.empty:
            st.warning("No working stations within that distance. Increase the max distance in the sidebar.")
        else:
            c = c.assign(score=METHODS[UPPER](c, lam)).sort_values("score")
            best = c.iloc[0]
            nearest = c.sort_values("dist_km").iloc[0]

            # Build recommendation text — handle both string and encoded network
            network_label = best.get("network", "")
            if "station_name" in c.columns:
                station_label = best["station_name"]
            else:
                station_label = best["station_id"]

            st.success(
                f"**Go to {station_label}** (network {network_label}), {best['dist_km']:.1f} km away. "
                f"Expected utilization in about 1 h: **{best['pred']:.0%}** "
                f"(90% range {best['lo']:.0%} to {best['hi']:.0%})."
            )
            if nearest["station_id"] != best["station_id"]:
                nearest_label = nearest.get("station_name", nearest["station_id"])
                st.info(
                    f"Nearest station is {nearest_label} ({nearest['dist_km']:.1f} km), "
                    f"predicted {nearest['pred']:.0%} (up to {nearest['hi']:.0%})."
                )

            # Build display table with available columns
            display_cols = ["station_id", "dist_km", "utilization_rate", "pred", "lo", "hi"]
            display_names = ["Station", "Distance (km)", "Utilization now", "Predicted +1h",
                             "Low (90%)", "High (90%)"]

            if "station_name" in c.columns:
                display_cols.insert(0, "station_name")
                display_names.insert(0, "Name")
            if "network" in c.columns:
                display_cols.insert(2 if "station_name" in c.columns else 1, "network")
                display_names.insert(2 if "station_name" in c.columns else 1, "Network")
            if "charger_type" in c.columns:
                display_cols.append("charger_type")
                display_names.append("Charger")
            if "power_output_kw" in c.columns:
                display_cols.append("power_output_kw")
                display_names.append("kW")

            show = c[[col for col in display_cols if col in c.columns]].copy()
            show.columns = display_names[:len(show.columns)]

            if st.checkbox("Reveal what actually happened (replay)"):
                show["Actual +1h"] = c["next_util"].to_numpy()
            st.dataframe(show.round(3), hide_index=True)
            st.map(c, latitude="latitude", longitude="longitude", size=200)

            # ── Forecast Timeline for Recommended / Selected Station ─────────────
            st.subheader(f"📈 Forecast Timeline — {best.get('station_name', best['station_id'])}")
            st_id = best["station_id"]
            station_series = df[df["station_id"] == st_id].sort_values("timestamp")
            
            # Show a 3-day window centered around the selected timestamp
            window_start = ts - timedelta(days=1)
            window_end = ts + timedelta(days=2)
            st_window = station_series[(station_series["timestamp"] >= window_start) & 
                                       (station_series["timestamp"] <= window_end)].copy()
            if not st_window.empty:
                chart_data = st_window.set_index("timestamp")[["pred", "lo", "hi", "utilization_rate"]].rename(
                    columns={
                        "pred": "Predicted +1h",
                        "lo": "90% Lower Bound",
                        "hi": "90% Upper Bound",
                        "utilization_rate": "Current Util",
                    }
                )
                st.line_chart(chart_data)
                st.caption(f"Displaying temporal profile for {best.get('station_name', best['station_id'])} around {ts.strftime('%b %d, %H:%M')}.")

    # ---------------- tab 2: replay evaluation
    with tab2:
        st.write("Random users at random times pick a station with each rule. "
                 "Each pick is then judged by the utilization that actually happened an hour later. "
                 "Lower is better.")
        n = st.slider("Number of simulated trips", 100, 2000, 500, 100)
        busy = st.slider("Count a station as 'busy' at utilization of", 0.5, 1.0, 0.9, 0.05)
        if st.button("Run replay test"):
            with st.spinner("Simulating..."):
                res, used = run_eval(df, snaps, n, k, max_km, lam, busy)
            if res is None:
                st.warning("Too few valid trips. Increase K or the max distance.")
            else:
                st.caption(f"{used} trips had at least 2 stations to choose from.")
                st.dataframe(res, hide_index=True)
                st.bar_chart(res.set_index("method")["avg utilization on arrival"])
                st.caption("'change vs nearest' is the average difference in arrival utilization; "
                           "negative means the rule beats picking the nearest station. "
                           "If the 95% CI includes 0, the difference is not clearly real.")


if __name__ == "__main__":
    main()
