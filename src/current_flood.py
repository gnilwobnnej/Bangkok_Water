"""Load the Sentinel-1 radar flood maps (scripts/fetch_current_flood.py, scripts/evaluate_current.py)."""
import json

import numpy as np
import pandas as pd
import rasterio

from src import config as C
from src.data_loader import StudyArea

# A cell counts as flooded when at least this % of it is flooded on a pass, and as dry when under DRY_PCT
FLOODED_PCT, DRY_PCT = 50, 10


def radar_available() -> bool:
    return C.RADAR_SUMMARY_FILE.exists()


def load_summary():
    """(passes, rainfall, river discharge DataFrames, full summary dict), or None if not fetched yet.

    passes gets each pass date's weather from pass_weather().
    """
    if not radar_available():
        return None
    summary = json.loads(C.RADAR_SUMMARY_FILE.read_text())
    rain = pd.DataFrame(summary.get("rainfall", []), columns=["date", "mm"])
    river = pd.DataFrame(summary.get("river", []), columns=["date", "m3s", "normal_m3s"])
    passes = pass_weather(pd.DataFrame(summary["passes"]), rain, river)
    return passes, rain, river, summary


def pass_weather(passes: pd.DataFrame, rain: pd.DataFrame, river: pd.DataFrame) -> pd.DataFrame:
    """Add rain on the pass date, rain over that day and the 2 before, and river discharge (NaN where missing)."""
    out = passes.copy()
    daily = rain.set_index(pd.to_datetime(rain["date"]))["mm"].astype(float)
    dates = pd.to_datetime(out["date"])
    out["rain_day_mm"] = daily.reindex(dates).to_numpy()
    out["rain_3day_mm"] = [daily.loc[d - pd.Timedelta(days=2):d].sum(min_count=3) for d in dates]
    flow = river.set_index("date")
    out["discharge_m3s"] = flow["m3s"].reindex(out["date"]).to_numpy(dtype=float)
    normal = flow["normal_m3s"].reindex(out["date"]).to_numpy(dtype=float)
    out["discharge_pct_normal"] = out["discharge_m3s"] / normal * 100
    return out


def conditions_alert(passes: pd.DataFrame, rain: pd.DataFrame, river: pd.DataFrame, today=None) -> list:
    """Plain-language warnings when recent conditions are unusual (thresholds in src/config.py); [] if none.

    Judged on the latest data: the newest radar pass (if recent), the last 3 days of rain and the latest
    river flow. `today` defaults to the newest date in the weather data (the data's "now").
    """
    if today is None:
        newest = [df["date"].max() for df in (rain, river) if len(df)]
        today = max(newest) if newest else pd.Timestamp.now().normalize()
    now = pd.Timestamp(today)
    alerts = []

    if len(passes) > C.ALERT_MIN_EARLIER_PASSES:
        ordered = passes.sort_values("date")
        latest, earlier = ordered.iloc[-1], ordered.iloc[:-1]["flooded_km2"]
        typical = earlier.median()
        recent = (now - pd.Timestamp(latest["date"])).days <= C.ALERT_PASS_MAX_AGE_DAYS
        if recent and typical > 0 and latest["flooded_km2"] >= C.ALERT_FLOOD_RATIO * typical:
            alerts.append(f"Radar on {latest['date']} found **{latest['flooded_km2']:.0f} km²** of unusual water, "
                          f"{latest['flooded_km2'] / typical:.1f}× the typical {typical:.0f} km² of earlier passes.")

    if len(river):
        last = river.dropna(subset=["m3s", "normal_m3s"]).sort_values("date").tail(1)
        if len(last) and last["normal_m3s"].iloc[0] > 0:
            pct = last["m3s"].iloc[0] / last["normal_m3s"].iloc[0] * 100
            if pct >= C.ALERT_RIVER_PCT:
                alerts.append(f"The Chao Phraya is running at **{pct:.0f}% of normal** for the date "
                              f"({last['date'].iloc[0]}, modelled flow).")

    if len(rain):
        daily = rain.set_index(pd.to_datetime(rain["date"]))["mm"].astype(float).sort_index()
        window = daily.loc[now - pd.Timedelta(days=2):now]
        if window.notna().sum() == 3 and window.sum() >= C.ALERT_RAIN_3DAY_MM:
            alerts.append(f"**{window.sum():.0f} mm of rain** fell in the 3 days to {now.date()} (weather-model estimate).")

    return alerts


def load_evaluation():
    """Per-date scores of each method against the radar maps, or None."""
    if not C.RADAR_EVAL_FILE.exists():
        return None
    ev = json.loads(C.RADAR_EVAL_FILE.read_text())
    rows = [{"date": r["date"], "scope": r["scope"], "flooded_share": r["flooded_share"],
             "method": m, **scores}
            for r in ev["results"] for m, scores in r["methods"].items()]
    return pd.DataFrame(rows)


def load_pass(date: str) -> dict:
    """% flooded, % permanent water and % with radar data per cell for one pass date."""
    with rasterio.open(C.RADAR_DIR / f"{date}.tif") as src:
        return {"flood_pct": src.read(1), "permanent_pct": src.read(2), "valid_pct": src.read(3)}


def radar_labels(radar: dict) -> np.ndarray:
    """1 flooded, 0 dry, -1 unknown (radar didn't see it, permanent water, or partly flooded)."""
    labels = np.full(radar["flood_pct"].shape, -1, dtype="int8")
    usable = (radar["valid_pct"] >= 50) & (radar["permanent_pct"] < 50)
    labels[usable & (radar["flood_pct"] < DRY_PCT)] = 0
    labels[usable & (radar["flood_pct"] >= FLOODED_PCT)] = 1
    return labels


def district_flooded_pct(area: StudyArea, radar: dict) -> pd.DataFrame:
    """Share of each district's radar-visible land flooded on that pass."""
    ids = area.district_ids.ravel()
    n = int(ids.max()) + 1
    cell = area.cell_area_km2.ravel()
    seen = cell * (radar["valid_pct"].ravel() / 100)
    flooded = cell * (radar["flood_pct"].ravel() / 100)
    seen_km2 = np.bincount(ids, weights=seen, minlength=n)
    flooded_km2 = np.bincount(ids, weights=flooded, minlength=n)
    pct = np.divide(flooded_km2, seen_km2, out=np.zeros(n), where=seen_km2 > 0) * 100
    return pd.DataFrame({"district_id": np.arange(1, n), "flooded_now_pct": pct[1:],
                         "flooded_now_km2": flooded_km2[1:]})
