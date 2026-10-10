"""Rainfall and river context for each training flood: how wet was each season, and how high was the river?

    python scripts/event_context.py

For 2011 and each radar wet season (2017-2025), over August-November:
    rain_total_mm            total rain, and as % of the 1996-2025 median for August-November
    rain_max_3day_mm         wettest 3 days
    river_peak_m3s           highest Chao Phraya flow at Nonthaburi, its date, and its % of normal for that date
    river_days_above_alert   days at or above config.ALERT_RIVER_PCT of normal

This is context for reading the scores, not model input: the ML model predicts *where* it floods, from the
place alone. Sources are free and need no key: rain from the Open-Meteo historical weather archive (ERA5
reanalysis, ~25 km, at the same east-Bangkok point the app uses), river flow from GloFAS via the Open-Meteo Flood
API (the same Nonthaburi point). Both are model estimates, not gauge readings. The GloFAS flow reads high on the
Chao Phraya, so compare it with its own normal.

Output: data/processed/flood_labels/context.json (scripts/train_model.py adds it to the report's events)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C  # noqa: E402
from fetch_current_flood import (  # noqa: E402
    OPEN_METEO_FLOOD, RAIN_LAT, RAIN_LON, RIVER_LAT, RIVER_LON, RIVER_NORMAL_YEARS,
)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
SEASON = ("08-01", "11-30")
CONTEXT_FILE = C.FLOOD_LABELS_DIR / "context.json"


def event_years() -> list:
    return [2011] + [y for y in C.ARCHIVE_YEARS if y < C.TEST_YEARS_FROM]


def daily(url: str, params: dict, var: str, start: str, end: str) -> pd.Series:
    r = requests.get(url, timeout=180, params={**params, "daily": var, "start_date": start, "end_date": end})
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.Series(d[var], index=pd.to_datetime(d["time"]), dtype="float64").dropna()


def season(s: pd.Series, year: int) -> pd.Series:
    return s[f"{year}-{SEASON[0]}":f"{year}-{SEASON[1]}"]


def year_context(year: int, rain: pd.Series, river: pd.Series, river_normal: pd.Series, rain_normal: float) -> dict:
    r = season(rain, year)
    q = season(river, year)
    normal = river_normal.reindex(q.index.strftime("%m-%d")).to_numpy()
    pct = 100 * q.to_numpy() / normal
    peak = int(np.argmax(q.to_numpy()))
    return {
        "year": year,
        "rain_total_mm": round(float(r.sum()), 0),
        "rain_pct_normal": round(float(100 * r.sum() / rain_normal), 0),
        "rain_max_3day_mm": round(float(r.rolling(3).sum().max()), 0),
        "river_peak_m3s": round(float(q.iloc[peak]), 0),
        "river_peak_date": str(q.index[peak].date()),
        "river_peak_pct_normal": round(float(pct[peak]), 0),
        "river_days_above_alert": int((pct >= C.ALERT_RIVER_PCT).sum()),
    }


def main():
    y0, y1 = RIVER_NORMAL_YEARS
    years = event_years()
    start, end = f"{min(y0, years[0])}-01-01", f"{max(y1, years[-1])}-12-31"
    rain = daily(OPEN_METEO_ARCHIVE, {"latitude": RAIN_LAT, "longitude": RAIN_LON, "timezone": "Asia/Bangkok"},
                 "precipitation_sum", start, end)
    river = daily(OPEN_METEO_FLOOD, {"latitude": RIVER_LAT, "longitude": RIVER_LON, "cell_selection": "nearest"},
                  "river_discharge", start, end)
    # Normals over the same years as the app's river normal: median per calendar day (smoothed over a week),
    # and the median August-November rain total
    base = river[str(y0):str(y1)]
    river_normal = base.groupby(base.index.strftime("%m-%d")).median().rolling(7, center=True, min_periods=1).mean()
    rain_normal = float(np.median([season(rain, y).sum() for y in range(y0, y1 + 1)]))

    rows = [year_context(y, rain, river, river_normal, rain_normal) for y in years]
    CONTEXT_FILE.write_text(json.dumps({
        "season": "August-November",
        "rain_source": f"Open-Meteo historical weather archive (ERA5 reanalysis) at {RAIN_LAT}N {RAIN_LON}E",
        "river_source": f"GloFAS river model via the Open-Meteo Flood API, Chao Phraya at Nonthaburi "
                        f"({RIVER_LAT}N {RIVER_LON}E)",
        "normal_years": [y0, y1],
        "rain_normal_mm": round(rain_normal, 0),
        "alert_river_pct": C.ALERT_RIVER_PCT,
        "years": rows,
    }, indent=2))
    print(f"Aug-Nov normal rain {rain_normal:.0f} mm")
    for r in rows:
        print(f"  {r['year']}: rain {r['rain_total_mm']:5.0f} mm ({r['rain_pct_normal']:3.0f}%), wettest 3 days "
              f"{r['rain_max_3day_mm']:3.0f} mm; river peak {r['river_peak_m3s']:6.0f} m³/s on {r['river_peak_date']} "
              f"({r['river_peak_pct_normal']:3.0f}% of normal), {r['river_days_above_alert']} days >= "
              f"{C.ALERT_RIVER_PCT}%")
    print(f"Wrote {CONTEXT_FILE}")


if __name__ == "__main__":
    main()
