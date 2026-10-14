"""Traffy Fondue citizen reports: Bangkok's complaint app, as ground truth for street flooding.

Satellites can't see water between buildings; residents can. Each report is a point with a time (Bangkok local),
categories, a description and usually a photo. Flood reports (category "น้ำท่วม") show where streets flooded;
reports of anything else show where people report things at all, which is what the labels need for their
"dry" class (see street_labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

FLOOD_TYPE = "น้ำท่วม"  # "flood": the category Traffy reports carry, and the API's problem_type filter


def report_row(feature: dict) -> dict | None:
    """The fields kept from one GeoJSON feature of the Traffy API (None without a usable point)."""
    p = feature.get("properties") or {}
    try:
        lon, lat = (float(v) for v in feature["geometry"]["coordinates"][:2])
    except (KeyError, TypeError, ValueError):
        return None
    types = [t for t in (p.get("problem_type_fondue") or []) if t]
    return {
        "ticket_id": p.get("ticket_id"),
        "lon": lon,
        "lat": lat,
        "time": p.get("timestamp"),  # Bangkok local time, no zone
        "types": types,
        "is_flood": FLOOD_TYPE in types,
        "state": p.get("state"),
        "district": p.get("district"),
        "has_photo": bool(p.get("photo_url")),
        "description": (p.get("description") or "")[:300],
    }


def cell_index(lon, lat):
    """(row, col, inside) on the analysis grid for arrays of points; row/col are -1 outside the study area."""
    lon, lat = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    col = np.floor((lon - C.WEST) / C.RES).astype("int64")
    row = np.floor((C.NORTH - lat) / C.RES).astype("int64")
    inside = (col >= 0) & (col < C.WIDTH) & (row >= 0) & (row < C.HEIGHT)
    return np.where(inside, row, -1), np.where(inside, col, -1), inside


def count_grid(lon, lat, shape=None) -> np.ndarray:
    """Reports per grid cell (int32), points outside the study area dropped."""
    shape = shape or (C.HEIGHT, C.WIDTH)
    row, col, inside = cell_index(lon, lat)
    grid = np.zeros(shape, dtype="int32")
    np.add.at(grid, (row[inside], col[inside]), 1)
    return grid


def storm_days(flood_times: pd.Series, rain: pd.Series | None = None, min_reports: int = None,
               spike_ratio: float = None, min_rain_mm: float = None) -> pd.DatetimeIndex:
    """Days when streets flooded city-wide: many more flood reports than usual for the time of year.

    A day counts when its flood reports are >= `min_reports` and >= `spike_ratio` x the median daily count of the
    surrounding 61 days. With `rain` (daily mm, Bangkok dates), the day must also have >= `min_rain_mm` on it or
    the day before, so a backlog of drain complaints after a quiet day doesn't count as a storm.
    """
    min_reports = C.TRAFFY_STORM_MIN_REPORTS if min_reports is None else min_reports
    spike_ratio = C.TRAFFY_STORM_SPIKE if spike_ratio is None else spike_ratio
    min_rain_mm = C.TRAFFY_STORM_MIN_RAIN_MM if min_rain_mm is None else min_rain_mm
    days = pd.to_datetime(flood_times).dt.normalize()
    if days.empty:
        return pd.DatetimeIndex([])
    daily = days.value_counts().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
    typical = daily.rolling(61, center=True, min_periods=1).median().clip(lower=1)
    storm = (daily >= min_reports) & (daily >= spike_ratio * typical)
    if rain is not None:  # days without rain data (the archive lags a few days) aren't ruled out
        r = rain.reindex(daily.index)
        wet = (r >= min_rain_mm) | (r.shift(1) >= min_rain_mm) | (r.isna() & r.shift(1).isna())
        storm &= wet.to_numpy()
    return daily.index[storm.to_numpy()]


def street_labels(n_storm_flood: np.ndarray, n_flood: np.ndarray, n_other: np.ndarray,
                  min_flood: int = None, min_other: int = None) -> np.ndarray:
    """1 flooded, 0 dry, LABEL_NODATA unknown for one year, by case-control.

    flooded  >= `min_flood` flood reports on storm days
    dry      no flood report at all, but >= `min_other` reports of other problems: people here use the app,
             so a flood would likely have been reported (this keeps "dry" in the same busy, built-up places as
             "flooded", instead of in empty farmland where nobody reports anything)
    unknown  everything else: no reports, few reports, or flood reports only on ordinary days (often chronic
             drain complaints rather than a flood that day)
    """
    min_flood = C.TRAFFY_MIN_STORM_REPORTS if min_flood is None else min_flood
    min_other = C.TRAFFY_MIN_OTHER_REPORTS if min_other is None else min_other
    labels = np.full(n_flood.shape, C.LABEL_NODATA, dtype="uint8")
    labels[(n_flood == 0) & (n_other >= min_other)] = 0
    labels[n_storm_flood >= min_flood] = 1
    return labels


def load_reports() -> pd.DataFrame | None:
    """Every flood report (scripts/fetch_traffy.py), oldest first, or None if not fetched."""
    files = report_files()
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def report_files() -> list:
    """The monthly flood-report files, oldest first."""
    return sorted(C.TRAFFY_REPORTS_DIR.glob("*.parquet")) if C.TRAFFY_REPORTS_DIR.exists() else []


def load_summary() -> dict | None:
    """Storm days and per-year label counts (scripts/build_traffy_labels.py), or None."""
    if not C.TRAFFY_STORMS_FILE.exists():
        return None
    import json
    return json.loads(C.TRAFFY_STORMS_FILE.read_text(encoding="utf-8"))


def reports_in(reports: pd.DataFrame, period: str, storms=()) -> pd.DataFrame:
    """Flood reports of a period: "YYYY" = that year's reports on storm days; "YYYY-MM-DD" = that storm day and
    the day after (reports trail the rain)."""
    day = reports["time"].dt.normalize()
    if len(period) == 4:
        return reports[(reports["time"].dt.year == int(period)) & day.isin(pd.to_datetime(list(storms)))]
    d = pd.Timestamp(period)
    return reports[day.isin([d, d + pd.Timedelta(days=1)])]
