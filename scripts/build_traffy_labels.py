"""Street-flood labels from Traffy Fondue citizen reports, one map per year, for the street-flood model.

    python scripts/build_traffy_labels.py

Needs scripts/fetch_traffy.py --all (flood reports, and every other report for the "dry" class). The daily
Update data workflow runs it after fetch_traffy.py --daily.

1. Storm days: days when far more flood reports came in than usual for the time of year, with rain on the day or
   the day before (src/traffy.storm_days; thresholds in src/config.py). Most flood reports on ordinary days are
   long-running drain complaints, not a flood that day.
2. Labels per year, by case-control (src/traffy.street_labels):
       flooded (1)    a flood report on a storm day
       dry (0)        no flood report all year, but at least TRAFFY_MIN_OTHER_REPORTS reports of other problems:
                      people here use the app, so they would likely have reported a flood
       unknown (255)  everything else, including all cells where nobody reports anything
   Choosing "dry" cells among places where people report keeps the two classes in the same kind of busy,
   built-up place, so the model can't just learn "where people live = where reports come from".

Rain: Open-Meteo historical weather archive (ERA5, no key), daily, at 12 points over Bangkok; a day's rain is the
wettest point's. ERA5 smooths out cloudbursts, so the threshold is low; it only rules out dry days.

Outputs:
    data/processed/flood_labels/traffy_YYYY.tif   1 flooded, 0 dry, 255 unknown, like the radar labels
    data/processed/traffy/summary.json            storm days with report counts and rain, label counts per year
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import PROFILE  # noqa: E402
from src import config as C  # noqa: E402
from src.traffy import TEXT_MODEL, count_grid, load_reports, storm_days, street_labels  # noqa: E402

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
RAIN_LATS = (13.62, 13.76, 13.90)
RAIN_LONS = (100.40, 100.55, 100.70, 100.85)


def city_rain(start: str, end: str) -> pd.Series:
    """Daily rain (mm) at the wettest of the RAIN_LATS x RAIN_LONS points, Bangkok dates (to yesterday at most:
    the archive ends there)."""
    end = min(pd.Timestamp(end), pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1))
    end = f"{end:%Y-%m-%d}"
    lats = [la for la in RAIN_LATS for _ in RAIN_LONS]
    lons = [lo for _ in RAIN_LATS for lo in RAIN_LONS]
    r = requests.get(OPEN_METEO_ARCHIVE, timeout=180, params={
        "latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
        "daily": "precipitation_sum", "timezone": "Asia/Bangkok", "start_date": start, "end_date": end})
    r.raise_for_status()
    points = r.json()
    points = points if isinstance(points, list) else [points]
    df = pd.DataFrame({i: pd.Series(p["daily"]["precipitation_sum"], index=pd.to_datetime(p["daily"]["time"]),
                                    dtype="float64") for i, p in enumerate(points)})
    return df.max(axis=1, skipna=True).where(df.notna().any(axis=1))


def load_other_counts() -> dict:
    """{year: per-cell count of non-flood reports} from fetch_traffy.py --all."""
    if not C.TRAFFY_OTHER_FILE.exists():
        raise SystemExit(f"{C.TRAFFY_OTHER_FILE} not found: run scripts/fetch_traffy.py --all first")
    with rasterio.open(C.TRAFFY_OTHER_FILE) as src:
        return {int(src.descriptions[i - 1]): src.read(i).astype("int32") for i in range(1, src.count + 1)}


def main():
    reports = load_reports()
    if reports is None:
        raise SystemExit(f"No flood reports in {C.TRAFFY_REPORTS_DIR}: run scripts/fetch_traffy.py first")
    other = load_other_counts()
    rain = city_rain(f"{reports['time'].min():%Y-%m-%d}", f"{reports['time'].max():%Y-%m-%d}")
    storms = storm_days(reports["time"], rain)
    day = reports["time"].dt.normalize()
    on_storm = day.isin(storms).to_numpy()
    print(f"{len(reports):,} flood reports, {len(storms)} storm days holding {on_storm.sum():,} of them "
          f"({on_storm.mean():.0%})")

    C.FLOOD_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    years = []
    for y in C.TRAFFY_YEARS:
        if y not in other:
            print(f"  {y}: no complete year of other reports, skipped")
            continue
        in_year = (reports["time"].dt.year == y).to_numpy()
        r_all, r_storm = reports[in_year], reports[in_year & on_storm]
        labels = street_labels(count_grid(r_storm["lon"], r_storm["lat"]),
                               count_grid(r_all["lon"], r_all["lat"]), other[y])
        out = C.FLOOD_LABELS_DIR / f"traffy_{y}.tif"
        with rasterio.open(out, "w", dtype="uint8", nodata=C.LABEL_NODATA, **PROFILE) as dst:
            dst.write(labels, 1)
        n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
        by_model = int((r_all["source"] == TEXT_MODEL).sum()) if "source" in r_all else 0
        years.append({"year": y, "flood_reports": int(in_year.sum()), "flood_reports_text_model": by_model,
                      "storm_flood_reports": len(r_storm),
                      "storm_days": int(sum(d.year == y for d in storms)), "other_reports": int(other[y].sum()),
                      "flooded_cells": n1, "dry_cells": n0, "test_year": y >= C.TEST_YEARS_FROM})
        print(f"  {y}: {years[-1]['storm_days']} storm days, {len(r_storm):,} storm flood reports -> "
              f"{n1:,} flooded and {n0:,} dry cells ({n1 / max(n1 + n0, 1):.1%} flooded)"
              f"{'  [test year]' if y >= C.TEST_YEARS_FROM else ''}")

    daily = day.value_counts()
    C.TRAFFY_STORMS_FILE.write_text(json.dumps({
        "source": "Traffy Fondue public API (publicapi.traffy.in.th), category 'น้ำท่วม' (flood); since 14 Oct 2026, "
                  "when Traffy stopped publishing categories, flood reports recognised from their text "
                  "(scripts/train_traffy_text.py; flood_reports_text_model per year)",
        "rain_source": f"Open-Meteo historical weather archive (ERA5), daily, wettest of {len(RAIN_LATS) * len(RAIN_LONS)}"
                       f" points over Bangkok",
        "rules": {"storm_min_reports": C.TRAFFY_STORM_MIN_REPORTS, "storm_spike": C.TRAFFY_STORM_SPIKE,
                  "storm_min_rain_mm": C.TRAFFY_STORM_MIN_RAIN_MM,
                  "flooded_min_storm_reports": C.TRAFFY_MIN_STORM_REPORTS,
                  "dry_min_other_reports": C.TRAFFY_MIN_OTHER_REPORTS},
        "storms": [{"date": str(d.date()), "flood_reports": int(daily.get(d, 0)),
                    "rain_mm": None if pd.isna(rain.get(d)) else round(float(rain.get(d)), 1),
                    "rain_day_before_mm": None if pd.isna(rain.get(d - pd.Timedelta(days=1)))
                    else round(float(rain.get(d - pd.Timedelta(days=1))), 1)} for d in storms],
        "years": years,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(years)} label files to {C.FLOOD_LABELS_DIR} and {C.TRAFFY_STORMS_FILE}")


if __name__ == "__main__":
    main()
