"""Map recent flooding in Bangkok from Sentinel-1 radar (Google Earth Engine).

    python scripts/fetch_current_flood.py --gee-project YOUR_CLOUD_PROJECT_ID
    python scripts/fetch_current_flood.py --gee-project ID --since 2026-08-01 --baseline dry
    python scripts/fetch_current_flood.py --weather-only    # refresh rain + river flow only, no Earth Engine

New passes are MERGED into the existing summary.json, so the archive grows over time (--replace starts
over). In GitHub Actions, set GEE_SERVICE_ACCOUNT_KEY to a service account's JSON key instead of logging in;
--gee-project then defaults to the key's project.

Radar sees through cloud. Open water reflects the signal away from the satellite, so it looks dark.
For every recent pass, a cell is marked flooded when it is now dark AND clearly darker than a baseline
from the SAME orbit (different orbits view the ground from different angles):

    --baseline seasonal (default): the same weeks last year  -> water that is unusual for the season
    --baseline dry:                January-March this year   -> all new water, incl. planted rice paddies

Permanent water (rivers, ponds) is removed with the JRC Global Surface Water occurrence layer.

Outputs (data/processed/radar_flood/):
    YYYY-MM-DD.tif   3 bands on the analysis grid: % of cell flooded, % permanent water, % with radar data
    summary.json     dates, orbits, thresholds and flooded area per pass, plus daily rainfall and
                     Chao Phraya discharge (with the 30-year normal for each date) for context
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import PROFILE, TRANSFORM  # noqa: E402
from src import config as C  # noqa: E402
from src.data_loader import load_study_area  # noqa: E402

S1 = "COPERNICUS/S1_GRD"
JRC = "JRC/GSW1_4/GlobalSurfaceWater"
WATER_DB = -18.0        # VV backscatter below this is treated as open water
CHANGE_DB = -3.0        # ...and it must be at least this much darker than the baseline
PERMANENT_PCT = 50      # JRC occurrence at or above this = permanent water
SEASONAL_HALF_WINDOW = 21  # days either side of the same date last year
SPECKLE_RADIUS_M = 30
DOWNLOAD_SCALE_M = 20
DRY, FLOOD, PERMANENT, NODATA = 0, 1, 2, 255
BANDS = ["flood_pct", "permanent_pct", "valid_pct"]
# Daily rainfall for context (Open-Meteo weather-model estimate, free, no key); east Bangkok
RAIN_LAT, RAIN_LON = 13.80, 100.75
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
# Daily Chao Phraya discharge (GloFAS via the Open-Meteo Flood API, free, no key). This ~5 km cell sits on the
# main channel at Nonthaburi; nearby cells snap to small side streams carrying only a few m³/s.
RIVER_LAT, RIVER_LON = 13.825, 100.475
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"
RIVER_NORMAL_YEARS = (1996, 2025)  # the "normal" for each calendar day is the median over these years
WEATHER_LEAD_DAYS = 7  # weather starts this long before the first pass, so it has its run-up rain
KEY_ENV = "GEE_SERVICE_ACCOUNT_KEY"  # service account JSON key (GitHub Actions secret)


def init_earth_engine(project: str | None):
    """Log in with the service account key in $GEE_SERVICE_ACCOUNT_KEY if set, else the user's own login."""
    import ee

    key = os.environ.get(KEY_ENV)
    if key:
        info = json.loads(key)
        ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], key_data=key),
                      project=project or info["project_id"])
        print(f"Earth Engine: service account {info['client_email']}")
    else:
        ee.Initialize(project=project)
    return ee


def merge_by_date(old: list, new: list) -> list:
    """Records from both lists, one per date (new wins), sorted by date. Used for passes and daily weather."""
    merged = {r["date"]: r for r in old}
    merged.update({r["date"]: r for r in new})
    return [merged[d] for d in sorted(merged)]


def load_existing_summary(mode: str) -> dict:
    """The current summary.json, or {} if there is none or it used a different baseline (not comparable)."""
    if not C.RADAR_SUMMARY_FILE.exists():
        return {}
    old = json.loads(C.RADAR_SUMMARY_FILE.read_text())
    if old.get("method", {}).get("baseline") != mode:
        print(f"  existing summary uses baseline {old.get('method', {}).get('baseline')!r}, not {mode!r}: "
              "starting a new archive")
        return {}
    return old


def s1_collection(ee, region):
    return (ee.ImageCollection(S1).filterBounds(region)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select("VV"))


def list_passes(ee, region, since, until):
    """Distinct (date, direction, orbit) passes in the date range."""
    info = (s1_collection(ee, region).filterDate(since, until)
            .reduceColumns(ee.Reducer.toList(3),
                           ["system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start"])
            .get("list").getInfo())
    passes = {(dt.datetime.utcfromtimestamp(t / 1000).date(), p, int(o)) for t, p, o in info}
    return sorted(passes)


def baseline_window(date: dt.date, mode: str):
    if mode == "dry":
        return dt.date(date.year, 1, 1), dt.date(date.year, 4, 1)
    last_year = date - dt.timedelta(days=365)
    return (last_year - dt.timedelta(days=SEASONAL_HALF_WINDOW),
            last_year + dt.timedelta(days=SEASONAL_HALF_WINDOW + 1))


def classify(ee, region, date, direction, orbit, mode):
    """Server-side flood classification for one pass. Returns (image, number of baseline scenes)."""
    same_orbit = (s1_collection(ee, region)
                  .filter(ee.Filter.eq("orbitProperties_pass", direction))
                  .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)))
    scenes = same_orbit.filterDate(str(date), str(date + dt.timedelta(days=1)))
    proj = scenes.first().projection()  # keep native 10 m so the speckle filter works in metres
    now = scenes.mosaic().setDefaultProjection(proj).focalMedian(SPECKLE_RADIUS_M, "circle", "meters")
    b0, b1 = baseline_window(date, mode)
    base_col = same_orbit.filterDate(str(b0), str(b1))
    base = base_col.median().setDefaultProjection(proj).focalMedian(SPECKLE_RADIUS_M, "circle", "meters")

    flooded = now.lt(WATER_DB).And(now.subtract(base).lt(CHANGE_DB))
    permanent = ee.Image(JRC).select("occurrence").unmask(0).gte(PERMANENT_PCT)
    valid = now.mask().And(base.mask())
    code = (ee.Image(DRY).where(flooded, FLOOD).where(permanent, PERMANENT)
            .where(valid.Not(), NODATA).unmask(NODATA).toUint8().clip(region))
    return code, base_col.size().getInfo(), (b0, b1)


def fetch_rainfall(since: str) -> list:
    past_days = min(92, (dt.date.today() - dt.date.fromisoformat(since)).days + 1)
    try:
        r = requests.get(OPEN_METEO, timeout=60, params={
            "latitude": RAIN_LAT, "longitude": RAIN_LON, "daily": "precipitation_sum",
            "past_days": past_days, "forecast_days": 1, "timezone": "Asia/Bangkok"})
        r.raise_for_status()
        d = r.json()["daily"]
        return [{"date": t, "mm": p} for t, p in zip(d["time"], d["precipitation_sum"])
                if t >= since and p is not None]
    except Exception as e:  # rainfall is context only; never block the flood maps
        print(f"  rainfall unavailable: {e}")
        return []


def fetch_river_discharge(since: str) -> list:
    """Daily discharge (m³/s) since `since`, with the median for the same calendar day over RIVER_NORMAL_YEARS."""
    point = {"latitude": RIVER_LAT, "longitude": RIVER_LON, "daily": "river_discharge",
             "cell_selection": "nearest"}
    try:
        r = requests.get(OPEN_METEO_FLOOD, timeout=60, params={
            **point, "start_date": since, "end_date": str(dt.date.today())})
        r.raise_for_status()
        recent = pd.DataFrame(r.json()["daily"]).dropna()
        y0, y1 = RIVER_NORMAL_YEARS
        r = requests.get(OPEN_METEO_FLOOD, timeout=180, params={
            **point, "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31"})
        r.raise_for_status()
        hist = pd.DataFrame(r.json()["daily"]).dropna()
        # Median per calendar day, smoothed over a week so one odd year doesn't make the normal jagged
        normal = hist.groupby(hist["time"].str[5:])["river_discharge"].median()
        normal = normal.rolling(7, center=True, min_periods=1).mean()
        return [{"date": t, "m3s": q, "normal_m3s": round(float(normal.get(t[5:], np.nan)), 1)}
                for t, q in zip(recent["time"], recent["river_discharge"])]
    except Exception as e:  # river flow is context only; never block the flood maps
        print(f"  river discharge unavailable: {e}")
        return []


def fetch_weather(since: str, old: dict = None) -> dict:
    """Rainfall and river discharge from WEATHER_LEAD_DAYS before `since`, as summary.json fields.

    Merged into the weather already in `old` (a previous summary): the rainfall source only reaches back
    92 days, so older days are kept from earlier runs, and a failed download keeps what was there.
    """
    old = old or {}
    start = str(dt.date.fromisoformat(since) - dt.timedelta(days=WEATHER_LEAD_DAYS))
    rain = merge_by_date(old.get("rainfall", []), fetch_rainfall(start))
    river = merge_by_date(old.get("river", []), fetch_river_discharge(start))
    print(f"  weather: {len(rain)} days of rainfall, {len(river)} days of river discharge")
    return {
        "rainfall": rain,
        "rainfall_source": f"Open-Meteo weather-model estimate at {RAIN_LAT}N {RAIN_LON}E",
        "river": river,
        "river_source": (f"GloFAS river model via the Open-Meteo Flood API, Chao Phraya at Nonthaburi "
                         f"({RIVER_LAT}N {RIVER_LON}E); normal = {RIVER_NORMAL_YEARS[0]}-{RIVER_NORMAL_YEARS[1]} "
                         f"median for the date"),
    }


def update_weather_only():
    """Refresh the weather in an existing summary.json without touching Earth Engine."""
    summary = json.loads(C.RADAR_SUMMARY_FILE.read_text())
    first_pass = min(p["date"] for p in summary["passes"])
    summary.update(fetch_weather(first_pass, summary))
    C.RADAR_SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"Updated weather in {C.RADAR_SUMMARY_FILE}")


def download(url: str, dest: Path):
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    dest.write_bytes(r.content)


def to_grid(raw_path: Path) -> np.ndarray:
    """Aggregate the 20 m class raster to % flooded / % permanent / % valid per analysis cell."""
    with rasterio.open(raw_path) as src:
        codes = src.read(1)
        src_transform, src_crs = src.transform, src.crs
    out = []
    for mask in (codes == FLOOD, codes == PERMANENT, codes != NODATA):
        grid = np.zeros((C.HEIGHT, C.WIDTH), dtype="float32")
        reproject(source=(mask * 100).astype("uint8"), destination=grid,
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=TRANSFORM, dst_crs=C.CRS, resampling=Resampling.average)
        out.append(grid)
    return np.stack(out)


def main(project: str | None, since: str, until: str, mode: str, refresh: bool, replace: bool = False):
    ee = init_earth_engine(project)
    old = {} if replace else load_existing_summary(mode)
    region = ee.Geometry.Rectangle([C.WEST, C.SOUTH, C.EAST, C.NORTH])
    C.RADAR_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = C.RAW_DIR / "radar"
    raw_dir.mkdir(parents=True, exist_ok=True)
    area = load_study_area()
    bkk = area.district_ids > 0

    passes = list_passes(ee, region, since, until)
    print(f"{len(passes)} radar passes between {since} and {until} (baseline: {mode})")
    summary = []
    for date, direction, orbit in passes:
        out_path = C.RADAR_DIR / f"{date}.tif"
        raw_path = raw_dir / f"{date}_{direction[:3].lower()}{orbit}_{mode}.tif"
        code, n_base, (b0, b1) = classify(ee, region, date, direction, orbit, mode)
        if n_base == 0:
            print(f"  {date} {direction} {orbit}: no baseline scenes, skipped")
            continue
        if refresh or not raw_path.exists():
            url = code.getDownloadURL({"region": region, "scale": DOWNLOAD_SCALE_M,
                                       "crs": "EPSG:4326", "format": "GEO_TIFF"})
            download(url, raw_path)
        grid = to_grid(raw_path)
        with rasterio.open(out_path, "w", dtype="float32", **{**PROFILE, "count": 3}) as dst:
            for i, (name, band) in enumerate(zip(BANDS, grid), start=1):
                dst.write(band, i)
                dst.set_band_description(i, name)

        flood_km2 = grid[0] / 100 * area.cell_area_km2
        valid = grid[2] >= 50
        rec = {
            "date": str(date), "direction": direction, "orbit": orbit,
            "baseline": mode, "baseline_from": str(b0), "baseline_to": str(b1), "baseline_scenes": n_base,
            "flooded_km2": float(flood_km2[valid].sum()),
            "flooded_km2_bangkok": float(flood_km2[valid & bkk].sum()),
            "coverage_pct": float(valid.mean() * 100),
            "coverage_pct_bangkok": float(valid[bkk].mean() * 100),
        }
        summary.append(rec)
        print(f"  {date} {direction[:4]} orbit {orbit}: {rec['flooded_km2']:7.1f} km² flooded "
              f"({rec['flooded_km2_bangkok']:.1f} in Bangkok), coverage {rec['coverage_pct']:.0f}%, "
              f"baseline {n_base} scenes")

    all_passes = merge_by_date(old.get("passes", []), summary)
    C.RADAR_SUMMARY_FILE.write_text(json.dumps({
        "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": {"water_db": WATER_DB, "change_db": CHANGE_DB, "permanent_pct": PERMANENT_PCT,
                   "baseline": mode, "speckle_radius_m": SPECKLE_RADIUS_M, "scale_m": DOWNLOAD_SCALE_M},
        "passes": all_passes,
        **fetch_weather(all_passes[0]["date"] if all_passes else since, old),
    }, indent=2))
    print(f"Wrote {len(summary)} dates to {C.RADAR_DIR}; the archive now has {len(all_passes)} passes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gee-project", help="Google Cloud project id registered for Earth Engine "
                                          f"(default with ${KEY_ENV}: the key's project)")
    ap.add_argument("--since", default=str(dt.date.today() - dt.timedelta(days=45)))
    ap.add_argument("--until", default=str(dt.date.today() + dt.timedelta(days=1)))
    ap.add_argument("--baseline", choices=["seasonal", "dry"], default="seasonal")
    ap.add_argument("--refresh", action="store_true", help="re-download dates already fetched")
    ap.add_argument("--weather-only", action="store_true",
                    help="only refresh rainfall and river discharge in the existing summary.json")
    ap.add_argument("--replace", action="store_true",
                    help="start a new archive instead of merging into the existing summary.json")
    args = ap.parse_args()
    if args.weather_only:
        update_weather_only()
    elif not (args.gee_project or os.environ.get(KEY_ENV)):
        ap.error(f"--gee-project is required (unless --weather-only or ${KEY_ENV} is set)")
    else:
        main(args.gee_project, args.since, args.until, args.baseline, args.refresh, args.replace)
