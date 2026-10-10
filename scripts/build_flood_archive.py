"""Build one flood map per wet season (2017-2025) from Sentinel-1 radar, as extra training floods for the ML model.

    python scripts/build_flood_archive.py --gee-project YOUR_CLOUD_PROJECT_ID
    python scripts/build_flood_archive.py --gee-project ID --years 2017 2022

Every August-November pass is classified exactly as in scripts/fetch_current_flood.py (seasonal baseline:
the same weeks of the year before, so the map shows water that is unusual for the season, not the yearly
rice-paddy cycle). Earth Engine averages each pass onto the analysis grid and returns three 0/1 bands per
cell (summing a whole month in Earth Engine runs out of memory, so passes are downloaded one by one, ~40 KB
each, and added up here):

    seen      radar data on >= 50% of the cell, and < 50% permanent water
    flooded   seen, and >= 50% of the cell flooded
    dry       seen, and < 10% of the cell flooded

A season's label is then:
    flooded (1)   flooded on at least --min-flood-passes passes (default 2, so one odd pass isn't enough)
    dry (0)       seen at least once and dry every time it was seen
    unknown (255) everything else (never seen, permanent water, partly flooded, flooded only once)

The cell thresholds are the ones evaluate_current.py uses (src/current_flood.py FLOODED_PCT, DRY_PCT).

Outputs:
    data/raw/flood_archive/YYYY/<date>_<orbit>.tif  one pass (cached; --refresh downloads again)
    data/processed/flood_labels/radar_YYYY.tif      1 flooded, 0 dry, 255 unknown, like flood_2011.tif
    data/processed/flood_labels/summary.json        passes used and label counts per year
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_current_flood import (FLOOD, KEY_ENV, NODATA, PERMANENT, classify, download,  # noqa: E402
                                 init_earth_engine, list_passes)
from prepare_data import PROFILE  # noqa: E402
from src import config as C  # noqa: E402
from src.current_flood import DRY_PCT, FLOODED_PCT  # noqa: E402

SEASON = ((8, 1), (12, 1))  # August-November: the end of the wet season, when Bangkok floods
MIN_FLOOD_PASSES = 2
COUNT_BANDS = ["seen", "flooded", "dry"]
WORK_SCALE_M = 20  # classify at this scale before averaging to the grid, as fetch_current_flood.py downloads
GRID_TRANSFORM = [C.RES, 0, C.WEST, 0, -C.RES, C.NORTH]
RAW_DIR = C.RAW_DIR / "flood_archive"


def season_labels(n_seen: np.ndarray, n_flooded: np.ndarray, n_dry: np.ndarray,
                  min_flood_passes: int = MIN_FLOOD_PASSES) -> np.ndarray:
    """1 flooded, 0 dry, LABEL_NODATA unknown, from the per-cell pass counts of one season."""
    labels = np.full(n_seen.shape, C.LABEL_NODATA, dtype="uint8")
    labels[(n_seen > 0) & (n_dry == n_seen)] = 0
    labels[n_flooded >= min_flood_passes] = 1
    return labels


def grid_fraction(ee, mask):
    """% of each analysis cell where `mask` is 1, computed in Earth Engine on the exact app grid."""
    grid = ee.Projection(C.CRS, GRID_TRANSFORM)
    return (mask.reproject(crs=C.CRS, scale=WORK_SCALE_M)
            .reduceResolution(ee.Reducer.mean(), maxPixels=1024)
            .reproject(grid).multiply(100))


def pass_counts(ee, code):
    """seen / flooded / dry 0/1 bands on the grid for one classified pass."""
    flood = grid_fraction(ee, code.eq(FLOOD))
    perm = grid_fraction(ee, code.eq(PERMANENT))
    valid = grid_fraction(ee, code.neq(NODATA))
    seen = valid.gte(50).And(perm.lt(50))
    return (ee.Image.cat(seen, seen.And(flood.gte(FLOODED_PCT)), seen.And(flood.lt(DRY_PCT)))
            .rename(COUNT_BANDS).toUint8())


def read_counts(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read()
        if data.shape != (3, C.HEIGHT, C.WIDTH) or not src.transform.almost_equals(PROFILE["transform"]):
            raise ValueError(f"{path} is not on the analysis grid ({data.shape}, {src.transform})")
    return data.astype("int32")


def build_year(ee, region, year: int, refresh: bool, min_flood_passes: int) -> dict:
    (m0, d0), (m1, d1) = SEASON
    raw = RAW_DIR / str(year)
    raw.mkdir(parents=True, exist_ok=True)
    counts = np.zeros((3, C.HEIGHT, C.WIDTH), dtype="int32")
    used = []
    for date, direction, orbit in list_passes(ee, region, str(dt.date(year, m0, d0)), str(dt.date(year, m1, d1))):
        dest = raw / f"{date}_{direction[:3].lower()}{orbit}.tif"
        skipped = dest.with_suffix(".skip")  # marks passes with no baseline, so reruns don't ask again
        if skipped.exists() and not refresh:
            continue
        if refresh or not dest.exists():
            code, n_base, _ = classify(ee, region, date, direction, orbit, "seasonal")
            if n_base == 0:
                print(f"    {date} {direction} {orbit}: no baseline scenes, skipped")
                skipped.touch()
                continue
            url = pass_counts(ee, code).getDownloadURL({
                "crs": C.CRS, "crs_transform": GRID_TRANSFORM, "dimensions": f"{C.WIDTH}x{C.HEIGHT}",
                "format": "GEO_TIFF"})
            download(url, dest)
        counts += read_counts(dest)
        used.append({"date": str(date), "direction": direction, "orbit": orbit})
    print(f"  {len(used)} passes")

    labels = season_labels(*counts, min_flood_passes=min_flood_passes)
    out = C.FLOOD_LABELS_DIR / f"radar_{year}.tif"
    with rasterio.open(out, "w", dtype="uint8", nodata=C.LABEL_NODATA, **PROFILE) as dst:
        dst.write(labels, 1)
    n_f, n_d = int((labels == 1).sum()), int((labels == 0).sum())
    rec = {"year": year, "file": out.name, "n_passes": len(used), "flooded_cells": n_f, "dry_cells": n_d,
           "unknown_cells": int(labels.size - n_f - n_d), "flooded_share": n_f / max(n_f + n_d, 1),
           "passes": used}
    print(f"  {year}: flooded {n_f:,}, dry {n_d:,} -> flooded share {rec['flooded_share']:.1%}")
    return rec


def main(project: str | None, years, refresh: bool, min_flood_passes: int):
    bad = [y for y in years if y >= C.TEST_YEARS_FROM]
    if bad:
        raise SystemExit(f"{bad} are test years (config.TEST_YEARS_FROM = {C.TEST_YEARS_FROM}); "
                         "they must stay out of the training archive")
    ee = init_earth_engine(project)
    region = ee.Geometry.Rectangle([C.WEST, C.SOUTH, C.EAST, C.NORTH])
    C.FLOOD_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = C.FLOOD_LABELS_DIR / "summary.json"
    old = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    by_year = {r["year"]: r for r in old.get("years", [])}
    for year in years:
        print(f"{year} (August-November, seasonal baseline from {year - 1})")
        by_year[year] = build_year(ee, region, year, refresh, min_flood_passes)
    summary_path.write_text(json.dumps({
        "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": {"season": "August-November", "baseline": "seasonal (same weeks of the year before)",
                   "flooded_pct_at_least": FLOODED_PCT, "dry_pct_below": DRY_PCT,
                   "min_flood_passes": min_flood_passes,
                   "seen_rule": "radar data on >= 50% of the cell and < 50% permanent water"},
        "years": [by_year[y] for y in sorted(by_year)],
    }, indent=2))
    print(f"Wrote {len(years)} years to {C.FLOOD_LABELS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gee-project", help="Google Cloud project id registered for Earth Engine "
                                          f"(default with ${KEY_ENV}: the key's project)")
    ap.add_argument("--years", type=int, nargs="+", default=list(C.ARCHIVE_YEARS))
    ap.add_argument("--min-flood-passes", type=int, default=MIN_FLOOD_PASSES)
    ap.add_argument("--refresh", action="store_true", help="re-download passes already fetched")
    args = ap.parse_args()
    if not (args.gee_project or os.environ.get(KEY_ENV)):
        ap.error(f"--gee-project is required (unless ${KEY_ENV} is set)")
    main(args.gee_project, args.years, args.refresh, args.min_flood_passes)
