"""Can Sentinel-1 radar see street flooding? A test of the "double bounce" against residents' flood reports.

    python scripts/double_bounce.py --gee-project ID

Open water looks dark to radar, which is how fetch_current_flood.py maps floods. Between buildings it can do the
opposite: the signal bounces off the water, then off a wall, and straight back to the satellite, so a flooded
street looks *brighter* than when it was dry. This script flags built-up ground (ESA WorldCover) whose VV
backscatter rose by at least BRIGHT_DB over a baseline (the median of the same orbit's scenes over the previous
BASELINE_DAYS), and tests the flag against Traffy Fondue flood reports before anything relies on it.

Passes tested (Bangkok has two Sentinel-1 tracks, ~06:00 and ~18:00 local):
    wet      on a storm day (or the day after), with >= MIN_WET_REPORTS flood reports within WINDOW_H hours
    control  January-March, with at most 2 flood reports within WINDOW_H hours: how much brightens anyway
Each wet pass is scored like the street-flood model: flooded = a cell with a flood report within WINDOW_H hours of
the pass, dry = that year's Traffy "dry" label (no flood report all year, but people report other problems).

Outputs:
    data/raw/double_bounce/<pass>.tif                     the 20 m classes (0 not brighter, 1 brighter, 255 n/a)
    data/processed/double_bounce/evaluation.json          per-pass scores, and wet vs control brightening
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_current import score  # noqa: E402
from fetch_current_flood import (DOWNLOAD_SCALE_M, SPECKLE_RADIUS_M, download, init_earth_engine,  # noqa: E402
                                 s1_collection)
from prepare_data import TRANSFORM  # noqa: E402
from src import config as C  # noqa: E402
from src.data_loader import load_study_area  # noqa: E402
from src.traffy import count_grid, load_reports, load_summary  # noqa: E402
from src.validation import block_bootstrap  # noqa: E402
from train_model import spatial_blocks  # noqa: E402

WORLDCOVER = "ESA/WorldCover/v200"
BUILT_UP_CLASS = 50
BRIGHT_DB = 3.0         # a built-up pixel this much brighter than its baseline is flagged
BASELINE_DAYS = 60      # baseline: median of the same orbit's scenes over this many days before the pass
WINDOW_H = 6            # flood reports within this many hours of the pass
MIN_WET_REPORTS = 20
N_CONTROLS = 12
MIN_BUILT_PCT = 25      # cells scored need at least this % of built-up ground the radar saw
MIN_CELLS = 20
RAW_DIR = C.RAW_DIR / "double_bounce"
OUT_DIR = C.PROCESSED_DIR / "double_bounce"
OUT_FILE = OUT_DIR / "evaluation.json"
BRIGHT, NOT_BRIGHT, NA = 1, 0, 255


def list_passes(ee, region, since: str, until: str) -> pd.DataFrame:
    """One row per pass: UTC start time, Bangkok local time, direction and orbit."""
    info = (s1_collection(ee, region).filterDate(since, until)
            .reduceColumns(ee.Reducer.toList(3),
                           ["system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start"])
            .get("list").getInfo())
    df = pd.DataFrame(info, columns=["ms", "direction", "orbit"])
    df["utc"] = pd.to_datetime(df["ms"], unit="ms")
    df = df.sort_values("utc").groupby([df["utc"].dt.floor("h"), "direction", "orbit"], as_index=False).first()
    df["local"] = df["utc"] + pd.Timedelta(hours=7)
    df["id"] = df["local"].dt.strftime("%Y-%m-%dT%H")
    return df[["id", "utc", "local", "direction", "orbit"]].reset_index(drop=True)


def reports_near(times: pd.Series, when: pd.Timestamp, hours: float = WINDOW_H) -> np.ndarray:
    """Which reports (Bangkok local times) fall within `hours` of `when`."""
    return ((times >= when - pd.Timedelta(hours=hours)) & (times <= when + pd.Timedelta(hours=hours))).to_numpy()


def choose_passes(passes: pd.DataFrame, report_times: pd.Series, storms: set) -> pd.DataFrame:
    """The wet passes and an even spread of dry-season control passes."""
    n = np.array([reports_near(report_times, t).sum() for t in passes["local"]])
    day = passes["local"].dt.normalize()
    on_storm = day.isin(storms) | (day - pd.Timedelta(days=1)).isin(storms)
    passes = passes.assign(reports=n)
    wet = passes[on_storm.to_numpy() & (n >= MIN_WET_REPORTS)].assign(kind="wet")
    dry = passes[passes["local"].dt.month.isin([1, 2, 3]) & (n <= 2)]
    pick = np.linspace(0, len(dry) - 1, min(N_CONTROLS, len(dry))).round().astype(int)
    return pd.concat([wet, dry.iloc[pick].assign(kind="control")]).sort_values("utc").reset_index(drop=True)


def brightening(ee, region, utc: pd.Timestamp, direction: str, orbit: int):
    """Server-side classes for one pass: built-up pixels BRIGHT_DB brighter than the baseline, or not."""
    same_orbit = (s1_collection(ee, region)
                  .filter(ee.Filter.eq("orbitProperties_pass", direction))
                  .filter(ee.Filter.eq("relativeOrbitNumber_start", int(orbit))))
    t0 = utc.floor("h")
    iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731
    scenes = same_orbit.filterDate(iso(t0), iso(t0 + pd.Timedelta(hours=1)))
    proj = scenes.first().projection()
    now = scenes.mosaic().setDefaultProjection(proj).focalMedian(SPECKLE_RADIUS_M, "circle", "meters")
    base_col = same_orbit.filterDate(iso(t0 - pd.Timedelta(days=BASELINE_DAYS)), iso(t0 - pd.Timedelta(hours=12)))
    base = base_col.median().setDefaultProjection(proj).focalMedian(SPECKLE_RADIUS_M, "circle", "meters")
    built = ee.ImageCollection(WORLDCOVER).first().eq(BUILT_UP_CLASS)
    seen = now.mask().And(base.mask()).And(built)
    code = (ee.Image(NA).where(seen, NOT_BRIGHT).where(seen.And(now.subtract(base).gte(BRIGHT_DB)), BRIGHT)
            .unmask(NA).toUint8().clip(region))
    return code, base_col.size().getInfo()


def to_grid(raw_path: Path) -> tuple:
    """(% of the seen built-up ground that brightened, % of the cell that is seen built-up ground) per cell."""
    with rasterio.open(raw_path) as src:
        codes, src_transform, src_crs = src.read(1), src.transform, src.crs
    out = []
    for mask in (codes == BRIGHT, codes != NA):
        grid = np.zeros((C.HEIGHT, C.WIDTH), dtype="float32")
        reproject(source=(mask * 100).astype("uint8"), destination=grid, src_transform=src_transform,
                  src_crs=src_crs, dst_transform=TRANSFORM, dst_crs=C.CRS, resampling=Resampling.average)
        out.append(grid)
    bright, seen = out
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen > 0, bright / seen * 100, np.nan), seen


def year_dry_labels(year: int) -> np.ndarray | None:
    path = C.FLOOD_LABELS_DIR / f"traffy_{year}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1) == 0


def main(project: str | None, refresh: bool):
    ee = init_earth_engine(project)
    region = ee.Geometry.Rectangle([C.WEST, C.SOUTH, C.EAST, C.NORTH])
    reports = load_reports()
    storms = set(pd.to_datetime([s["date"] for s in load_summary()["storms"]]))
    passes = list_passes(ee, region, f"{reports['time'].min():%Y-%m-%d}", f"{dt.date.today()}")
    chosen = choose_passes(passes, reports["time"], storms)
    print(f"{len(passes)} passes; testing {sum(chosen['kind'] == 'wet')} wet and "
          f"{sum(chosen['kind'] == 'control')} control passes")

    area = load_study_area()
    bkk = area.district_ids > 0
    blocks = spatial_blocks().reshape(bkk.shape)
    street = None
    if C.ML_STREET_PROB_FILE.exists():
        with rasterio.open(C.ML_STREET_PROB_FILE) as src:
            street = src.read(1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for p in chosen.itertuples():
        raw = RAW_DIR / f"{p.id}_{p.direction[:3].lower()}{p.orbit}.tif"
        code, n_base = brightening(ee, region, p.utc, p.direction, p.orbit)
        if n_base == 0:
            print(f"  {p.id}: no baseline scenes, skipped")
            continue
        if refresh or not raw.exists():
            url = code.getDownloadURL({"region": region, "scale": DOWNLOAD_SCALE_M, "crs": "EPSG:4326",
                                       "format": "GEO_TIFF"})
            download(url, raw)
        bright, seen = to_grid(raw)
        usable = bkk & (seen >= MIN_BUILT_PCT) & np.isfinite(bright)
        rec = {"pass": p.id, "kind": p.kind, "direction": p.direction, "orbit": int(p.orbit),
               "baseline_scenes": n_base, "reports_in_window": int(p.reports),
               "bright_pct_bangkok": float(np.nanmean(bright[usable]))}
        near = reports[reports_near(reports["time"], p.local)]
        dry = year_dry_labels(p.local.year)
        if p.kind == "wet" and dry is not None:
            flooded = count_grid(near["lon"], near["lat"]) > 0
            known = usable & (flooded | dry)
            if street is not None:
                known &= np.isfinite(street)
            y = flooded[known].astype(int)
            rec.update(n_flooded=int(y.sum()), n_dry=int((y == 0).sum()))
            if y.sum() >= MIN_CELLS and (y == 0).sum() >= MIN_CELLS:
                methods = {"Double bounce": bright[known]}
                if street is not None:
                    methods["ML street"] = street[known]
                ci = block_bootstrap(y, methods, blocks[known], reference="ML street" if street is not None
                                     else "Double bounce")
                rec["methods"] = {m: {**score(y, s), **ci[m]} for m, s in methods.items()}
                rec["bright_pct_flooded"] = float(bright[known][y == 1].mean())
                rec["bright_pct_dry"] = float(bright[known][y == 0].mean())
        results.append(rec)
        extra = ""
        if "methods" in rec:
            m = rec["methods"]
            extra = (f"  flooded {rec['bright_pct_flooded']:.1f}% vs dry {rec['bright_pct_dry']:.1f}% brighter; "
                     f"ROC-AUC double bounce {m['Double bounce']['roc_auc']:.3f}"
                     + (f", ML street {m['ML street']['roc_auc']:.3f}" if "ML street" in m else ""))
        elif "n_flooded" in rec:
            extra = f"  too few cells ({rec['n_flooded']} flooded, {rec['n_dry']} dry)"
        print(f"  {p.id} {p.kind:7s} {p.reports:5d} reports, {rec['bright_pct_bangkok']:.1f}% of built-up "
              f"brighter{extra}", flush=True)

    wet = [r["bright_pct_bangkok"] for r in results if r["kind"] == "wet"]
    ctl = [r["bright_pct_bangkok"] for r in results if r["kind"] == "control"]
    scored = [r for r in results if "methods" in r]
    OUT_FILE.write_text(json.dumps({
        "method": {"bright_db": BRIGHT_DB, "baseline_days": BASELINE_DAYS, "window_hours": WINDOW_H,
                   "min_wet_reports": MIN_WET_REPORTS, "built_up": f"{WORLDCOVER} class {BUILT_UP_CLASS}",
                   "min_built_pct": MIN_BUILT_PCT, "speckle_radius_m": SPECKLE_RADIUS_M},
        "summary": {"wet_passes": len(wet), "control_passes": len(ctl),
                    "median_bright_pct_wet": float(np.median(wet)) if wet else None,
                    "median_bright_pct_control": float(np.median(ctl)) if ctl else None,
                    "scored_passes": len(scored)},
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nBuilt-up Bangkok brighter by >= {BRIGHT_DB} dB: median {np.median(wet):.1f}% on wet passes, "
          f"{np.median(ctl):.1f}% on dry-season controls")
    if scored:
        aucs = [r["methods"]["Double bounce"]["roc_auc"] for r in scored]
        print(f"Double bounce ROC-AUC on {len(scored)} scored wet passes: median {np.median(aucs):.3f} "
              f"(range {min(aucs):.3f}-{max(aucs):.3f})")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gee-project", help="Google Cloud project for Earth Engine")
    ap.add_argument("--refresh", action="store_true", help="download passes again")
    args = ap.parse_args()
    main(args.gee_project, args.refresh)
