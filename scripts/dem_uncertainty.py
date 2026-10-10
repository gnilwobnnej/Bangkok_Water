"""Elevation uncertainty for the bathtub simulation: rerun it on many plausible versions of the elevation map.

    python scripts/dem_uncertainty.py                         # 50 runs, sigma 0.7 m, correlation 300 m
    python scripts/dem_uncertainty.py --sigma 1.4 --out-dir data/processed/dem_uncertainty_pessimistic

Each run adds random, spatially correlated error to the elevation (src/uncertainty.py explains the error model
and its sources) and floods the result at every level from 0 to 3 m. This is done in each simulation mode:
water from rivers and canals, the same with flood defences (if prepared), and every low cell. Defence crest
heights are kept as they are; only the ground moves. Takes about 10-15 minutes.

Outputs (in data/processed/dem_uncertainty/):
    summary.json          per mode and level: P5, P50 and P95 of flooded area, people affected and districts
                          more than 25% flooded, over the runs
    chance_<mode>.tif     % of runs in which each cell floods, one uint8 band per level in PROB_LEVELS
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.data_loader import load_study_area  # noqa: E402
from src.risk import cell_size_m  # noqa: E402
from src.simulate import flood_mask  # noqa: E402
from src.uncertainty import CORR_M, MODES, N_RUNS, PROB_LEVELS, SIGMA_M, correlated_noise  # noqa: E402

LEVELS = np.round(np.arange(0, 3.001, 0.1), 1)  # the app's flood curve levels
SEED = 42
PCTS = (5, 50, 95)


def onset_levels(area, connected: bool, defences: bool) -> np.ndarray:
    """Index into LEVELS of the lowest level at which each cell floods (len(LEVELS) = never)."""
    onset = np.full(area.dem.shape, len(LEVELS), dtype="int8")
    for i in range(len(LEVELS) - 1, -1, -1):
        onset[flood_mask(area, float(LEVELS[i]), connected, defences)] = i
    return onset


def run_stats(area, onset: np.ndarray, district_area: np.ndarray) -> np.ndarray:
    """(levels, 3) flooded km², people and districts > 25% flooded at each level, from one run's onsets."""
    ids = area.district_ids.ravel()
    n = len(district_area)
    flat = onset.ravel()
    cell_area = area.cell_area_km2.ravel()
    pop = area.population.ravel()
    # Per cell, add its area and people at its onset level, then accumulate up the levels
    by_level = np.zeros((len(LEVELS) + 1, 2))
    np.add.at(by_level[:, 0], flat, cell_area)
    np.add.at(by_level[:, 1], flat, pop)
    totals = np.cumsum(by_level[:-1], axis=0)
    # Districts: flooded area per district per level
    per_district = np.zeros((len(LEVELS) + 1, n))
    np.add.at(per_district, (flat, ids), cell_area)
    pct = np.cumsum(per_district[:-1], axis=0) / np.maximum(district_area, 1e-9)
    over_25 = (pct[:, 1:] > 0.25).sum(axis=1)
    return np.column_stack([totals, over_25])


def main(sigma: float, corr: float, runs: int, out_dir: Path):
    area = load_study_area()
    dem = area.dem.copy()
    cell_m = cell_size_m()
    ids = area.district_ids.ravel()
    district_area = np.bincount(ids, weights=area.cell_area_km2.ravel())
    modes = {k: v for k, v in MODES.items() if not v[1] or area.defences is not None}
    stats = {m: np.zeros((runs, len(LEVELS), 3)) for m in modes}
    chance = {m: np.zeros((len(PROB_LEVELS),) + dem.shape, dtype="uint16") for m in modes}
    prob_idx = [int(np.flatnonzero(np.isclose(LEVELS, lv))[0]) for lv in PROB_LEVELS]
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for r in range(runs):
        area.dem = dem + correlated_noise(dem.shape, cell_m, sigma, corr, rng)
        for m, (connected, defences) in modes.items():
            onset = onset_levels(area, connected, defences)
            stats[m][r] = run_stats(area, onset, district_area)
            for j, i in enumerate(prob_idx):
                chance[m][j] += onset <= i
        print(f"  run {r + 1}/{runs} done ({time.time() - t0:.0f} s)")
    area.dem = dem

    curves = {}
    for m in modes:
        p = np.percentile(stats[m], PCTS, axis=0)  # (3 pcts, levels, 3 quantities)
        curves[m] = [
            {"level": float(lv), **{f"{q}_p{pc}": float(p[k, i, j])
                                    for j, q in enumerate(("area_km2", "people", "districts_25"))
                                    for k, pc in enumerate(PCTS)}}
            for i, lv in enumerate(LEVELS)
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=C.HEIGHT, width=C.WIDTH, count=len(PROB_LEVELS), crs=C.CRS,
                   transform=area.transform, compress="deflate", dtype="uint8")
    for m in modes:
        with rasterio.open(out_dir / f"chance_{m}.tif", "w", **profile) as dst:
            dst.write(np.round(100 * chance[m] / runs).astype("uint8"))
            for b, lv in enumerate(PROB_LEVELS, start=1):
                dst.set_band_description(b, f"% of runs flooded at {lv:.1f} m")
    (out_dir / C.DEM_UNCERTAINTY_FILE.name).write_text(json.dumps({
        "sigma_m": sigma, "corr_m": corr, "runs": runs, "seed": SEED, "percentiles": list(PCTS),
        "prob_levels": list(PROB_LEVELS),
        "error_model": "zero-mean Gaussian random field, correlation exp(-(r/corr_m)^2), added to the DEM; "
                       "defence crests unchanged (src/uncertainty.py)",
        "curves": curves,
    }, indent=2))

    print(f"\nFlooded area and people, P5-P95 over {runs} runs (sigma {sigma} m, correlation {corr:.0f} m):")
    for m in modes:
        for row in curves[m]:
            if row["level"] in (1.0, 2.0):
                print(f"  {m:15s} {row['level']:.1f} m: {row['area_km2_p5']:5.0f}-{row['area_km2_p95']:5.0f} km², "
                      f"{row['people_p5'] / 1e6:.2f}-{row['people_p95'] / 1e6:.2f} M people")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=SIGMA_M, help="elevation error standard deviation, m")
    ap.add_argument("--corr", type=float, default=CORR_M, help="error correlation distance, m")
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--out-dir", type=Path, default=C.DEM_UNCERTAINTY_DIR)
    args = ap.parse_args()
    main(args.sigma, args.corr, args.runs, args.out_dir)
