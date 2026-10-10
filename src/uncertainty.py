"""Elevation uncertainty: how much the flood results move when the elevation data is a little wrong.

scripts/dem_uncertainty.py reruns the bathtub simulation on many copies of the elevation map, each with
random, spatially correlated error added, and stores the spread of the results. This module holds the noise
model (shared with the tests) and loads the results for the app.

Default error: standard deviation 0.7 m, correlated over about 300 m. Both are assumptions, settable in the
script. Reasons:
- Copernicus GLO-30 errors are spatially correlated rather than independent per cell, and simulating DEM error
  as a correlated random field is the approach of Hawker et al. (2018). Their Mekong Delta case is similar
  terrain to Bangkok.
- In built-up areas, Copernicus GLO-30 has a mean absolute error of 1.61 m, much of it from buildings
  (Hawker et al. 2022, FABDEM). This project takes the lowest value in each ~65 m cell, which removes most of
  that building bias. 0.7 m is therefore an estimate of the error that remains, not a measured value. Run the
  script with --sigma 1.4 for a pessimistic case.

References:
  Hawker L, Bates P, Neal J, Rougier J (2018). Perspectives on digital elevation model (DEM) simulation for
    flood modeling in the absence of a high-accuracy open access global DEM. Frontiers in Earth Science 6:233.
    doi:10.3389/feart.2018.00233
  Hawker L, Uhe P, Paulo L, Sosa J, Savage J, Sampson C, Neal J (2022). A 30 m global map of elevation with
    forests and buildings removed. Environmental Research Letters 17:024016. doi:10.1088/1748-9326/ac4d4f
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter

from src import config as C

SIGMA_M = 0.7     # standard deviation of the elevation error, m
CORR_M = 300.0    # distance at which the error's correlation falls to 1/e, m
N_RUNS = 50
PROB_LEVELS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)  # water levels with a "chance flooded" map
# Simulation modes, as (connected, defences); the keys name the files
MODES = {"river": (True, False), "river_defences": (True, True), "anywhere": (False, False)}


def mode_key(connected: bool, defences: bool) -> str:
    return "anywhere" if not connected else ("river_defences" if defences else "river")


def correlated_noise(shape: tuple, cell_m: tuple, sigma_m: float = SIGMA_M, corr_m: float = CORR_M,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Zero-mean random field with standard deviation `sigma_m` and correlation exp(-(r / corr_m)^2).

    White noise smoothed with a Gaussian kernel of standard deviation s has correlation exp(-r^2 / (4 s^2)),
    so s = corr_m / 2. cell_m is the (row, column) cell size in metres. The field is rescaled to exactly
    `sigma_m`, which removes the variance lost to smoothing.
    """
    rng = rng or np.random.default_rng()
    s = (corr_m / 2 / cell_m[0], corr_m / 2 / cell_m[1])
    field = gaussian_filter(rng.standard_normal(shape), s, mode="wrap")
    return (field / field.std() * sigma_m).astype("float32")


@dataclass
class DemUncertainty:
    meta: dict
    curves: dict  # mode -> DataFrame: level, area_km2_p5/p50/p95, people_p5/..., districts_25_p5/...

    def curve(self, connected: bool, defences: bool) -> pd.DataFrame | None:
        return self.curves.get(mode_key(connected, defences))

    def at(self, connected: bool, defences: bool, level: float) -> pd.Series | None:
        """The percentiles at the level closest to `level`."""
        c = self.curve(connected, defences)
        if c is None:
            return None
        return c.iloc[(c["level"] - level).abs().argmin()]


def load_dem_uncertainty(path=None) -> DemUncertainty | None:
    path = path or C.DEM_UNCERTAINTY_FILE
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return DemUncertainty(meta={k: v for k, v in d.items() if k != "curves"},
                          curves={m: pd.DataFrame(rows) for m, rows in d["curves"].items()})


def nearest_prob_level(level: float) -> float:
    return min(PROB_LEVELS, key=lambda lv: abs(lv - level))


def load_flood_chance(connected: bool, defences: bool, level: float) -> tuple[np.ndarray, float] | None:
    """Share of runs (0-1) in which each cell floods, at the precomputed level nearest `level`."""
    path = C.DEM_UNCERTAINTY_DIR / f"chance_{mode_key(connected, defences)}.tif"
    if not path.exists():
        return None
    lv = nearest_prob_level(level)
    with rasterio.open(path) as src:
        return src.read(PROB_LEVELS.index(lv) + 1).astype("float32") / 100, lv
