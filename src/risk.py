"""Flood risk index: a weighted blend of terrain, proximity to water, and exposure."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from src import config as C
from src.data_loader import KM_PER_DEG, StudyArea

# Distance (m) at which the proximity factor decays to ~37%.
PROXIMITY_SCALE_M = 1000.0


@dataclass
class RiskFactors:
    low_elevation: np.ndarray  # 1 = lowest ground, 0 = highest
    near_water: np.ndarray     # 1 = on a river/canal, decays with distance
    exposure: np.ndarray       # 1 = densest population


def cell_size_m() -> tuple:
    """(row, column) cell size in metres at the study area's mid-latitude."""
    mid_lat = np.radians((C.NORTH + C.SOUTH) / 2)
    return (C.RES * KM_PER_DEG * 1000, C.RES * KM_PER_DEG * 1000 * np.cos(mid_lat))


def distance_to_water_m(area: StudyArea, mask: np.ndarray = None, return_indices: bool = False):
    """Distance (m) from every cell to the nearest `mask` cell (default: any river/canal).

    With `return_indices`, also returns the (row, col) index arrays of that nearest cell.
    """
    target = area.water if mask is None else mask
    return distance_transform_edt(~target, sampling=cell_size_m(), return_indices=return_indices)


def compute_factors(area: StudyArea) -> RiskFactors:
    lo, hi = np.percentile(area.dem, [2, 98])
    low_elevation = np.clip((hi - area.dem) / (hi - lo), 0, 1)

    near_water = np.exp(-distance_to_water_m(area) / PROXIMITY_SCALE_M)

    density = area.population / area.cell_area_km2
    log_density = np.log1p(density)
    exposure = log_density / log_density.max()

    return RiskFactors(
        low_elevation.astype("float32"), near_water.astype("float32"), exposure.astype("float32")
    )


def risk_index(f: RiskFactors, w_elev: float, w_water: float, w_pop: float) -> np.ndarray:
    total = w_elev + w_water + w_pop
    if total == 0:
        return np.zeros_like(f.low_elevation)
    return (w_elev * f.low_elevation + w_water * f.near_water + w_pop * f.exposure) / total


def district_mean(area: StudyArea, values: np.ndarray, name: str) -> pd.Series:
    """Mean of a per-cell raster within each district, indexed by district_id."""
    ids = area.district_ids.ravel()
    n = int(ids.max()) + 1
    sums = np.bincount(ids, weights=values.ravel(), minlength=n)
    counts = np.bincount(ids, minlength=n)
    mean = np.divide(sums, counts, out=np.zeros(n), where=counts > 0)
    return pd.Series(mean[1:], index=pd.Index(np.arange(1, n), name="district_id"), name=name)


def district_risk(area: StudyArea, risk: np.ndarray) -> pd.Series:
    """Mean risk per district, indexed by district_id."""
    return district_mean(area, risk, "risk")
