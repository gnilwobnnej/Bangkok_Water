"""Buildings and critical facilities for the 3D flood-impact view (scripts/prepare_buildings.py)."""
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import maximum_filter

from src import config as C

MIN_DISPLAY_HEIGHT_M = 3.0  # the height model reads some low buildings as ~0 m


@dataclass
class ImpactMode:
    """One way of judging whether a building is affected, as a value on the analysis grid."""
    key: str
    label: str             # shown in the picker
    grid: np.ndarray       # value per analysis cell
    affected_min: float    # value at or above which a building counts as affected
    severe_min: float      # ... and as severely affected
    value_label: str       # e.g. "Water depth (m)"
    severe_label: str      # e.g. "Deeper than 0.5 m"
    vmax: float            # colour scale maximum
    unit: str = ""


def available_districts() -> list:
    """District ids with prepared building files."""
    if not C.BUILDINGS_DIR.exists():
        return []
    return sorted(int(p.stem) for p in C.BUILDINGS_DIR.glob("*.parquet"))


def load_buildings(district_id: int) -> gpd.GeoDataFrame:
    df = gpd.read_parquet(C.BUILDINGS_DIR / f"{district_id}.parquet")
    df["height_m"] = np.maximum(df["height_m"], MIN_DISPLAY_HEIGHT_M)
    return df


def load_facilities():
    return gpd.read_file(C.FACILITIES_FILE) if C.FACILITIES_FILE.exists() else None


def load_building_grid():
    """(building count, mean height) per analysis cell, or None."""
    if not C.BUILDINGS_GRID_FILE.exists():
        return None
    with rasterio.open(C.BUILDINGS_GRID_FILE) as src:
        return src.read(1), src.read(2)


def impact_modes(result, radar_pass=None, radar_date=None, ml=None) -> dict:
    """Every flood view that has data, keyed for the picker. `result` is a FloodResult."""
    modes = {"sim": ImpactMode(
        "sim", f"Simulated flood at {result.level:.1f} m", result.depth,
        affected_min=0.01, severe_min=0.5, value_label="Water depth", severe_label="Deeper than 0.5 m",
        vmax=3.0, unit=" m")}
    if radar_pass is not None:
        # Radar sees water in open ground, not between buildings, so cells with buildings are almost never
        # flagged themselves. Count a building as affected when floodwater was detected within one cell
        # (~65 m) of it.
        modes["radar"] = ImpactMode(
            "radar", f"Next to radar floodwater on {radar_date}",
            maximum_filter(radar_pass["flood_pct"], size=3),
            affected_min=50, severe_min=90, value_label="Nearby cell flooded",
            severe_label="Next to ≥ 90%-flooded cell", vmax=100, unit="%")
    if ml is not None:
        modes["ml"] = ImpactMode(
            "ml", "ML flood susceptibility", ml.prob,
            affected_min=0.5, severe_min=0.8, value_label="Flood probability", severe_label="Probability ≥ 0.8",
            vmax=1.0)
        modes["2011"] = ImpactMode(
            "2011", "Flooded in 2011", (ml.observed == 1).astype("float32"),
            affected_min=1, severe_min=2, value_label="Flooded in 2011", severe_label="—", vmax=1.0)
    return modes


def values_at(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Grid value at each building / facility (every row carries its analysis-grid row/col)."""
    return grid[df["row"].to_numpy(int), df["col"].to_numpy(int)]


def district_counts(area, grid_counts: np.ndarray, mode: ImpactMode) -> pd.DataFrame:
    """Buildings and affected buildings per district, from the building-count grid (fast, whole city)."""
    ids = area.district_ids.ravel()
    n = int(ids.max()) + 1
    total = np.bincount(ids, weights=grid_counts.ravel(), minlength=n)
    affected = np.bincount(ids, weights=grid_counts.ravel() * (mode.grid.ravel() >= mode.affected_min),
                           minlength=n)
    return pd.DataFrame({"district_id": np.arange(1, n), "buildings": total[1:], "affected": affected[1:]})
