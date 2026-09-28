"""Per-cell features for the ML flood-susceptibility model."""
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter

from src import config as C
from src.data_loader import StudyArea
from src.risk import cell_size_m, distance_to_water_m

MAIN_RIVER = "Chao Phraya River"
LOCAL_WINDOW_M = 1000  # neighbourhood for "relative elevation"

# Human-readable names for charts and the app
FEATURE_LABELS = {
    "elevation": "Ground elevation",
    "hand": "Height above nearest waterway",
    "relative_elevation": "Elevation vs. surroundings (1 km)",
    "slope": "Slope",
    "dist_waterway_m": "Distance to any river/canal",
    "dist_main_river_m": "Distance to Chao Phraya",
    "log_pop_density": "Population density (log)",
    "built_up": "Built-up land (%)",
    "cropland": "Cropland (%)",
    "water": "Open water (%)",
    "tree": "Tree cover (%)",
    "grassland": "Grassland (%)",
    "wetland": "Wetland (%)",
}


def _landcover() -> dict:
    with rasterio.open(C.LANDCOVER_FILE) as src:
        return {src.descriptions[i - 1]: src.read(i) for i in range(1, src.count + 1)}


def feature_rasters(area: StudyArea) -> dict:
    """All features as 2-D arrays on the analysis grid."""
    dem = area.dem
    dy, dx = cell_size_m()

    dist_any, (ri, ci) = distance_to_water_m(area, return_indices=True)
    # HAND: how far above the nearest river/canal surface each cell sits.
    hand = dem - dem[ri, ci]

    river = area.waterways[area.waterways["name"] == MAIN_RIVER]
    river_mask = rasterize(
        ((g, 1) for g in river.geometry), out_shape=dem.shape,
        transform=area.transform, fill=0, all_touched=True, dtype="uint8",
    ).astype(bool)
    dist_river = distance_to_water_m(area, mask=river_mask)

    window = max(3, int(round(LOCAL_WINDOW_M / dy)) | 1)
    relative = dem - uniform_filter(dem, size=window, mode="nearest")

    gy, gx = np.gradient(dem, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    rasters = {
        "elevation": dem,
        "hand": hand,
        "relative_elevation": relative,
        "slope": slope,
        "dist_waterway_m": dist_any,
        "dist_main_river_m": dist_river,
        "log_pop_density": np.log1p(area.population / area.cell_area_km2),
    }
    rasters.update(_landcover())
    return {k: np.asarray(v, dtype="float32") for k, v in rasters.items()}


def build_features(area: StudyArea) -> pd.DataFrame:
    """One row per grid cell (row-major), one column per feature."""
    return pd.DataFrame({k: v.ravel() for k, v in feature_rasters(area).items()})
