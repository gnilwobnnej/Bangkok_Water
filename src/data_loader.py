"""Load the preprocessed rasters and vectors produced by scripts/prepare_data.py."""
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.features import rasterize

from src import config as C

KM_PER_DEG = 111.32


@dataclass
class StudyArea:
    dem: np.ndarray            # elevation, m
    population: np.ndarray     # people per cell
    water: np.ndarray          # bool, river/canal cells (flood sources)
    cell_area_km2: np.ndarray  # area of each cell, km² (varies with latitude)
    district_ids: np.ndarray   # 0 = outside Bangkok, 1..N = district_id
    districts: gpd.GeoDataFrame
    waterways: gpd.GeoDataFrame
    transform: object


def missing_files():
    files = [C.DEM_FILE, C.POP_FILE, C.WATER_FILE, C.DISTRICTS_FILE, C.WATERWAY_LINES_FILE]
    return [f for f in files if not f.exists()]


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform


def load_study_area() -> StudyArea:
    dem, transform = _read(C.DEM_FILE)
    population, _ = _read(C.POP_FILE)
    water, _ = _read(C.WATER_FILE)

    # Cell area shrinks with cos(latitude); compute once per row.
    lats = C.NORTH - (np.arange(C.HEIGHT) + 0.5) * C.RES
    row_area = (C.RES * KM_PER_DEG) ** 2 * np.cos(np.radians(lats))
    cell_area = np.repeat(row_area[:, None], C.WIDTH, axis=1).astype("float32")

    districts = gpd.read_file(C.DISTRICTS_FILE)
    district_ids = rasterize(
        zip(districts.geometry, districts["district_id"]),
        out_shape=dem.shape, transform=transform, fill=0, dtype="int32",
    )
    waterways = gpd.read_file(C.WATERWAY_LINES_FILE)
    # Simplify and round coordinates (~10 m) for a lighter map payload.
    waterways["geometry"] = shapely.set_precision(waterways.geometry.simplify(0.0002).values, 1e-4)

    return StudyArea(
        dem=dem.astype("float32"), population=population.astype("float32"),
        water=water.astype(bool), cell_area_km2=cell_area, district_ids=district_ids,
        districts=districts, waterways=waterways, transform=transform,
    )
