"""The city zone: where the street-flood model beats the river-flood model.

The dense city (cells at least config.DENSE_BUILT_UP_PCT built-up) is where radar can't see floods and the street
model, tested on residents' reports, scores best. Cell by cell it is speckle, so it is smoothed into a few solid
areas for the map's "Best model by area" layer and for scoring each model in its own zone.
"""
from __future__ import annotations

import json

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from src import config as C

# Evaluation scopes: each model scored in the zone where the map shows it
CITY_SCOPE = "City zone"
OUTSIDE_SCOPE = "Outside the city zone (50 districts)"


def city_zone(built_up: np.ndarray, inside: np.ndarray, cell_km2: np.ndarray | float,
              window: int = C.CITY_ZONE_WINDOW, min_km2: float = C.CITY_ZONE_MIN_KM2) -> np.ndarray:
    """Bool grid: mostly dense-city neighbourhoods (over a `window`-cell square), holes filled, pieces smaller
    than `min_km2` dropped, inside `inside` (Bangkok's 50 districts)."""
    dense = (built_up >= C.DENSE_BUILT_UP_PCT).astype("float32")
    zone = ndimage.binary_fill_holes(ndimage.uniform_filter(dense, window) >= 0.5) & inside
    labels, n = ndimage.label(zone)
    if not n:
        return zone
    size = ndimage.sum(np.broadcast_to(cell_km2, zone.shape), labels, np.arange(1, n + 1))
    return np.isin(labels, np.flatnonzero(size >= min_km2) + 1)


def load_city_zone(area) -> np.ndarray | None:
    """The city zone on the study grid, or None if the land-cover fractions haven't been prepared."""
    if not C.LANDCOVER_FILE.exists():
        return None
    with rasterio.open(C.LANDCOVER_FILE) as src:
        built_up = src.read(1 + list(src.descriptions).index("built_up"))
    return city_zone(built_up, area.district_ids > 0, area.cell_area_km2)


def zone_outline(zone: np.ndarray, transform, tolerance: float = 0.0004) -> str:
    """GeoJSON (a FeatureCollection with one feature) of the zone's edge, simplified for the map."""
    polys = [shape(g) for g, v in shapes(zone.astype("uint8"), mask=zone, transform=transform) if v]
    geom = unary_union(polys).simplify(tolerance) if polys else None
    features = [] if geom is None else [{"type": "Feature", "properties": {}, "geometry": mapping(geom.boundary)}]
    return json.dumps({"type": "FeatureCollection", "features": features})
