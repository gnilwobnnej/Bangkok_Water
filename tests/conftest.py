"""Shared fixtures: a tiny synthetic study area, so tests don't need the processed data.

Columns, west to east (every row is the same):

    0       river (water cell, 0.0 m)
    1-5     lowland open to the river (0.5 m)
    6       ridge (3.0 m)
    7-12    pocket sealed off by the ridge (0.5 m)
    13-19   higher ground (2.0 m)

District 1 = columns 0-6, district 2 = columns 7-19; the bottom row is outside Bangkok (district 0).
Population rises from west to east (column index people per cell); every cell is 0.01 km².
"""
import geopandas as gpd
import numpy as np
import pytest

from src.data_loader import StudyArea

N = 20
RIVER, LOWLAND, RIDGE, POCKET, HIGH = [0], range(1, 6), [6], range(7, 13), range(13, 20)


def column_values(spec: dict, dtype="float32") -> np.ndarray:
    """A 20×20 grid where each column takes the value given for it in `spec` ({columns: value})."""
    row = np.zeros(N, dtype=dtype)
    for cols, value in spec.items():
        row[list(cols)] = value
    return np.tile(row, (N, 1))


@pytest.fixture
def area() -> StudyArea:
    dem = column_values({tuple(RIVER): 0.0, tuple(LOWLAND): 0.5, tuple(RIDGE): 3.0,
                         tuple(POCKET): 0.5, tuple(HIGH): 2.0})
    water = np.zeros((N, N), dtype=bool)
    water[:, RIVER] = True
    district_ids = column_values({range(0, 7): 1, range(7, N): 2}, dtype="int32")
    district_ids[-1, :] = 0
    return StudyArea(
        dem=dem,
        population=np.tile(np.arange(N, dtype="float32"), (N, 1)),
        water=water,
        cell_area_km2=np.full((N, N), 0.01, dtype="float32"),
        district_ids=district_ids,
        districts=gpd.GeoDataFrame({"district_id": [1, 2], "district": ["West", "East"]}, geometry=[None, None]),
        waterways=gpd.GeoDataFrame(geometry=[]),
        transform=None,
    )
