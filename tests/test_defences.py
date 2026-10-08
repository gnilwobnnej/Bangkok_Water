"""Flood defences: walls block water below their crest, let it over above it, and don't leak diagonally."""
import dataclasses

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString

from src.defences import crest_grid
from src.simulate import flood_mask, level_curve, simulate
from tests.conftest import LOWLAND, N, POCKET, RIVER


def flooded_columns(mask):
    return set(np.flatnonzero(mask.all(axis=0)))


def with_defences(area, crest: np.ndarray):
    return dataclasses.replace(area, defences=crest.astype("float32"))


@pytest.fixture
def walled(area):
    """The fixture area with a 1.0 m wall along column 3, in the middle of the lowland."""
    crest = np.full((N, N), np.nan)
    crest[:, 3] = 1.0
    return with_defences(area, crest)


def test_wall_holds_below_its_crest(walled):
    mask = flood_mask(walled, 0.8, connected=True, defences=True)
    assert flooded_columns(mask) == {0, 1, 2}
    assert not mask[:, 3:].any(), "land behind the wall must stay dry"


def test_water_goes_over_the_wall_above_its_crest(walled):
    mask = flood_mask(walled, 1.2, connected=True, defences=True)
    assert flooded_columns(mask) == set(RIVER) | set(LOWLAND)


def test_defences_off_or_missing_changes_nothing(area, walled):
    for lv in (0.4, 0.8, 1.2, 3.0):
        plain = flood_mask(area, lv, connected=True)
        assert np.array_equal(flood_mask(walled, lv, connected=True, defences=False), plain)
        assert np.array_equal(flood_mask(area, lv, connected=True, defences=True), plain)  # area.defences is None


def test_rain_mode_ignores_walls(walled):
    mask = flood_mask(walled, 0.8, connected=False, defences=True)
    assert flooded_columns(mask) == set(RIVER) | set(LOWLAND) | set(POCKET)


def flat_area(area):
    """Flat 0.5 m land, the river along the west edge."""
    return dataclasses.replace(area, dem=np.full((N, N), 0.5, dtype="float32"))


def test_canal_inside_a_ring_dike_does_not_flood_it(area):
    a = flat_area(area)
    crest = np.full((N, N), np.nan)
    crest[5, 5:15] = crest[14, 5:15] = 1.0
    crest[5:15, 5] = crest[5:15, 14] = 1.0
    a.water[10, 7:13] = True  # a canal inside the ring
    inside = np.zeros((N, N), bool)
    inside[6:14, 6:14] = True
    a = with_defences(a, crest)
    assert flood_mask(a, 0.8, connected=True)[inside].all(), "without defences the canal floods the ring"
    held = flood_mask(a, 0.8, connected=True, defences=True)
    assert not held[inside].any(), "the ring's canal is pumped, so the ring stays dry while the dike holds"
    assert held[~inside & np.isnan(crest)].all(), "everything outside the ring floods"
    assert flood_mask(a, 1.2, connected=True, defences=True)[inside].all(), "over the crest, the ring floods"


def test_diagonal_wall_from_a_line_does_not_leak(area):
    """A diagonal line drawn with crest_grid is a chain of edge-sharing cells, so 8-neighbour flooding can't
    slip between its corners. (A one-cell-per-step diagonal would leak.)"""
    a = flat_area(area)
    transform = from_origin(0, N, 1, 1)  # cell (row r, col c) covers x in [c, c+1], y in [N-r-1, N-r]
    line = gpd.GeoDataFrame({"crest_m": [1.0]}, geometry=[LineString([(1.5, 0), (N, N - 1.5)])])
    crest = crest_grid(line, (N, N), transform)
    walled = with_defences(a, crest)
    mask = flood_mask(walled, 0.8, connected=True, defences=True)
    rows, cols = np.indices((N, N))
    x, y = cols + 0.5, N - rows - 0.5  # cell centres in map coordinates
    behind = (y < x - 1.5 - 1.0) & np.isnan(crest)  # south-east of the line y = x - 1.5, away from the river
    assert behind.sum() > 50 and not mask[behind].any()

    staircase = np.full((N, N), np.nan)
    staircase[rows == cols - 2] = 1.0  # touches only at corners
    leaky = flood_mask(with_defences(a, staircase), 0.8, connected=True, defences=True)
    assert leaky[(cols > rows + 2)].any(), "corner-only walls leak; this is why crest_grid uses all_touched"


def test_crest_grid_keeps_the_higher_crest_where_lines_cross():
    transform = from_origin(0, 10, 1, 1)
    lines = gpd.GeoDataFrame({"crest_m": [3.0, 2.0]},
                             geometry=[LineString([(5.5, 0), (5.5, 10)]), LineString([(0, 4.5), (10, 4.5)])])
    crest = crest_grid(lines, (10, 10), transform)
    assert crest[5, 5] == 3.0 and crest[5, 0] == 2.0 and np.isnan(crest[0, 0])


def test_simulate_and_curve_pass_defences_through(walled):
    res = simulate(walled, 0.8, connected=True, defences=True)
    assert res.area_km2 == pytest.approx(3 * N * 0.01)
    curve = level_curve(walled, [0.8, 1.2], connected=True, defences=True)
    assert curve["area_km2"].tolist() == pytest.approx([3 * N * 0.01, 6 * N * 0.01])
