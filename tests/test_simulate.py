import numpy as np
import pytest

from src.simulate import flood_mask, level_curve, simulate
from tests.conftest import HIGH, LOWLAND, POCKET, RIDGE, RIVER


def flooded_columns(mask):
    """Columns where every cell is flooded (the fixture is the same in every row)."""
    return set(np.flatnonzero(mask.all(axis=0)))


def test_connected_flooding_stops_at_the_ridge(area):
    mask = flood_mask(area, 1.0, connected=True)
    assert flooded_columns(mask) == set(RIVER) | set(LOWLAND)
    assert not mask[:, list(POCKET)].any(), "the sealed pocket must stay dry while the ridge holds"


def test_unconnected_flooding_fills_every_low_cell(area):
    mask = flood_mask(area, 1.0, connected=False)
    assert flooded_columns(mask) == set(RIVER) | set(LOWLAND) | set(POCKET)


def test_water_over_the_ridge_reaches_the_pocket(area):
    mask = flood_mask(area, 3.0, connected=True)
    assert flooded_columns(mask) == set(range(20)), "at the ridge height everything is connected and low enough"


def test_nothing_floods_without_a_water_source(area):
    area.water[:] = False
    assert not flood_mask(area, 2.5, connected=True).any()
    assert flood_mask(area, 2.5, connected=False).any()


def test_depth_is_level_minus_ground_where_flooded(area):
    r = simulate(area, 1.0)
    assert np.allclose(r.depth[r.mask], 1.0 - area.dem[r.mask])
    assert (r.depth[~r.mask] == 0).all()
    assert (r.depth >= 0).all()


def test_totals_match_the_mask(area):
    r = simulate(area, 1.0)
    assert r.area_km2 == pytest.approx(area.cell_area_km2[r.mask].sum())
    assert r.people == pytest.approx(area.population[r.mask].sum())
    inside = r.mask & (area.district_ids > 0)
    assert r.by_district["flooded_km2"].sum() == pytest.approx(area.cell_area_km2[inside].sum())
    assert r.by_district["people_affected"].sum() == pytest.approx(area.population[inside].sum())


def test_district_shares(area):
    by = simulate(area, 1.0).by_district
    assert list(by.index) == [1, 2]
    # District 1 = river + 5 lowland + ridge columns, rows 0-18: 6 of its 7 columns are flooded
    assert by.loc[1, "pct_flooded"] == pytest.approx(100 * 6 / 7)
    assert by.loc[2, "pct_flooded"] == 0
    assert by["pct_flooded"].between(0, 100).all()


@pytest.mark.parametrize("connected", [True, False])
def test_flood_curve_never_goes_down(area, connected):
    curve = level_curve(area, np.round(np.arange(0, 3.01, 0.25), 2), connected)
    assert curve["area_km2"].is_monotonic_increasing
    assert curve["people"].is_monotonic_increasing


def test_flood_curve_matches_simulate(area):
    curve = level_curve(area, [1.0, 2.0]).set_index("level")
    for level in (1.0, 2.0):
        r = simulate(area, level)
        assert curve.loc[level, "area_km2"] == pytest.approx(r.area_km2)
        assert curve.loc[level, "people"] == pytest.approx(r.people)


def test_high_ground_floods_only_when_the_level_reaches_it(area):
    assert not flood_mask(area, 1.9, connected=False)[:, list(HIGH)].any()
    assert flood_mask(area, 2.0, connected=False)[:, list(HIGH)].all()
    assert not flood_mask(area, 2.9, connected=True)[:, list(RIDGE)].any()
