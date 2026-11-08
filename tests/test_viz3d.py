"""3D buildings tab helpers (src/viz3d.py): colours, tooltips, camera, decks and footprints."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from src import config as C
from src import viz3d as V
from src.buildings import ImpactMode
from src.viz import OBSERVED_COLOR, RADAR_COLOR

N = 20


def make_mode(key: str, grid: np.ndarray | None = None) -> ImpactMode:
    """Modes with the thresholds src/buildings.impact_modes uses."""
    grid = np.zeros((N, N), dtype="float32") if grid is None else grid
    params = {"sim": (0.01, 3.0), "radar": (50, 100), "ml": (0.5, 1.0), "2011": (1, 1.0)}
    affected_min, vmax = params[key]
    return ImpactMode(key, key, grid, affected_min=affected_min, severe_min=affected_min * 2,
                      value_label="Value", severe_label="-", vmax=vmax)


def buildings_at(cells, heights=None) -> pd.DataFrame:
    rows, cols = zip(*cells)
    return pd.DataFrame({
        "row": rows, "col": cols,
        "lon": [100.5 + 0.001 * c for c in cols], "lat": [13.7 + 0.001 * r for r in rows],
        "height_m": heights if heights is not None else [10.0] * len(cells),
    })


# ---- mode_colors -------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["sim", "radar", "ml", "2011"])
def test_unaffected_items_are_grey_in_every_mode(key):
    mode = make_mode(key)
    below = np.array([0.0, mode.affected_min * 0.99])
    assert (V.mode_colors(below, mode) == V.DRY_COLOR).all()


def test_affected_threshold_is_inclusive():
    mode = make_mode("radar")
    out = V.mode_colors(np.array([49.9, 50.0]), mode)
    assert list(out[0]) == V.DRY_COLOR
    assert list(out[1]) == list(RADAR_COLOR)


def test_observed_2011_uses_the_amber_colour():
    out = V.mode_colors(np.array([0.0, 1.0]), make_mode("2011"))
    assert list(out[1]) == list(OBSERVED_COLOR)


def test_sim_colours_darken_with_depth_and_saturate_at_vmax():
    out = V.mode_colors(np.array([0.1, 1.5, 3.0, 10.0]), make_mode("sim"))
    assert out[0].sum() > out[1].sum() > out[2].sum()  # deeper water = darker blue
    assert (out[2] == out[3]).all(), "values above vmax are clipped to the end of the scale"
    assert out.dtype.kind == "i" and out.shape == (4, 3)


def test_mode_colors_handles_empty_input():
    assert V.mode_colors(np.array([]), make_mode("sim")).shape == (0, 3)


# ---- _fmt (tooltip text) -----------------------------------------------------------------------

def test_tooltip_text_per_mode():
    v = np.array([0.0, 0.456, 1.0])
    assert V._fmt(v, make_mode("sim")) == ["dry", "0.46 m", "1.00 m"]
    assert V._fmt(np.array([0.0, 49.6]), make_mode("radar")) == ["0%", "50%"]
    assert V._fmt(v, make_mode("ml")) == ["0.00", "0.46", "1.00"]
    assert V._fmt(v, make_mode("2011")) == ["no", "no", "yes"]


# ---- facilities_layer --------------------------------------------------------------------------

def test_facilities_flag_affected_and_keep_type_colours():
    grid = np.zeros((N, N), dtype="float32")
    grid[2, 3] = 0.8
    fac = pd.DataFrame({"row": [9, 2, 5], "col": [9, 3, 5], "lon": [1.0, 2.0, 3.0], "lat": [4.0, 5.0, 6.0],
                        "name": ["Drop", "Wat School", "Station"], "type": ["Hospital", "School", "Police"]})
    # A filtered subset with a non-default index, the way the app passes it
    layer = V.facilities_layer(fac.iloc[1:], make_mode("sim", grid), height_scale=2)
    rows = layer.data
    assert [r["n"] for r in rows] == ["Wat School (School)", "Station (Police)"]
    assert rows[0]["v"] == "AFFECTED · 0.80 m" and rows[0]["c"] == [255, 40, 120]
    assert rows[1]["v"] == "dry" and rows[1]["c"] == V.FACILITY_COLORS["Police"]
    assert (rows[0]["lon"], rows[0]["lat"]) == (2.0, 5.0)


def test_unknown_facility_type_falls_back_to_white():
    fac = pd.DataFrame({"row": [0], "col": [0], "lon": [1.0], "lat": [1.0], "name": ["X"], "type": ["Temple"]})
    assert V.facilities_layer(fac, make_mode("sim"), 1).data[0]["c"] == [255, 255, 255]


# ---- district_deck -----------------------------------------------------------------------------

def test_light_mode_deck_uses_columns_and_compact_json():
    grid = np.zeros((N, N), dtype="float32")
    grid[1, 1] = 0.5
    b = buildings_at([(1, 1), (4, 4)], heights=[12.34, 3.0])
    view = V.district_view(b)
    deck = V.district_deck(b, None, make_mode("sim", grid), 1.5, None, view)
    assert len(deck.layers) == 1
    layer = deck.layers[0]
    assert layer.type == "ColumnLayer" and layer.radius == V.LIGHT_RADIUS_M
    assert [r["h"] for r in layer.data] == [12.3, 3.0]
    assert [r["v"] for r in layer.data] == ["0.50 m", "dry"]
    assert layer.data[1]["c"] == V.DRY_COLOR
    text = deck.to_json()
    assert "\n" not in text and json.loads(text)["layers"][0]["@@type"] == "ColumnLayer"


def test_footprint_deck_uses_polygons_and_adds_facilities():
    b = buildings_at([(0, 0)])
    coords = [[[100.5, 13.7], [100.501, 13.7], [100.501, 13.701], [100.5, 13.7]]]
    fac = pd.DataFrame({"row": [0], "col": [0], "lon": [1.0], "lat": [1.0], "name": ["H"], "type": ["Hospital"]})
    deck = V.district_deck(b, coords, make_mode("sim"), 1, fac, V.district_view(b))
    assert [l.type for l in deck.layers] == ["PolygonLayer", "ColumnLayer"]
    assert deck.layers[0].data[0]["p"] == coords[0]


def test_empty_facilities_add_no_layer():
    b = buildings_at([(0, 0)])
    fac = pd.DataFrame(columns=["row", "col", "lon", "lat", "name", "type"])
    deck = V.district_deck(b, None, make_mode("sim"), 1, fac, V.district_view(b))
    assert len(deck.layers) == 1


# ---- district_view -----------------------------------------------------------------------------

def test_view_fits_the_district_when_few_are_affected():
    b = pd.DataFrame({"lon": [100.50, 100.60], "lat": [13.70, 13.76]})
    view = V.district_view(b, np.array([True, False]))
    assert view["longitude"] == pytest.approx(100.55) and view["latitude"] == pytest.approx(13.73)
    # span = max(0.10, 0.06 * 1.3) = 0.10 -> log2(3600) + 0.2
    assert view["zoom"] == pytest.approx(np.log2(3600) + 0.2)
    assert view["pitch"] == 50


def test_view_of_a_single_building_is_clamped_not_infinite():
    view = V.district_view(pd.DataFrame({"lon": [100.5], "lat": [13.7]}))
    assert np.isfinite(view["zoom"]) and 11 <= view["zoom"] <= 16


def test_view_focuses_on_the_affected_buildings_when_there_are_enough():
    n = V.FOCUS_MIN_AFFECTED
    lon = np.r_[np.full(n, 100.40), np.full(n, 100.90)]
    b = pd.DataFrame({"lon": lon, "lat": np.full(2 * n, 13.7)})
    affected = np.r_[np.zeros(n, bool), np.ones(n, bool)]
    view = V.district_view(b, affected)
    assert view["longitude"] == pytest.approx(100.90) and view["zoom"] == V.FOCUS_ZOOM


def test_one_short_of_the_focus_threshold_fits_the_district():
    n = V.FOCUS_MIN_AFFECTED
    b = pd.DataFrame({"lon": np.linspace(100.4, 100.9, n), "lat": np.full(n, 13.7)})
    affected = np.ones(n, bool)
    affected[0] = False
    assert V.district_view(b, affected)["pitch"] == 50


# ---- city_deck ---------------------------------------------------------------------------------

def test_city_blocks_count_only_buildings_inside_bangkok(area, monkeypatch):
    # Shrink the city grid to the 20x20 fixture: whole 8x8 blocks cover rows/cols 0-15 (2x2 blocks).
    monkeypatch.setattr(C, "HEIGHT", N)
    monkeypatch.setattr(C, "WIDTH", N)
    counts = np.ones((N, N))
    area.district_ids[0, 0] = 0  # one cell outside Bangkok in the top-left block
    grid = np.zeros((N, N), dtype="float32")
    grid[:8, :4] = 1.0  # half of the top-left block flooded
    deck = V.city_deck(area, counts, make_mode("sim", grid), 1)
    rows = deck.layers[0].data
    assert len(rows) == 4
    top_left = rows[0]
    assert (top_left["t"], top_left["a"], top_left["s"]) == (63, 31, 49)  # 64 - 1 outside; 32 - 1 outside
    assert top_left["lon"] == pytest.approx(C.WEST + 4 * C.RES)
    assert top_left["lat"] == pytest.approx(C.NORTH - 4 * C.RES)
    assert all(r["c"] == V.CITY_EMPTY_COLOR for r in rows[1:]), "blocks with nothing affected stay dark grey"
    assert top_left["c"] != V.CITY_EMPTY_COLOR


def test_city_blocks_without_buildings_are_dropped(area, monkeypatch):
    monkeypatch.setattr(C, "HEIGHT", N)
    monkeypatch.setattr(C, "WIDTH", N)
    counts = np.zeros((N, N))
    counts[10, 10] = 3
    deck = V.city_deck(area, counts, make_mode("sim"), 1)
    rows = deck.layers[0].data
    assert len(rows) == 1 and rows[0]["t"] == 3 and rows[0]["s"] == 0


# ---- footprint_coords --------------------------------------------------------------------------

def test_footprints_round_and_use_the_largest_part_of_multipolygons():
    small, big = box(0, 0, 1e-4, 1e-4), box(1, 1, 1.002, 1.002)
    poly = Polygon([(100.1234567, 13.1234567), (100.2, 13.1), (100.2, 13.2)])
    gdf = gpd.GeoDataFrame(geometry=[poly, MultiPolygon([small, big])])
    out = V.footprint_coords(gdf)
    assert out[0][0] == [100.123457, 13.123457]
    assert out[0][0] == out[0][-1], "rings are closed"
    assert {tuple(p) for p in out[1]} == {(1.0, 1.0), (1.002, 1.0), (1.002, 1.002), (1.0, 1.002)}


def test_footprints_of_no_buildings_is_empty():
    assert V.footprint_coords(gpd.GeoDataFrame(geometry=[])) == []
