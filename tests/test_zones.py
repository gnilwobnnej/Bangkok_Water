import json

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from src.ml import ML_ALL, ML_STREET
from src.validation import zone_scores
from src.viz import ML_CMAP, STREET_CMAP, best_model_rgba, probability_rgba
from src.zones import CITY_SCOPE, OUTSIDE_SCOPE, city_zone, zone_outline


def test_city_zone_smooths_speckle_fills_holes_and_stays_in_bangkok():
    built = np.zeros((40, 40), dtype="uint8")
    built[5:25, 5:25] = 100     # a dense block...
    built[14, 14] = 0           # ...with a small open gap (a park)
    built[35, 35] = 100         # a lone dense cell far away
    inside = np.ones_like(built, bool)
    inside[:, 20:] = False      # the block's east part is outside Bangkok
    zone = city_zone(built, inside, cell_km2=0.01, window=5, min_km2=0.5)
    assert zone[14, 14]                     # the gap is filled
    assert not zone[35, 35]                 # the lone cell is dropped
    assert not zone[inside == False].any()  # noqa: E712
    assert zone[10:20, 10:19].all()


def test_city_zone_drops_pieces_below_the_minimum_size():
    built = np.zeros((30, 30), dtype="uint8")
    built[2:8, 2:8] = 100       # 36 cells = 0.36 km²
    built[12:28, 12:28] = 100   # 256 cells = 2.56 km²
    zone = city_zone(built, np.ones_like(built, bool), cell_km2=0.01, window=3, min_km2=1.0)
    assert not zone[2:8, 2:8].any() and zone[15:25, 15:25].all()


def test_zone_outline_is_a_line_around_the_zone():
    zone = np.zeros((10, 10), bool)
    zone[2:6, 3:8] = True
    gj = json.loads(zone_outline(zone, from_origin(100.0, 14.0, 0.01, 0.01), tolerance=0))
    geom = gj["features"][0]["geometry"]
    assert geom["type"] in ("LineString", "MultiLineString")
    xs = [x for x, _ in (geom["coordinates"] if geom["type"] == "LineString" else geom["coordinates"][0])]
    assert min(xs) == 100.03 and max(xs) == 100.08
    assert json.loads(zone_outline(np.zeros((5, 5), bool), from_origin(0, 0, 1, 1)))["features"] == []


def test_best_model_shows_street_model_in_the_city_and_river_model_elsewhere():
    river = np.full((2, 2), 0.8, "float32")
    street = np.array([[0.3, np.nan], [0.3, np.nan]], "float32")
    zone = np.array([[True, True], [False, False]])
    rgba = best_model_rgba(river, street, zone)
    green, pink = probability_rgba(street, cmap=STREET_CMAP), probability_rgba(river, cmap=ML_CMAP)
    assert (rgba[0, 0] == green[0, 0]).all()   # city: street model
    assert (rgba[0, 1] == pink[0, 1]).all()    # city, but no street prediction: river model
    assert (rgba[1, 0] == pink[1, 0]).all()    # outside: river model, even where the street model has a value


def test_zone_scores_puts_each_zone_on_its_own_test():
    def rows(scope, date, kind, scores):
        return [{"date": date, "kind": kind, "scope": scope, "method": m, "roc_auc": v} for m, v in scores.items()]
    street = pd.DataFrame(rows(CITY_SCOPE, "2025", "year", {ML_STREET: 0.7, ML_ALL: 0.6})
                          + rows(CITY_SCOPE, "2026", "year", {ML_STREET: 0.78, ML_ALL: 0.64})
                          + rows(OUTSIDE_SCOPE, "2026", "year", {ML_STREET: 0.71, ML_ALL: 0.58}))
    radar = pd.DataFrame(rows(OUTSIDE_SCOPE, "2026-09-01", "pass", {ML_STREET: 0.4, ML_ALL: 0.9})
                         + rows(OUTSIDE_SCOPE, "2026-09-05", "pass", {ML_STREET: 0.5, ML_ALL: 0.95})
                         + rows("Greater Bangkok", "2026-09-01", "pass", {ML_ALL: 0.1}))
    zs = zone_scores(radar, street, [ML_STREET, ML_ALL, "Bathtub + defences"], CITY_SCOPE, OUTSIDE_SCOPE)
    assert list(zs.columns) == [ML_STREET, ML_ALL]
    city, outside = zs.iloc[0], zs.iloc[1]
    assert "2026" in zs.index[0] and city[ML_STREET] == 0.78 and city[ML_ALL] == 0.64
    assert "2 passes" in zs.index[1] and outside[ML_ALL] == 0.925 and outside[ML_STREET] == 0.45
    assert zone_scores(None, None, [ML_ALL], CITY_SCOPE, OUTSIDE_SCOPE).empty
