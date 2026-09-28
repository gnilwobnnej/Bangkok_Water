"""3D maps (pydeck / deck.gl) for the buildings flood-impact tab."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pydeck as pdk
from pydeck.bindings.json_tools import default_serialize

from src import config as C
from src.buildings import ImpactMode, values_at
from src.viz import ML_CMAP, OBSERVED_COLOR, RADAR_COLOR, WATER_CMAP

MAP_STYLE = pdk.map_styles.CARTO_DARK


class CompactDeck(pdk.Deck):
    """pydeck serialises with indent=2 (every coordinate on its own line), which Streamlit sends as-is.
    Compact JSON makes large districts roughly 3x smaller and faster to load."""

    def to_json(self):
        return json.dumps(self, sort_keys=True, default=default_serialize, separators=(",", ":"))
DRY_COLOR = [150, 156, 168]
FACILITY_COLORS = {"Hospital": [255, 255, 255], "Clinic": [255, 255, 255], "School": [255, 214, 90],
                   "University": [255, 214, 90], "Police": [120, 200, 255], "Fire station": [120, 200, 255]}
BLOCK = 8  # analysis cells per side of a whole-city column (~520 m)
CITY_AFFECTED_COLOR = [255, 64, 96]
CITY_EMPTY_COLOR = [70, 76, 88]
LIGHT_RADIUS_M = 6  # light mode: every building drawn as a ~12 m square column
FOCUS_MIN_AFFECTED = 50  # open the district view on the flooded area when at least this many are affected
FOCUS_ZOOM = 13.8


def mode_colors(values: np.ndarray, mode: ImpactMode) -> np.ndarray:
    """RGB per item: grey when not affected, the mode's colour scale when affected."""
    affected = values >= mode.affected_min
    frac = np.clip(values / mode.vmax, 0, 1)
    if mode.key == "sim":
        rgb = WATER_CMAP(frac)[:, :3] * 255
    elif mode.key == "ml":
        rgb = ML_CMAP(0.35 + 0.65 * frac)[:, :3] * 255
    elif mode.key == "radar":
        rgb = np.tile(RADAR_COLOR, (len(values), 1)).astype(float)
    else:
        rgb = np.tile(OBSERVED_COLOR, (len(values), 1)).astype(float)
    out = np.tile(DRY_COLOR, (len(values), 1)).astype(float)
    out[affected] = rgb[affected]
    return out.astype(int)


def _fmt(values: np.ndarray, mode: ImpactMode) -> list:
    if mode.key == "2011":
        return np.where(values >= 1, "yes", "no").tolist()
    if mode.key == "radar":
        return [f"{v:.0f}%" for v in values]
    if mode.key == "sim":
        return [f"{v:.2f} m" if v > 0 else "dry" for v in values]
    return [f"{v:.2f}" for v in values]


def facilities_layer(fac: pd.DataFrame, mode: ImpactMode, height_scale: float):
    fac = fac.reset_index(drop=True)  # callers pass a filtered subset; align with the new columns
    v = values_at(fac, mode.grid)
    affected = v >= mode.affected_min
    data = pd.DataFrame({
        "lon": fac["lon"], "lat": fac["lat"],
        "n": fac["name"] + " (" + fac["type"] + ")",
        "h": "", "v": np.where(affected, "AFFECTED · ", "") + pd.Series(_fmt(v, mode)),
        "c": [[255, 40, 120] if a else FACILITY_COLORS.get(t, [255, 255, 255])
              for a, t in zip(affected, fac["type"])],
    })
    return pdk.Layer(
        "ColumnLayer", data, get_position=["lon", "lat"], get_elevation=60 * height_scale,
        radius=28, get_fill_color="c", pickable=True, extruded=True, disk_resolution=4,
    )


def district_deck(buildings, coords: list | None, mode: ImpactMode, height_scale: float,
                  facilities: pd.DataFrame | None, view: dict) -> pdk.Deck:
    """3D buildings coloured by flood impact. `coords` = footprint rings, or None for light columns."""
    v = values_at(buildings, mode.grid)
    colors = mode_colors(v, mode)
    base = pd.DataFrame({
        "n": "", "h": buildings["height_m"].astype("float64").round(1).to_numpy(),
        "v": _fmt(v, mode), "c": colors.tolist(),
    })
    if coords is not None:
        base["p"] = coords
        layer = pdk.Layer(
            "PolygonLayer", base, get_polygon="p", get_elevation=f"h * {height_scale}",
            get_fill_color="c", extruded=True, wireframe=False, pickable=True, auto_highlight=True,
        )
    else:
        # deck.gl's ColumnLayer takes one radius for every column, so light mode uses a typical house size.
        base["lon"], base["lat"] = buildings["lon"].round(5).to_numpy(), buildings["lat"].round(5).to_numpy()
        layer = pdk.Layer(
            "ColumnLayer", base, get_position=["lon", "lat"], get_elevation=f"h * {height_scale}",
            radius=LIGHT_RADIUS_M, get_fill_color="c", extruded=True, pickable=True, disk_resolution=4,
            angle=45,
        )
    layers = [layer]
    if facilities is not None and len(facilities):
        layers.append(facilities_layer(facilities, mode, height_scale))
    tooltip = {"html": "<b>{n}</b><br/>Height: {h} m<br/>" + mode.value_label + ": {v}",
               "style": {"fontSize": "12px"}}
    return CompactDeck(layers=layers, initial_view_state=pdk.ViewState(**view), map_style=MAP_STYLE,
                    tooltip=tooltip)


def district_view(buildings, affected: np.ndarray | None = None) -> dict:
    """Tilted camera: close in on the affected buildings if there are enough, else fit the district."""
    if affected is not None and affected.sum() >= FOCUS_MIN_AFFECTED:
        lon, lat = buildings["lon"].to_numpy()[affected], buildings["lat"].to_numpy()[affected]
        return dict(longitude=float(np.median(lon)), latitude=float(np.median(lat)), zoom=FOCUS_ZOOM,
                    pitch=55, bearing=-20)
    lon0, lon1 = buildings["lon"].min(), buildings["lon"].max()
    lat0, lat1 = buildings["lat"].min(), buildings["lat"].max()
    span = max(lon1 - lon0, (lat1 - lat0) * 1.3, 0.01)
    zoom = float(np.clip(np.log2(360 / span) + 0.2, 11, 16))
    return dict(longitude=float((lon0 + lon1) / 2), latitude=float((lat0 + lat1) / 2),
                zoom=zoom, pitch=50, bearing=-20)


def city_deck(area, counts: np.ndarray, mode: ImpactMode, height_scale: float) -> pdk.Deck:
    """Whole city as ~520 m columns: height = affected buildings, colour = share affected."""
    h, w = (C.HEIGHT // BLOCK) * BLOCK, (C.WIDTH // BLOCK) * BLOCK
    inside = (area.district_ids[:h, :w] > 0)
    cnt = counts[:h, :w] * inside
    aff = cnt * (mode.grid[:h, :w] >= mode.affected_min)

    def blocks(a):
        return a.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).sum(axis=(1, 3))
    total, affected = blocks(cnt), blocks(aff)
    br, bc = np.nonzero(total >= 1)
    share = affected[br, bc] / total[br, bc]
    data = pd.DataFrame({
        "lon": C.WEST + (bc + 0.5) * BLOCK * C.RES, "lat": C.NORTH - (br + 0.5) * BLOCK * C.RES,
        "a": affected[br, bc].round(0), "t": total[br, bc].round(0), "s": (share * 100).round(0),
    })
    # Blend grey -> red by the share affected; blocks with nothing affected stay dark grey.
    shade = np.clip(0.25 + 0.75 * share, 0, 1)[:, None]
    rgb = np.array(DRY_COLOR) * (1 - shade) + np.array(CITY_AFFECTED_COLOR) * shade
    rgb[data["a"].to_numpy() <= 0] = CITY_EMPTY_COLOR
    data["c"] = rgb.astype(int).tolist()
    layer = pdk.Layer(
        "ColumnLayer", data, get_position=["lon", "lat"], get_elevation=f"a * {2 * height_scale}",
        radius=BLOCK * C.RES * 111_000 * 0.45, get_fill_color="c", extruded=True, pickable=True,
        elevation_scale=1, disk_resolution=6,
    )
    view = dict(longitude=(C.WEST + C.EAST) / 2 + 0.03, latitude=(C.SOUTH + C.NORTH) / 2, zoom=10.2,
                pitch=55, bearing=-15)
    tooltip = {"html": "<b>{a}</b> of {t} buildings affected ({s}%)", "style": {"fontSize": "12px"}}
    return CompactDeck(layers=[layer], initial_view_state=pdk.ViewState(**view), map_style=MAP_STYLE,
                    tooltip=tooltip)


def footprint_coords(buildings) -> list:
    """Exterior rings as nested lists (cached per district by the app)."""
    return [[[round(x, 6), round(y, 6)] for x, y in g.exterior.coords] if g.geom_type == "Polygon"
            else [[round(x, 6), round(y, 6)] for x, y in max(g.geoms, key=lambda p: p.area).exterior.coords]
            for g in buildings.geometry]
