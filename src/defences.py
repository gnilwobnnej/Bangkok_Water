"""Flood defences (river walls and dikes): turn lines into a crest-height grid, and load them for the map.

Built by scripts/prepare_defences.py. src/simulate.py uses the grid (StudyArea.defences) to block water.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from src import config as C


def crest_grid(lines: gpd.GeoDataFrame, shape: tuple, transform) -> np.ndarray:
    """Crest height (m) per cell under each line, NaN elsewhere; where lines overlap, the higher crest wins.

    all_touched=True marks every cell a line passes through, so each line is a chain of cells that share
    edges. Water spreading to 8 neighbours can't slip diagonally between them, which it could through a
    line drawn one cell per step.
    """
    crest = np.full(shape, np.nan, dtype="float32")
    for _, f in lines.sort_values("crest_m").iterrows():  # higher crests drawn last, so they win
        cells = rasterize([(f.geometry, 1)], out_shape=shape, transform=transform, fill=0,
                          all_touched=True, dtype="uint8").astype(bool)
        crest[cells] = f["crest_m"]
    return crest


def load_defence_lines() -> gpd.GeoDataFrame | None:
    """The defence lines (name, crest_m, source, note), or None if scripts/prepare_defences.py hasn't run."""
    if not C.DEFENCE_LINES_FILE.exists():
        return None
    return gpd.read_file(C.DEFENCE_LINES_FILE)
