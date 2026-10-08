"""'Bathtub' flood simulation: fill terrain to a water level from rivers and canals."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import label

from src.data_loader import StudyArea


@dataclass
class FloodResult:
    level: float
    mask: np.ndarray        # bool, flooded cells
    depth: np.ndarray       # m of water above ground (0 where dry)
    area_km2: float
    people: float
    by_district: pd.DataFrame  # per district_id: flooded_km2, pct_flooded, people_affected


EIGHT = np.ones((3, 3), dtype=bool)  # 8-neighbour connectivity


def flood_mask(area: StudyArea, level: float, connected: bool = True, defences: bool = False) -> np.ndarray:
    """Cells at or below `level`; if `connected`, only those linked to a river/canal.

    Connectivity uses 8-neighbour regions so low pockets sealed off by higher ground stay dry.

    With `defences` (river-connected mode only), walls and dikes whose crest is above the level block water,
    and only rivers and canals *outside* the walls feed the flood: canals inside a walled area are drained by
    its pumps and gates. "Outside" means connected to the map edge (the Gulf and the rivers running off the
    map) without crossing a standing wall. Once the level passes a wall's crest, water flows over it.
    """
    below = area.dem <= level
    if not connected:
        return below
    sources = area.water & below
    if defences and area.defences is not None:
        standing = area.defences > level  # NaN (no defence) compares False
        below &= ~standing
        sources &= ~standing
        regions, _ = label(~standing, structure=EIGHT)
        edge = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
        sources &= np.isin(regions, edge[edge > 0])
    labels, n = label(below, structure=EIGHT)
    touched = np.unique(labels[sources])
    keep = np.zeros(n + 1, dtype=bool)
    keep[touched] = True
    keep[0] = False
    return keep[labels]


def simulate(area: StudyArea, level: float, connected: bool = True, defences: bool = False) -> FloodResult:
    mask = flood_mask(area, level, connected, defences)
    depth = np.where(mask, level - area.dem, 0).astype("float32")

    ids = area.district_ids.ravel()
    n = int(ids.max()) + 1
    flat_mask = mask.ravel()
    cell_area = area.cell_area_km2.ravel()
    total_km2 = np.bincount(ids, weights=cell_area, minlength=n)
    flooded_km2 = np.bincount(ids, weights=cell_area * flat_mask, minlength=n)
    people = np.bincount(ids, weights=area.population.ravel() * flat_mask, minlength=n)
    by_district = pd.DataFrame({
        "flooded_km2": flooded_km2[1:],
        "pct_flooded": np.divide(flooded_km2, total_km2, out=np.zeros(n), where=total_km2 > 0)[1:] * 100,
        "people_affected": people[1:],
    }, index=pd.Index(np.arange(1, n), name="district_id"))

    return FloodResult(
        level=level, mask=mask, depth=depth,
        area_km2=float(cell_area[flat_mask].sum()),
        people=float(area.population[mask].sum()),
        by_district=by_district,
    )


def level_curve(area: StudyArea, levels, connected: bool = True, defences: bool = False) -> pd.DataFrame:
    """Flooded area and affected people across a range of water levels."""
    rows = []
    for lv in levels:
        mask = flood_mask(area, lv, connected, defences)
        rows.append({
            "level": lv,
            "area_km2": float(area.cell_area_km2[mask].sum()),
            "people": float(area.population[mask].sum()),
        })
    return pd.DataFrame(rows)
