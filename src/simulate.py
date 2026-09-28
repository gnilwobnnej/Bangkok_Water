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


def flood_mask(area: StudyArea, level: float, connected: bool = True) -> np.ndarray:
    """Cells at or below `level`; if `connected`, only those linked to a river/canal.

    Connectivity uses 8-neighbour regions so low pockets sealed off by higher ground stay dry.
    """
    below = area.dem <= level
    if not connected:
        return below
    labels, n = label(below, structure=np.ones((3, 3), dtype=bool))
    touched = np.unique(labels[area.water & below])
    keep = np.zeros(n + 1, dtype=bool)
    keep[touched] = True
    keep[0] = False
    return keep[labels]


def simulate(area: StudyArea, level: float, connected: bool = True) -> FloodResult:
    mask = flood_mask(area, level, connected)
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


def level_curve(area: StudyArea, levels, connected: bool = True) -> pd.DataFrame:
    """Flooded area and affected people across a range of water levels."""
    rows = []
    for lv in levels:
        mask = flood_mask(area, lv, connected)
        rows.append({
            "level": lv,
            "area_km2": float(area.cell_area_km2[mask].sum()),
            "people": float(area.population[mask].sum()),
        })
    return pd.DataFrame(rows)
