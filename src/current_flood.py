"""Load the Sentinel-1 radar flood maps (scripts/fetch_current_flood.py, scripts/evaluate_current.py)."""
import json

import numpy as np
import pandas as pd
import rasterio

from src import config as C
from src.data_loader import StudyArea


def radar_available() -> bool:
    return C.RADAR_SUMMARY_FILE.exists()


def load_summary():
    """(passes DataFrame, rainfall DataFrame, full summary dict), or None if not fetched yet."""
    if not radar_available():
        return None
    summary = json.loads(C.RADAR_SUMMARY_FILE.read_text())
    passes = pd.DataFrame(summary["passes"])
    rain = pd.DataFrame(summary.get("rainfall", []), columns=["date", "mm"])
    return passes, rain, summary


def load_evaluation():
    """Per-date scores of each method against the radar maps, or None."""
    if not C.RADAR_EVAL_FILE.exists():
        return None
    ev = json.loads(C.RADAR_EVAL_FILE.read_text())
    rows = [{"date": r["date"], "scope": r["scope"], "flooded_share": r["flooded_share"],
             "method": m, **scores}
            for r in ev["results"] for m, scores in r["methods"].items()]
    return pd.DataFrame(rows)


def load_pass(date: str) -> dict:
    """% flooded, % permanent water and % with radar data per cell for one pass date."""
    with rasterio.open(C.RADAR_DIR / f"{date}.tif") as src:
        return {"flood_pct": src.read(1), "permanent_pct": src.read(2), "valid_pct": src.read(3)}


def district_flooded_pct(area: StudyArea, radar: dict) -> pd.DataFrame:
    """Share of each district's radar-visible land flooded on that pass."""
    ids = area.district_ids.ravel()
    n = int(ids.max()) + 1
    cell = area.cell_area_km2.ravel()
    seen = cell * (radar["valid_pct"].ravel() / 100)
    flooded = cell * (radar["flood_pct"].ravel() / 100)
    seen_km2 = np.bincount(ids, weights=seen, minlength=n)
    flooded_km2 = np.bincount(ids, weights=flooded, minlength=n)
    pct = np.divide(flooded_km2, seen_km2, out=np.zeros(n), where=seen_km2 > 0) * 100
    return pd.DataFrame({"district_id": np.arange(1, n), "flooded_now_pct": pct[1:],
                         "flooded_now_km2": flooded_km2[1:]})
