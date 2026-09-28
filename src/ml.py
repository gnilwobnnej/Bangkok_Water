"""Load the trained ML flood-susceptibility outputs (see scripts/train_model.py)."""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio

from src import config as C


@dataclass
class MLResult:
    prob: np.ndarray          # flood probability 0-1 per cell
    observed: np.ndarray      # 2011 labels: 1 flooded, 0 dry, 255 unknown
    report: dict
    districts: pd.DataFrame   # district_id, ml_score, observed_2011_pct, observed_known_pct
    cv: pd.DataFrame          # method, roc_auc, roc_auc_std, pr_auc, pr_auc_std
    features: pd.DataFrame    # feature, label, mean_abs_shap, direction, gain


def ml_available() -> bool:
    return all(f.exists() for f in (C.ML_PROB_FILE, C.ML_REPORT_FILE, C.FLOOD_2011_FILE))


def load_ml():
    """Return MLResult, or None if the model hasn't been trained yet."""
    if not ml_available():
        return None
    with rasterio.open(C.ML_PROB_FILE) as src:
        prob = src.read(1)
    with rasterio.open(C.FLOOD_2011_FILE) as src:
        observed = src.read(1)
    report = json.loads(C.ML_REPORT_FILE.read_text())
    return MLResult(
        prob=prob, observed=observed, report=report,
        districts=pd.DataFrame(report["districts"]),
        cv=pd.DataFrame(report["cv"]["summary"]),
        features=pd.DataFrame(report["features"]).sort_values("mean_abs_shap", ascending=False),
    )
