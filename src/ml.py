"""Load the trained ML flood-susceptibility outputs (see scripts/train_model.py)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio

from src import config as C

# Method names used in the report, the radar evaluation and the app
ML_ALL = "ML model (all floods)"   # 2011 + every radar wet season before config.TEST_YEARS_FROM
ML_2011 = "ML model (2011 only)"   # the original model, kept for the before/after comparison


@dataclass
class MLResult:
    prob: np.ndarray          # flood probability 0-1 per cell
    observed: np.ndarray      # 2011 labels: 1 flooded, 0 dry, 255 unknown
    report: dict
    districts: pd.DataFrame   # district_id, ml_score, observed_2011_pct, observed_known_pct
    cv: pd.DataFrame          # 2011 spatial CV: method, roc_auc, pr_auc, and roc_auc_lo/hi... (95% interval)
                              # or, for older reports, roc_auc_std/pr_auc_std (spread over the folds)
    features: pd.DataFrame    # feature, label, mean_abs_shap, direction, gain
    disagreement: np.ndarray | None = None  # std of the CV fold models' probabilities per cell


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
    disagreement = None
    if C.ML_DISAGREEMENT_FILE.exists():
        with rasterio.open(C.ML_DISAGREEMENT_FILE) as src:
            disagreement = src.read(1)
    return MLResult(
        prob=prob, observed=observed, report=report,
        districts=pd.DataFrame(report["districts"]),
        cv=cv_2011(report),
        features=pd.DataFrame(report["features"]).sort_values("mean_abs_shap", ascending=False),
        disagreement=disagreement,
    )


def cv_2011(report: dict) -> pd.DataFrame:
    """The 2011 flood's spatial CV scores: pooled out-of-fold with 95% block bootstrap intervals when the
    report has them, else the mean and standard deviation over the folds."""
    pooled = [r for r in report["cv"].get("pooled", []) if str(r["event"]).startswith("2011")]
    if not pooled:
        return pd.DataFrame(report["cv"]["summary"])
    df = pd.DataFrame(pooled)
    for key in ("roc_auc", "pr_auc"):
        df[f"{key}_lo"] = [v[0] for v in df[f"{key}_ci"]]
        df[f"{key}_hi"] = [v[1] for v in df[f"{key}_ci"]]
    return df[["method", "roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "pr_auc_lo", "pr_auc_hi"]]
