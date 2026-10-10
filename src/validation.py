"""Validation: confidence intervals for the model scores, and the tables behind the app's Validation tab.

Scores are bootstrapped by resampling whole spatial blocks (4 km, as in the model's cross-validation),
because neighbouring cells flood together and aren't independent. Each resample reuses the same blocks for
every method, so the difference between two methods gets its own interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml import ML_2011

N_BOOT = 200
CI = 95
REFERENCE = "Bathtub simulation"   # the physics baseline every other method is compared with
# How a row of the validation table was tested
TEST_CV = "Spatial CV (training flood, unseen blocks)"
TEST_LOEO = "Left-out flood"
TEST_RADAR = "Recent radar pass (never trained on)"


def score_bins(s: np.ndarray) -> np.ndarray:
    """Rank of each score among the distinct values (0 = lowest). Tied scores share a bin."""
    return np.unique(s, return_inverse=True)[1].ravel()


def weighted_scores(bins: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and average precision (PR-AUC) with per-cell weights.

    With all weights 1 these equal sklearn's roc_auc_score and average_precision_score. A weight of k counts
    a cell k times, which is how a bootstrap resample is applied without copying the data.
    """
    n = int(bins.max()) + 1
    pos = np.bincount(bins, weights=w * y, minlength=n)
    neg = np.bincount(bins, weights=w * (1 - y), minlength=n)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan, np.nan
    # ROC-AUC: chance a flooded cell outranks a dry one (ties count half)
    neg_below = np.cumsum(neg) - neg
    roc = float((pos * (neg_below + neg / 2)).sum() / (P * N))
    # Average precision: precision at each threshold (highest scores first), weighted by the recall it adds
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    flooded_here = pos[::-1] > 0
    ap = float((pos[::-1][flooded_here] / P * tp[flooded_here] / (tp + fp)[flooded_here]).sum())
    return roc, ap


def block_bootstrap(y: np.ndarray, scores: dict, blocks: np.ndarray, n_boot: int = N_BOOT,
                    seed: int = 0, reference: str = REFERENCE) -> dict:
    """95% intervals for each method's ROC-AUC and PR-AUC, and for its difference from `reference`.

    y: 1 flooded / 0 dry per cell; scores: method -> score per cell (higher = more likely to flood);
    blocks: spatial block id per cell. Returns {method: {"roc_auc_ci": [lo, hi], "pr_auc_ci": [lo, hi],
    and, except for the reference, "roc_auc_diff_ci", "pr_auc_diff_ci"}}.
    """
    y = np.asarray(y, dtype="float64")
    block_ids, cell_block = np.unique(blocks, return_inverse=True)
    cell_block = cell_block.ravel()
    bins = {m: score_bins(np.asarray(s)) for m, s in scores.items()}
    rng = np.random.default_rng(seed)
    draws = {m: np.full((n_boot, 2), np.nan) for m in scores}
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, len(block_ids), len(block_ids)), minlength=len(block_ids))
        w = counts[cell_block].astype("float64")
        for m in scores:
            draws[m][i] = weighted_scores(bins[m], y, w)

    lo, hi = (100 - CI) / 2, 100 - (100 - CI) / 2

    def interval(a):
        a = a[~np.isnan(a)]
        return [float(np.percentile(a, lo)), float(np.percentile(a, hi))] if len(a) else [np.nan, np.nan]

    out = {}
    for m, d in draws.items():
        out[m] = {"roc_auc_ci": interval(d[:, 0]), "pr_auc_ci": interval(d[:, 1])}
        if reference in draws and m != reference:
            diff = d - draws[reference]
            out[m]["roc_auc_diff_ci"] = interval(diff[:, 0])
            out[m]["pr_auc_diff_ci"] = interval(diff[:, 1])
    return out


def cv_table(report: dict) -> pd.DataFrame:
    """The cross-validation scores of every training event, in the same shape as the radar results
    (spread = fold min-max). Reports from before multi-flood training have only the 2011 folds."""
    if "per_event_folds" in report["cv"]:
        folds = pd.DataFrame(report["cv"]["per_event_folds"])
        events = {e["name"]: e for e in report["events"]}
    else:
        folds = pd.DataFrame(report["cv"]["folds_detail"]).assign(event="2011 flood (MODIS)")
        folds["method"] = folds["method"].where(~folds["method"].str.startswith("ML model"), ML_2011)
        events = {"2011 flood (MODIS)": {"year": 2011, "flooded_share": report["flooded_share"],
                                         "n_cells": report["n_training_cells"]}}
    rows = []
    for (event, m), g in folds.groupby(["event", "method"], sort=False):
        e = events[event]
        rows.append({
            "event": event, "date": str(e["year"]), "scope": "Greater Bangkok", "method": m,
            "roc_auc": g["roc_auc"].mean(), "roc_auc_lo": g["roc_auc"].min(), "roc_auc_hi": g["roc_auc"].max(),
            "pr_auc": g["pr_auc"].mean(), "pr_auc_lo": g["pr_auc"].min(), "pr_auc_hi": g["pr_auc"].max(),
            "flooded_share": e["flooded_share"], "n_cells": e["n_cells"],
            "interval": f"range over {report['cv']['folds']} CV folds",
        })
    return pd.DataFrame(rows)


def loeo_table(report: dict | None) -> pd.DataFrame:
    """Leave-one-event-out scores: each flood predicted by a model trained on the other floods, with 95% block
    bootstrap intervals. Empty for a report without multi-flood training."""
    rows = (report or {}).get("leave_one_event_out", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for key in ("roc_auc", "pr_auc"):
        for suffix, col in (("", f"{key}_ci"), ("_diff", f"{key}_diff_ci")):
            vals = df[col] if col in df else pd.Series([None] * len(df))
            lohi = [v if isinstance(v, (list, tuple)) else (np.nan, np.nan) for v in vals]
            df[f"{key}{suffix}_lo"] = [v[0] for v in lohi]
            df[f"{key}{suffix}_hi"] = [v[1] for v in lohi]
    year = {e["name"]: e["year"] for e in report["events"]}
    df["date"] = df["event"].map(year).astype(str)
    df["scope"] = "Greater Bangkok"
    df["random_pr_auc"] = df["flooded_share"]
    return df.drop(columns=[c for c in df if c.endswith("_ci")])


def validation_table(radar_eval: pd.DataFrame | None, report: dict | None) -> pd.DataFrame:
    """Every method x every flood event: scores, 95% intervals, random-guess PR-AUC, cells, flooded share."""
    parts = []
    if report is not None:
        parts.append(cv_table(report).assign(test=TEST_CV))
        loeo = loeo_table(report)
        if not loeo.empty:
            parts.append(loeo.assign(interval=f"{CI}% block bootstrap", test=TEST_LOEO))
    if radar_eval is not None and not radar_eval.empty:
        ev = radar_eval.copy()
        ev["event"] = "Radar " + ev["date"]
        ev["interval"] = np.where(ev["roc_auc_lo"].notna(), f"{CI}% block bootstrap", "")
        parts.append(ev.assign(test=TEST_RADAR))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["random_pr_auc"] = df["flooded_share"]
    cols = ["test", "event", "date", "scope", "method", "roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc",
            "pr_auc_lo", "pr_auc_hi", "random_pr_auc", "flooded_share", "n_cells", "interval",
            "roc_auc_diff_lo", "roc_auc_diff_hi"]
    return df.reindex(columns=cols)


def wins(radar_eval: pd.DataFrame, method: str, scope: str, metric: str = "roc_auc",
         reference: str = REFERENCE) -> dict:
    """How often `method` beats `reference` on the radar passes (or any table with date/scope/method rows): in
    total, and clearly (interval of the difference above zero) or clearly worse (below zero). The stored
    difference intervals are against REFERENCE, so for any other reference only the total is counted."""
    ev = radar_eval[radar_eval["scope"] == scope]
    a = ev[ev["method"] == method].set_index("date")
    b = ev[ev["method"] == reference].set_index("date")[metric]
    dates = a.index.intersection(b.index)
    diff = a.loc[dates, metric] - b.loc[dates]
    none = pd.Series(np.nan, index=dates)
    lo = a.loc[dates].get(f"{metric}_diff_lo", none) if reference == REFERENCE else none
    hi = a.loc[dates].get(f"{metric}_diff_hi", none) if reference == REFERENCE else none
    return {"passes": len(dates), "better": int((diff > 0).sum()),
            "clearly_better": int((lo > 0).sum()), "clearly_worse": int((hi < 0).sum()),
            "has_intervals": bool(lo.notna().any())}
