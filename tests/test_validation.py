import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src.current_flood import load_evaluation
from src.validation import block_bootstrap, score_bins, validation_table, weighted_scores, wins


@pytest.fixture
def scored():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.2).astype(int)
    good = y + rng.normal(0, 0.8, y.size)               # informative, continuous
    coarse = np.round(y + rng.normal(0, 1.5, y.size))   # weaker, many ties
    blocks = np.repeat(np.arange(40), 50)
    return y, {"Good": good, "Bathtub simulation": coarse}, blocks


def test_weighted_scores_match_sklearn(scored):
    y, scores, _ = scored
    for s in scores.values():
        roc, ap = weighted_scores(score_bins(s), y, np.ones(y.size))
        assert roc == pytest.approx(roc_auc_score(y, s))
        assert ap == pytest.approx(average_precision_score(y, s))


def test_integer_weights_equal_repeated_cells(scored):
    y, scores, _ = scored
    s = scores["Bathtub simulation"]
    w = np.random.default_rng(2).integers(0, 4, y.size)
    roc, ap = weighted_scores(score_bins(s), y, w.astype(float))
    rep_y, rep_s = np.repeat(y, w), np.repeat(s, w)
    assert roc == pytest.approx(roc_auc_score(rep_y, rep_s))
    assert ap == pytest.approx(average_precision_score(rep_y, rep_s))


def test_one_class_gives_nan():
    roc, ap = weighted_scores(np.array([0, 1, 2]), np.array([1.0, 1, 1]), np.ones(3))
    assert np.isnan(roc) and np.isnan(ap)


def test_bootstrap_intervals(scored):
    y, scores, blocks = scored
    out = block_bootstrap(y, scores, blocks, n_boot=100, seed=3)
    for m, s in scores.items():
        lo, hi = out[m]["roc_auc_ci"]
        assert lo <= roc_auc_score(y, s) <= hi
        assert lo < hi
    # the reference has no difference from itself; the better method is clearly better than it
    assert "roc_auc_diff_ci" not in out["Bathtub simulation"]
    assert out["Good"]["roc_auc_diff_ci"][0] > 0
    assert out == block_bootstrap(y, scores, blocks, n_boot=100, seed=3)  # reproducible


def eval_frame():
    rows = []
    for date, ml, bt, lo, hi in [("2026-09-01", 0.8, 0.7, 0.05, 0.15), ("2026-09-05", 0.72, 0.7, -0.03, 0.07),
                                 ("2026-09-09", 0.6, 0.7, -0.15, -0.05)]:
        rows.append({"date": date, "scope": "Greater Bangkok", "method": "ML", "roc_auc": ml,
                     "roc_auc_diff_lo": lo, "roc_auc_diff_hi": hi})
        rows.append({"date": date, "scope": "Greater Bangkok", "method": "Bathtub simulation", "roc_auc": bt,
                     "roc_auc_diff_lo": np.nan, "roc_auc_diff_hi": np.nan})
    return pd.DataFrame(rows)


def test_wins_counts():
    w = wins(eval_frame(), "ML", "Greater Bangkok")
    assert w == {"passes": 3, "better": 2, "clearly_better": 1, "clearly_worse": 1, "has_intervals": True}


def test_validation_table_and_old_eval_files(tmp_path, monkeypatch):
    # an evaluation.json from before intervals were added still loads, with blank intervals
    f = tmp_path / "evaluation.json"
    f.write_text(json.dumps({"results": [{"date": "2026-09-01", "scope": "Greater Bangkok", "n_cells": 100,
                                          "flooded_share": 0.1,
                                          "methods": {"ML": {"roc_auc": 0.8, "pr_auc": 0.3}}}]}))
    monkeypatch.setattr(C, "RADAR_EVAL_FILE", f)
    ev = load_evaluation()
    assert np.isnan(ev.loc[0, "roc_auc_lo"])
    report = {"event": {"began": "2011-08-05"}, "flooded_share": 0.2, "n_training_cells": 500,
              "cv": {"folds": 2, "folds_detail": [
                  {"fold": 1, "method": "ML model (LightGBM)", "roc_auc": 0.8, "pr_auc": 0.6},
                  {"fold": 2, "method": "ML model (LightGBM)", "roc_auc": 0.9, "pr_auc": 0.7}]}}
    t = validation_table(ev, report)
    assert list(t["event"]) == ["2011 flood (MODIS)", "Radar 2026-09-01"]
    ml2011 = t.iloc[0]
    assert ml2011["method"] == "ML model (trained on 2011)"
    assert (ml2011["roc_auc"], ml2011["roc_auc_lo"], ml2011["roc_auc_hi"]) == pytest.approx((0.85, 0.8, 0.9))
    assert t.iloc[1]["random_pr_auc"] == 0.1
    assert t.iloc[1]["interval"] == ""
