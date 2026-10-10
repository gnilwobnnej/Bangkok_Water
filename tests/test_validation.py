import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src.current_flood import load_evaluation
from src.ml import ML_2011, ML_ALL
from src.validation import (TEST_CV, TEST_LOEO, TEST_RADAR, block_bootstrap, loeo_table, score_bins,
                            validation_table, weighted_scores, wins)


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
    assert list(t["test"]) == [TEST_CV, TEST_RADAR]
    ml2011 = t.iloc[0]
    assert ml2011["method"] == ML_2011  # an old report's single ML model was the 2011-only one
    assert (ml2011["roc_auc"], ml2011["roc_auc_lo"], ml2011["roc_auc_hi"]) == pytest.approx((0.85, 0.8, 0.9))
    assert t.iloc[1]["random_pr_auc"] == 0.1
    assert t.iloc[1]["interval"] == ""


def test_wins_against_another_model_ignores_the_bathtub_intervals():
    ev = eval_frame()
    ev.loc[ev["method"] == "Bathtub simulation", "method"] = "Other"
    w = wins(ev, "ML", "Greater Bangkok", reference="Other")
    assert w == {"passes": 3, "better": 2, "clearly_better": 0, "clearly_worse": 0, "has_intervals": False}


def multi_flood_report():
    events = [{"id": "2011", "year": 2011, "name": "2011 flood (MODIS)", "flooded_share": 0.2, "n_cells": 500},
              {"id": "radar_2017", "year": 2017, "name": "2017 wet season (radar)", "flooded_share": 0.03,
               "n_cells": 400}]
    folds = [{"fold": f, "event": e["name"], "method": m, "roc_auc": a + f / 100, "pr_auc": 0.5}
             for f in (1, 2) for e in events for m, a in [(ML_ALL, 0.8), ("Bathtub simulation", 0.6)]]
    loeo = [{"event": "2017 wet season (radar)", "method": m, "roc_auc": a, "pr_auc": 0.1,
             "roc_auc_ci": [a - 0.05, a + 0.05], "pr_auc_ci": [0.05, 0.15], "flooded_share": 0.03, "n_cells": 400,
             **({"roc_auc_diff_ci": [0.1, 0.3], "pr_auc_diff_ci": [0.0, 0.1]} if m != "Bathtub simulation" else {})}
            for m, a in [(ML_ALL, 0.85), (ML_2011, 0.8), ("Bathtub simulation", 0.65)]]
    return {"event": {"began": "2011-08-05"}, "flooded_share": 0.2, "n_training_cells": 900, "events": events,
            "cv": {"folds": 2, "folds_detail": [], "per_event_folds": folds}, "leave_one_event_out": loeo}


def test_validation_table_has_every_training_flood_and_the_left_out_floods():
    t = validation_table(None, multi_flood_report())
    cv = t[t["test"] == TEST_CV].set_index(["event", "method"])
    assert len(cv) == 4
    row = cv.loc[("2017 wet season (radar)", ML_ALL)]
    assert (row["roc_auc"], row["roc_auc_lo"], row["roc_auc_hi"]) == pytest.approx((0.815, 0.81, 0.82))
    assert row["flooded_share"] == 0.03 and row["date"] == "2017"
    lo = t[t["test"] == TEST_LOEO].set_index("method")
    assert list(lo.index) == [ML_ALL, ML_2011, "Bathtub simulation"]
    assert (lo.loc[ML_ALL, "roc_auc_lo"], lo.loc[ML_ALL, "roc_auc_hi"]) == pytest.approx((0.8, 0.9))
    assert lo.loc[ML_ALL, "roc_auc_diff_lo"] == pytest.approx(0.1)
    assert np.isnan(lo.loc["Bathtub simulation", "roc_auc_diff_lo"])


def test_loeo_table_counts_wins_per_flood():
    loeo = loeo_table(multi_flood_report())
    assert wins(loeo, ML_ALL, "Greater Bangkok")["clearly_better"] == 1
    assert wins(loeo, ML_ALL, "Greater Bangkok", reference=ML_2011)["better"] == 1
    assert loeo_table(None).empty and loeo_table({"cv": {}}).empty


def test_cv_table_prefers_pooled_scores_with_bootstrap_intervals():
    report = multi_flood_report()
    report["cv"]["pooled"] = [
        {"event": "2011 flood (MODIS)", "method": ML_ALL, "roc_auc": 0.86, "pr_auc": 0.6,
         "roc_auc_ci": [0.84, 0.88], "pr_auc_ci": [0.55, 0.65], "roc_auc_diff_ci": [0.1, 0.2],
         "pr_auc_diff_ci": [0.0, 0.1], "flooded_share": 0.2, "n_cells": 500},
        {"event": "2011 flood (MODIS)", "method": "Bathtub simulation", "roc_auc": 0.7, "pr_auc": 0.4,
         "roc_auc_ci": [0.68, 0.72], "pr_auc_ci": [0.35, 0.45], "flooded_share": 0.2, "n_cells": 500},
    ]
    cv = validation_table(None, report)
    cv = cv[cv["test"] == TEST_CV].set_index("method")
    assert len(cv) == 2 and cv.loc[ML_ALL, "date"] == "2011"
    assert (cv.loc[ML_ALL, "roc_auc"], cv.loc[ML_ALL, "roc_auc_lo"], cv.loc[ML_ALL, "roc_auc_hi"]) == \
        pytest.approx((0.86, 0.84, 0.88))
    assert cv.loc[ML_ALL, "roc_auc_diff_lo"] == pytest.approx(0.1)
    assert "bootstrap" in cv.loc[ML_ALL, "interval"]


def test_ml_cv_2011_uses_pooled_intervals_or_falls_back_to_fold_spread():
    from src.ml import cv_2011
    report = multi_flood_report()
    report["cv"]["summary"] = [{"method": ML_ALL, "roc_auc": 0.8, "roc_auc_std": 0.01, "pr_auc": 0.5,
                                "pr_auc_std": 0.02}]
    assert "roc_auc_std" in cv_2011(report)
    report["cv"]["pooled"] = [{"event": "2011 flood (MODIS)", "method": ML_ALL, "roc_auc": 0.86, "pr_auc": 0.6,
                               "roc_auc_ci": [0.84, 0.88], "pr_auc_ci": [0.55, 0.65]},
                              {"event": "2017 wet season (radar)", "method": ML_ALL, "roc_auc": 0.9,
                               "pr_auc": 0.3, "roc_auc_ci": [0.8, 0.95], "pr_auc_ci": [0.2, 0.4]}]
    cv = cv_2011(report)
    assert list(cv["method"]) == [ML_ALL] and cv["roc_auc_lo"].iloc[0] == 0.84
