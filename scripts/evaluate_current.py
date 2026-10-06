"""Test every method against the recent radar flood maps (floods none of them was trained on).

    python scripts/evaluate_current.py

Needs data/processed/radar_flood/ (scripts/fetch_current_flood.py). The ML model is included if it has
been trained (scripts/train_model.py). A cell counts as flooded when at least half of it is flooded in the
radar map, and as dry when under 10% is flooded, it isn't permanent water and radar saw it. Cells in
between are left out.

Output: data/processed/radar_flood/evaluation.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C  # noqa: E402
from src.current_flood import DRY_PCT, FLOODED_PCT, load_pass, radar_labels  # noqa: E402
from src.data_loader import load_study_area  # noqa: E402
from src.risk import compute_factors, risk_index  # noqa: E402
from train_model import bathtub_onset  # noqa: E402


def score(y, s) -> dict:
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def main():
    summary = json.loads(C.RADAR_SUMMARY_FILE.read_text())
    area = load_study_area()
    bkk = area.district_ids > 0
    methods = {
        "Bathtub simulation": -bathtub_onset(area),
        "Risk index (default weights)": risk_index(compute_factors(area), 0.5, 0.3, 0.2),
    }
    if C.ML_PROB_FILE.exists():
        with rasterio.open(C.ML_PROB_FILE) as src:
            methods = {"ML model (trained on 2011)": src.read(1), **methods}

    results = []
    for p in summary["passes"]:
        labels = radar_labels(load_pass(p["date"]))
        for scope, mask in [("Greater Bangkok", np.ones_like(bkk)), ("Bangkok (50 districts)", bkk)]:
            known = (labels >= 0) & mask
            y = labels[known]
            if y.sum() < 20 or (y == 0).sum() < 20:
                continue
            rec = {"date": p["date"], "scope": scope, "n_cells": int(known.sum()),
                   "flooded_share": float(y.mean()), "methods": {}}
            for name, s in methods.items():
                rec["methods"][name] = score(y, s[known])
            results.append(rec)

    C.RADAR_EVAL_FILE.write_text(json.dumps({
        "label_rule": {"flooded_pct_at_least": FLOODED_PCT, "dry_pct_below": DRY_PCT},
        "baseline": summary["method"]["baseline"],
        "results": results,
    }, indent=2))

    latest = summary["passes"][-1]["date"]
    for rec in [r for r in results if r["date"] == latest]:
        print(f"\n{latest}, {rec['scope']}: {rec['n_cells']:,} cells, {rec['flooded_share']:.1%} flooded "
              f"(PR-AUC of a random guess = {rec['flooded_share']:.3f})")
        for name, m in rec["methods"].items():
            print(f"  {name:30s} ROC-AUC {m['roc_auc']:.3f}   PR-AUC {m['pr_auc']:.3f}")
    print("\nROC-AUC over time (greater Bangkok):")
    for rec in [r for r in results if r["scope"] == "Greater Bangkok"]:
        print(f"  {rec['date']}  " + "  ".join(f"{n.split(' (')[0]}: {m['roc_auc']:.2f}" for n, m in rec["methods"].items()))
    print(f"\nWrote {C.RADAR_EVAL_FILE}")


if __name__ == "__main__":
    main()
