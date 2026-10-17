"""Test every method against street flooding reported by residents (Traffy Fondue), in years none was trained on.

    python scripts/evaluate_street.py

Needs the street-flood labels (scripts/build_traffy_labels.py). Scores each method on the test years
(config.TEST_YEARS_FROM on), as a whole and storm by storm:
    year    flooded = a flood report on any storm day of the year; dry = no flood report all year, but at least
            TRAFFY_MIN_OTHER_REPORTS reports of other problems (as in the training labels)
    storm   flooded = a flood report on that day (or the next, as reports trail the rain); dry = as above
in Bangkok's 50 districts and in the dense city (cells >= config.DENSE_BUILT_UP_PCT built-up), which radar can't
test at all; and in the city zone (src/zones.py) and the rest of the 50 districts, the map's "Best model by area" split.

Each score gets a 95% block bootstrap interval, as in evaluate_current.py.

Output: data/processed/traffy/evaluation.json (the same layout as radar_flood/evaluation.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_current import score  # noqa: E402
from src import config as C  # noqa: E402
from src.data_loader import load_study_area  # noqa: E402
from src.ml import ML_2011, ML_ALL, ML_STREET  # noqa: E402
from src.traffy import count_grid, load_reports, reports_in  # noqa: E402
from src.validation import N_BOOT, block_bootstrap  # noqa: E402
from src.zones import CITY_SCOPE, OUTSIDE_SCOPE, load_city_zone  # noqa: E402
from train_model import BLOCK_M, baseline_methods, spatial_blocks  # noqa: E402

MIN_CELLS = 20  # an event needs at least this many flooded and dry cells to be scored


def main():
    area = load_study_area()
    bkk = area.district_ids > 0
    with rasterio.open(C.LANDCOVER_FILE) as src:
        built_up = src.read(1 + list(src.descriptions).index("built_up"))
    scopes = [("Bangkok (50 districts)", bkk), ("Dense city", bkk & (built_up >= C.DENSE_BUILT_UP_PCT))]
    zone = load_city_zone(area)  # the map's "Best model by area" split
    scopes += [(CITY_SCOPE, zone), (OUTSIDE_SCOPE, bkk & ~zone)]
    blocks = spatial_blocks().reshape(bkk.shape)
    methods = baseline_methods(area)
    for name, path in [(ML_2011, C.ML_PROB_2011_FILE), (ML_ALL, C.ML_PROB_FILE), (ML_STREET, C.ML_STREET_PROB_FILE)]:
        if path.exists():
            with rasterio.open(path) as src:
                methods = {name: src.read(1), **methods}

    summary = json.loads(C.TRAFFY_STORMS_FILE.read_text(encoding="utf-8"))
    storms = pd.to_datetime([s["date"] for s in summary["storms"]])
    reports = load_reports()

    events = []  # (date label, kind, labels)
    for y in [y for y in C.TRAFFY_YEARS if y >= C.TEST_YEARS_FROM]:
        path = C.FLOOD_LABELS_DIR / f"traffy_{y}.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            year_labels = src.read(1)
        events.append((str(y), "year", year_labels))
        dry = year_labels == 0
        for d in storms[storms.year == y]:
            r = reports_in(reports, str(d.date()))
            labels = np.full(year_labels.shape, C.LABEL_NODATA, dtype="uint8")
            labels[dry] = 0
            labels[count_grid(r["lon"], r["lat"]) > 0] = 1
            events.append((str(d.date()), "storm", labels))

    results = []
    for date, kind, labels in events:
        for scope, mask in scopes:
            known = (labels != C.LABEL_NODATA) & mask
            for s in methods.values():
                known &= np.isfinite(s)
            y = (labels[known] == 1).astype(int)
            if y.sum() < MIN_CELLS or (y == 0).sum() < MIN_CELLS:
                continue
            rec = {"date": date, "kind": kind, "scope": scope, "n_cells": int(known.sum()),
                   "flooded_share": float(y.mean()), "methods": {}}
            for name, s in methods.items():
                rec["methods"][name] = score(y, s[known])
            ci = block_bootstrap(y, {n: s[known] for n, s in methods.items()}, blocks[known])
            for name in methods:
                rec["methods"][name].update(ci[name])
            results.append(rec)
            print(f"  {date} {scope}: {int(y.sum()):,} flooded / {len(y):,} cells")

    C.TRAFFY_EVAL_FILE.write_text(json.dumps({
        "label_rule": summary["rules"],
        "dense_built_up_pct": C.DENSE_BUILT_UP_PCT,
        "bootstrap": {"block_m": BLOCK_M, "resamples": N_BOOT, "interval_pct": 95},
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for rec in [r for r in results if r["kind"] == "year"]:
        print(f"\n{rec['date']}, {rec['scope']}: {rec['n_cells']:,} cells, {rec['flooded_share']:.1%} flooded")
        for name, m in rec["methods"].items():
            lo, hi = m["roc_auc_ci"]
            print(f"  {name:30s} ROC-AUC {m['roc_auc']:.3f} ({lo:.3f}-{hi:.3f})   PR-AUC {m['pr_auc']:.3f}")
    print(f"\nWrote {C.TRAFFY_EVAL_FILE}")


if __name__ == "__main__":
    main()
