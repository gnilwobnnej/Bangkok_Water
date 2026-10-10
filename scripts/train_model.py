"""Train the ML flood-susceptibility model on several floods and compare it with the existing methods.

    python scripts/train_model.py                       # 2011 + every radar season in data/processed/flood_labels
    python scripts/train_model.py --labels data/processed/flood_2011.tif   # one flood only

The floods are the 2011 MODIS map (scripts/prepare_ml_data.py) and one radar map per wet season 2017-2025
(scripts/build_flood_archive.py). Floods from config.TEST_YEARS_FROM on (2026) are never trained on, so
scripts/evaluate_current.py stays an honest test. The events are pooled and each counts equally: the same
number of cells is sampled from each, with weights making up for any event that has fewer.

Two models are trained side by side, so the gain from the extra floods can be measured:
    ML model (all floods)   every event
    ML model (2011 only)    the 2011 flood only, as before

Evaluation:
    Spatial block CV (~4 km blocks, 5 folds). A block is held out of every event at once, so a model is never
    trained on a place it is tested on, in any year. Scored per event, against the bathtub and risk index:
    per fold, and pooled (every cell scored by the fold model that never saw it) with 95% block bootstrap
    intervals. How much the 5 fold models disagree on each cell is saved as a map.
    Leave-one-event-out: train on every flood but one, test on that one (all its cells), with 95% block
    bootstrap intervals. This asks "how well does it predict a flood year it has never seen?".

Outputs:
    data/processed/ml_susceptibility.tif        flood probability 0-1, all-floods model
    data/processed/ml_susceptibility_2011.tif   the same from the 2011-only model
    data/processed/ml_disagreement.tif          standard deviation of the 5 CV fold models' probabilities
    data/processed/ml_report.json               events, CV and leave-one-event-out scores, feature
                                                importance, per-district stats
    models/flood_lgbm.txt, models/flood_lgbm_2011.txt
"""
import argparse
import json
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.data_loader import load_study_area  # noqa: E402
from src.features import FEATURE_LABELS, build_features  # noqa: E402
from src.ml import ML_2011, ML_ALL  # noqa: E402
from src.risk import cell_size_m, compute_factors, risk_index  # noqa: E402
from src.simulate import flood_mask  # noqa: E402
from src.validation import block_bootstrap  # noqa: E402

BLOCK_M = 4000
# Left out by default: with them the model learns "dense city = dry", which mostly reflects 2011 flood
# defences and MODIS missing water between buildings, not lower physical susceptibility. Dropping them
# cost only ~0.03 ROC-AUC (0.893 -> 0.867). Pass `--exclude` with no names to use every feature.
DEFAULT_EXCLUDE = ["log_pop_density", "built_up"]
N_FOLDS = 5
CELLS_PER_EVENT = 200_000  # training cells sampled from each event, so every flood counts the same
SHAP_SAMPLE = 20000
SEED = 42
PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1,
)


def spatial_blocks() -> np.ndarray:
    dy, dx = cell_size_m()
    rows, cols = np.indices((C.HEIGHT, C.WIDTH))
    br, bc = rows // max(1, round(BLOCK_M / dy)), cols // max(1, round(BLOCK_M / dx))
    return (br * (bc.max() + 1) + bc).ravel()


def bathtub_onset(area, levels=np.round(np.arange(0, 3.01, 0.1), 1), defences: bool = False) -> np.ndarray:
    """Lowest simulated water level at which each cell floods (river-connected mode)."""
    onset = np.full(area.dem.shape, levels[-1] + 1.0, dtype="float32")
    for lv in levels[::-1]:
        onset[flood_mask(area, float(lv), connected=True, defences=defences)] = lv
    return onset


def baseline_methods(area) -> dict:
    """Score per cell for each method that isn't trained (higher = more likely to flood)."""
    methods = {
        "Risk index (default weights)": risk_index(compute_factors(area), 0.5, 0.3, 0.2),
        "Bathtub simulation": -bathtub_onset(area),  # floods earlier = higher score
    }
    if area.defences is not None:
        methods["Bathtub + defences"] = -bathtub_onset(area, defences=True)
    return methods


def scores(y, s) -> dict:
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def event_info(path: Path) -> dict:
    """id, year, display name and source of a label file (flood_2011.tif or flood_labels/radar_YYYY.tif)."""
    m = re.fullmatch(r"radar_(\d{4})", path.stem)
    if m:
        year = int(m.group(1))
        return {"id": f"radar_{year}", "year": year, "name": f"{year} wet season (radar)",
                "source": "Sentinel-1 radar, Aug-Nov, seasonal baseline (scripts/build_flood_archive.py)"}
    if path.resolve() == C.FLOOD_2011_FILE.resolve():
        return {"id": "2011", "year": 2011, "name": "2011 flood (MODIS)",
                "source": "Global Flood Database DFO_3850 (MODIS, 250 m), 2011-08-05 to 2012-01-09"}
    return {"id": path.stem, "year": None, "name": path.stem, "source": str(path)}


def default_label_files() -> list:
    radar = sorted(C.FLOOD_LABELS_DIR.glob("radar_*.tif"))
    return [C.FLOOD_2011_FILE] + [p for p in radar if event_info(p)["year"] < C.TEST_YEARS_FROM]


def load_events(paths) -> list:
    """[{**event_info, labels (flat uint8)}], refusing test years."""
    events = []
    for p in paths:
        info = event_info(Path(p))
        if info["year"] is not None and info["year"] >= C.TEST_YEARS_FROM:
            raise SystemExit(f"{p}: {info['year']} is a test year (config.TEST_YEARS_FROM = {C.TEST_YEARS_FROM})")
        with rasterio.open(p) as src:
            info["labels"] = src.read(1).ravel()
        known = info["labels"] != C.LABEL_NODATA
        info["n_cells"] = int(known.sum())
        info["flooded_share"] = float((info["labels"][known] == 1).mean())
        events.append(info)
    return events


def training_set(events, cell_ok=None, cap=CELLS_PER_EVENT, seed=SEED):
    """Pooled (cell index, label, weight) arrays over the events, `cap` cells sampled from each.

    cell_ok: optional boolean mask of usable cells (the CV training blocks). Weights make each event's
    total weight equal (relevant only when an event has fewer than `cap` usable cells).
    """
    rng = np.random.default_rng(seed)
    idx, ys, ws = [], [], []
    for e in events:
        ok = e["labels"] != C.LABEL_NODATA
        if cell_ok is not None:
            ok &= cell_ok
        cells = np.flatnonzero(ok)
        if len(cells) > cap:
            cells = rng.choice(cells, size=cap, replace=False)
        idx.append(cells)
        ys.append(e["labels"][cells].astype(int))
        ws.append(np.full(len(cells), cap / max(len(cells), 1)))
    w = np.concatenate(ws)
    return np.concatenate(idx), np.concatenate(ys), w / w.mean()


def fit_predict(X: pd.DataFrame, cells, y, w) -> tuple:
    """Fit on the given cells, predict every cell of the grid. Returns (model, probability per cell)."""
    pos = np.average(y, weights=w)
    model = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=(1 - pos) / pos)
    model.fit(X.iloc[cells], y, sample_weight=w)
    return model, model.predict_proba(X)[:, 1].astype("float32")


def cv_summary(rows: list, keys=("method",)) -> list:
    """Mean and standard deviation over the folds, per method (and per event, if keyed)."""
    df = pd.DataFrame(rows)
    summary = df.groupby(list(keys), sort=False)[["roc_auc", "pr_auc"]].agg(["mean", "std"])
    out = []
    for key, r in summary.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(keys, key)), "roc_auc": r[("roc_auc", "mean")], "roc_auc_std": r[("roc_auc", "std")],
                    "pr_auc": r[("pr_auc", "mean")], "pr_auc_std": r[("pr_auc", "std")]})
    return out


def print_table(title: str, rows: list):
    print(f"\n{title}")
    for r in rows:
        ev = f"{r['event']:26s} " if "event" in r else ""
        sd = f" +/- {r['roc_auc_std']:.3f}" if "roc_auc_std" in r else ""
        print(f"  {ev}{r['method']:30s} ROC-AUC {r['roc_auc']:.3f}{sd}   PR-AUC {r['pr_auc']:.3f}")


def rescore_baselines(labels_path: Path, report_path: Path):
    """Re-score only the untrained methods on the 2011 folds; the trained ML models and their scores are kept."""
    area = load_study_area()
    with rasterio.open(labels_path) as src:
        labels = src.read(1).ravel()
    known = labels != C.LABEL_NODATA
    y = labels[known].astype(int)
    fold_of = block_folds(spatial_blocks())[known]
    baselines = {n: s.ravel()[known] for n, s in baseline_methods(area).items()}
    report = json.loads(report_path.read_text())
    rows = [r for r in report["cv"]["folds_detail"] if r["method"].startswith("ML model")]
    for fold in range(1, N_FOLDS + 1):
        te = fold_of == fold
        rows += [{"fold": fold, "method": n, **scores(y[te], s[te])} for n, s in baselines.items()]
    rows.sort(key=lambda r: r["fold"])
    report["cv"]["folds_detail"] = rows
    report["cv"]["summary"] = cv_summary(rows)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nUpdated {report_path} (2011 scores only)")


def load_context() -> tuple:
    """Per-year rainfall and river context (scripts/event_context.py): ({year: dict}, sources), or ({}, None)."""
    path = C.FLOOD_LABELS_DIR / "context.json"
    if not path.exists():
        return {}, None
    d = json.loads(path.read_text())
    return ({r["year"]: {k: v for k, v in r.items() if k != "year"} for r in d["years"]},
            {k: v for k, v in d.items() if k != "years"})


def block_folds(blocks: np.ndarray) -> np.ndarray:
    """Fold number (1..N_FOLDS) per cell, from GroupKFold over the spatial blocks of the whole grid."""
    fold_of = np.zeros(len(blocks), dtype="int8")
    for fold, (_, te) in enumerate(GroupKFold(N_FOLDS).split(np.zeros(len(blocks)), groups=blocks), start=1):
        fold_of[te] = fold
    return fold_of


def main(label_paths, out_dir: Path, model_path: Path, exclude=()):
    area = load_study_area()
    X = build_features(area).drop(columns=list(exclude))
    events = load_events(label_paths)
    blocks = spatial_blocks()
    fold_of = block_folds(blocks)
    ev2011 = [e for e in events if e["id"] == "2011"]
    compare_2011 = bool(ev2011) and len(events) > 1  # also train the 2011-only "before" model
    print(f"{len(events)} flood events, features: {X.shape[1]}")
    for e in events:
        print(f"  {e['name']:26s} {e['n_cells']:>9,} labelled cells, {e['flooded_share']:5.1%} flooded")

    baselines = {n: s.ravel() for n, s in baseline_methods(area).items()}

    def score_all(preds: dict, ev, cells_ok) -> list:
        """Scores of each prediction on one event's labelled cells within cells_ok."""
        known = (ev["labels"] != C.LABEL_NODATA) & cells_ok
        y = ev["labels"][known].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return []
        return [{"event": ev["name"], "method": n, **scores(y, p[known])} for n, p in preds.items()]

    # ---- Spatial block CV: each fold's blocks are held out of every event ----
    rows = []
    fold_probs = []  # each fold model's prediction everywhere, for the disagreement map
    oof = {m: np.zeros(len(blocks), dtype="float32") for m in ([ML_ALL, ML_2011] if compare_2011 else [ML_ALL])}
    for fold in range(1, N_FOLDS + 1):
        train_ok, test_ok = fold_of != fold, fold_of == fold
        preds = {ML_ALL: fit_predict(X, *training_set(events, train_ok))[1]}
        if compare_2011:
            preds[ML_2011] = fit_predict(X, *training_set(ev2011, train_ok, cap=10 ** 9))[1]
        fold_probs.append(preds[ML_ALL])
        for m in oof:
            oof[m][test_ok] = preds[m][test_ok]  # out-of-fold: each cell predicted by the model that never saw it
        preds.update(baselines)
        for ev in events:
            rows += [{"fold": fold, **r} for r in score_all(preds, ev, test_ok)]
        print(f"  CV fold {fold} done")
    disagreement = np.std(fold_probs, axis=0).astype("float32").reshape(C.HEIGHT, C.WIDTH)
    del fold_probs

    # Every cell scored once by a model that never saw its block, per event, with 95% block bootstrap intervals
    pooled = []
    for ev in events:
        known = ev["labels"] != C.LABEL_NODATA
        y = ev["labels"][known].astype(int)
        preds = {**oof, **baselines}
        ci = block_bootstrap(y, {n: p[known] for n, p in preds.items()}, blocks[known])
        for r in score_all(preds, ev, np.ones(len(blocks), dtype=bool)):
            pooled.append({**r, **ci[r["method"]], "flooded_share": ev["flooded_share"], "n_cells": ev["n_cells"]})
    print_table("Spatial CV, out-of-fold scores per event:", pooled)
    per_event = cv_summary(rows, keys=("event", "method"))
    print_table("Spatial cross-validation, per event (mean +/- std over folds):", per_event)
    mean_over_events = (pd.DataFrame(per_event).groupby("method", sort=False)[["roc_auc", "pr_auc"]].mean()
                        .reset_index().to_dict("records"))
    print_table("...averaged over the events:", mean_over_events)

    # The 2011-only model on all 2011 cells: the "before" model, also tested on every radar season below
    if compare_2011:
        model_2011, prob_2011 = fit_predict(X, *training_set(ev2011, cap=10 ** 9))

    # ---- Leave one event out: train on the other floods, test on all of this one ----
    loeo = []
    if len(events) > 1:
        everywhere = np.ones(len(blocks), dtype=bool)
        for ev in events:
            others = [e for e in events if e is not ev]
            preds = {ML_ALL: fit_predict(X, *training_set(others))[1]}
            if compare_2011 and ev["id"] != "2011":
                preds[ML_2011] = prob_2011
            preds.update(baselines)
            known = ev["labels"] != C.LABEL_NODATA
            y = ev["labels"][known].astype(int)
            ci = block_bootstrap(y, {n: p[known] for n, p in preds.items()}, blocks[known])
            for r in score_all(preds, ev, everywhere):
                loeo.append({**r, **ci[r["method"]], "flooded_share": ev["flooded_share"],
                             "n_cells": ev["n_cells"]})
            print(f"  left out {ev['name']}: done")
        print_table("Leave one event out (trained on the other floods):",
                    [r for r in loeo if r["method"].startswith("ML") or r["method"] == "Bathtub + defences"])

    # ---- Final models on all labelled cells, applied everywhere ----
    model, prob = fit_predict(X, *training_set(events))
    prob = prob.reshape(C.HEIGHT, C.WIDTH)

    import shap
    rng = np.random.default_rng(SEED)
    cells, _, _ = training_set(events)
    sample = X.iloc[rng.choice(cells, size=min(SHAP_SAMPLE, len(cells)), replace=False)]
    shap_values = shap.TreeExplainer(model.booster_).shap_values(sample)
    if isinstance(shap_values, list):  # older shap returns one array per class
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    # Direction: does a higher feature value push the prediction up (+) or down (-)?
    direction = [float(np.corrcoef(sample[c], shap_values[:, i])[0, 1]) if sample[c].std() > 0 else 0.0
                 for i, c in enumerate(X.columns)]
    gain = model.booster_.feature_importance("gain")

    # Per-district summary: model score, share flooded in 2011, and in how many radar seasons
    ids = area.district_ids.ravel()
    n = int(ids.max()) + 1
    cnt = np.bincount(ids, minlength=n)
    mean_prob = np.bincount(ids, weights=prob.ravel(), minlength=n) / np.maximum(cnt, 1)
    labels_2011 = ev2011[0]["labels"] if ev2011 else np.full(len(ids), C.LABEL_NODATA)
    known_all = labels_2011 != C.LABEL_NODATA
    obs_flooded = np.bincount(ids, weights=(labels_2011 == 1), minlength=n)
    obs_known = np.bincount(ids, weights=known_all, minlength=n)
    districts = [
        {"district_id": int(i), "ml_score": float(mean_prob[i]),
         "observed_2011_pct": float(100 * obs_flooded[i] / obs_known[i]) if obs_known[i] > 0 else None,
         "observed_known_pct": float(100 * obs_known[i] / cnt[i])}
        for i in range(1, n)
    ]

    cv_2011 = [r for r in rows if r["event"] == ev2011[0]["name"]] if ev2011 else rows
    context, context_meta = load_context()
    report = {
        # 2011 kept as "event" for the app's 2011 comparisons; every event is in "events"
        "event": {"id": "DFO_3850", "source": "Global Flood Database (MODIS, 250 m)",
                  "began": "2011-08-05", "ended": "2012-01-09"},
        "events": [{**{k: e[k] for k in ("id", "year", "name", "source", "n_cells", "flooded_share")},
                    **({"context": context[e["year"]]} if e["year"] in context else {})} for e in events],
        "event_context": context_meta,
        "test_years_from": C.TEST_YEARS_FROM,
        "cells_per_event": CELLS_PER_EVENT,
        "n_training_cells": int(sum(min(e["n_cells"], CELLS_PER_EVENT) for e in events)),
        "flooded_share": ev2011[0]["flooded_share"] if ev2011 else events[0]["flooded_share"],
        "excluded_features": [FEATURE_LABELS.get(c, c) for c in exclude],
        "cv": {"block_m": BLOCK_M, "folds": N_FOLDS,
               "summary": cv_summary(cv_2011),          # 2011 flood only, as before
               "folds_detail": [{k: v for k, v in r.items() if k != "event"} for r in cv_2011],
               "per_event": per_event,
               "mean_over_events": mean_over_events,
               "per_event_folds": rows,
               "pooled": pooled},
        "leave_one_event_out": loeo,
        "features": [
            {"feature": c, "label": FEATURE_LABELS.get(c, c), "mean_abs_shap": float(mean_abs[i]),
             "direction": direction[i], "gain": float(gain[i])}
            for i, c in enumerate(X.columns)
        ],
        "districts": districts,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=C.HEIGHT, width=C.WIDTH, count=1, crs=C.CRS,
                   transform=area.transform, compress="deflate", dtype="float32")
    with rasterio.open(out_dir / C.ML_PROB_FILE.name, "w", **profile) as dst:
        dst.write(prob, 1)
    with rasterio.open(out_dir / C.ML_DISAGREEMENT_FILE.name, "w", **profile) as dst:
        dst.write(disagreement, 1)
    if compare_2011:
        with rasterio.open(out_dir / C.ML_PROB_2011_FILE.name, "w", **profile) as dst:
            dst.write(prob_2011.reshape(C.HEIGHT, C.WIDTH), 1)
        model_2011.booster_.save_model(str(model_path.with_name(model_path.stem + "_2011" + model_path.suffix)))
    (out_dir / C.ML_REPORT_FILE.name).write_text(json.dumps(report, indent=2))
    model.booster_.save_model(str(model_path))

    print("\nTop features (mean |SHAP|):")
    for f in sorted(report["features"], key=lambda f: -f["mean_abs_shap"])[:6]:
        print(f"  {f['label']:36s} {f['mean_abs_shap']:.3f}  ({'+' if f['direction'] > 0 else '-'} flood)")
    print(f"\nWrote {out_dir / C.ML_PROB_FILE.name}, {out_dir / C.ML_REPORT_FILE.name}, {model_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, nargs="+", default=None,
                    help="label files (default: flood_2011.tif + flood_labels/radar_YYYY.tif before the test years)")
    ap.add_argument("--out-dir", type=Path, default=C.PROCESSED_DIR)
    ap.add_argument("--model-path", type=Path, default=C.ML_MODEL_FILE)
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help="feature names to leave out (default: %(default)s; give none to use all)")
    ap.add_argument("--baselines-only", action="store_true",
                    help="re-score only the untrained methods (bathtub, risk index) on the 2011 folds of the "
                         "existing report, e.g. after preparing flood defences; the ML models are left as they are")
    args = ap.parse_args()
    labels = args.labels or default_label_files()
    missing = [p for p in labels if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{missing} not found. Run scripts/prepare_ml_data.py (2011) and "
                         "scripts/build_flood_archive.py (radar seasons) first.")
    if args.baselines_only:
        rescore_baselines(C.FLOOD_2011_FILE, args.out_dir / C.ML_REPORT_FILE.name)
    else:
        main(labels, args.out_dir, args.model_path, args.exclude)
