"""Train the ML flood-susceptibility model on the 2011 flood and compare it with the existing methods.

    python scripts/train_model.py

Evaluation uses spatial block cross-validation (~4 km blocks): whole neighbourhoods are held out,
so the score reflects how well the model generalises to places it hasn't seen, not how well it
memorises neighbouring cells.

Outputs:
    data/processed/ml_susceptibility.tif   flood probability 0-1 for every cell
    data/processed/ml_report.json          CV metrics vs baselines, feature importance, per-district stats
    models/flood_lgbm.txt                  the trained LightGBM model
"""
import argparse
import json
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
from src.risk import cell_size_m, compute_factors, risk_index  # noqa: E402
from src.simulate import flood_mask  # noqa: E402

BLOCK_M = 4000
# Left out by default: with them the model learns "dense city = dry", which mostly reflects 2011 flood
# defences and MODIS missing water between buildings, not lower physical susceptibility. Dropping them
# cost only ~0.03 ROC-AUC (0.893 -> 0.867). Pass `--exclude` with no names to use every feature.
DEFAULT_EXCLUDE = ["log_pop_density", "built_up"]
N_FOLDS = 5
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


def bathtub_onset(area, levels=np.round(np.arange(0, 3.01, 0.1), 1)) -> np.ndarray:
    """Lowest simulated water level at which each cell floods (river-connected mode)."""
    onset = np.full(area.dem.shape, levels[-1] + 1.0, dtype="float32")
    for lv in levels[::-1]:
        onset[flood_mask(area, float(lv), connected=True)] = lv
    return onset


def scores(y, s) -> dict:
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def main(labels_path: Path, out_dir: Path, model_path: Path, exclude=()):
    area = load_study_area()
    X = build_features(area).drop(columns=list(exclude))
    with rasterio.open(labels_path) as src:
        labels = src.read(1).ravel()
    known = labels != C.LABEL_NODATA
    y = labels[known].astype(int)
    Xk = X[known].reset_index(drop=True)
    groups = spatial_blocks()[known]
    pos_rate = y.mean()
    print(f"Training cells: {len(y):,}  flooded share: {pos_rate:.1%}  features: {X.shape[1]}")

    baselines = {
        "Risk index (default weights)": risk_index(compute_factors(area), 0.5, 0.3, 0.2).ravel()[known],
        "Bathtub simulation": -bathtub_onset(area).ravel()[known],  # floods earlier = higher score
    }

    rows = []
    oof = np.zeros(len(y), dtype="float32")
    for fold, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(Xk, y, groups), start=1):
        model = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=(1 - y[tr].mean()) / y[tr].mean())
        model.fit(Xk.iloc[tr], y[tr])
        oof[te] = model.predict_proba(Xk.iloc[te])[:, 1]
        rows.append({"fold": fold, "method": "ML model (LightGBM)", **scores(y[te], oof[te])})
        for name, s in baselines.items():
            rows.append({"fold": fold, "method": name, **scores(y[te], s[te])})
        print(f"  fold {fold}: ML ROC-AUC {rows[-3]['roc_auc']:.3f}")

    cv = pd.DataFrame(rows)
    summary = cv.groupby("method")[["roc_auc", "pr_auc"]].agg(["mean", "std"])
    print("\nSpatial cross-validation (mean +/- std over folds):")
    for method, r in summary.iterrows():
        print(f"  {method:32s} ROC-AUC {r[('roc_auc', 'mean')]:.3f} +/- {r[('roc_auc', 'std')]:.3f}   "
              f"PR-AUC {r[('pr_auc', 'mean')]:.3f} +/- {r[('pr_auc', 'std')]:.3f}")
    print(f"  (PR-AUC of a random guess = flooded share = {pos_rate:.3f})")

    # Final model on all labelled cells, applied everywhere
    model = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=(1 - pos_rate) / pos_rate)
    model.fit(Xk, y)
    prob = model.predict_proba(X)[:, 1].reshape(C.HEIGHT, C.WIDTH).astype("float32")

    import shap
    rng = np.random.default_rng(SEED)
    sample = Xk.iloc[rng.choice(len(Xk), size=min(SHAP_SAMPLE, len(Xk)), replace=False)]
    shap_values = shap.TreeExplainer(model.booster_).shap_values(sample)
    if isinstance(shap_values, list):  # older shap returns one array per class
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    # Direction: does a higher feature value push the prediction up (+) or down (-)?
    direction = [float(np.corrcoef(sample[c], shap_values[:, i])[0, 1]) if sample[c].std() > 0 else 0.0
                 for i, c in enumerate(X.columns)]
    gain = model.booster_.feature_importance("gain")

    # Per-district summary
    ids = area.district_ids.ravel()
    n = int(ids.max()) + 1
    cnt = np.bincount(ids, minlength=n)
    mean_prob = np.bincount(ids, weights=prob.ravel(), minlength=n) / np.maximum(cnt, 1)
    known_all = labels != C.LABEL_NODATA
    obs_flooded = np.bincount(ids, weights=(labels == 1), minlength=n)
    obs_known = np.bincount(ids, weights=known_all, minlength=n)
    districts = [
        {"district_id": int(i), "ml_score": float(mean_prob[i]),
         "observed_2011_pct": float(100 * obs_flooded[i] / obs_known[i]) if obs_known[i] > 0 else None,
         "observed_known_pct": float(100 * obs_known[i] / cnt[i])}
        for i in range(1, n)
    ]

    report = {
        "event": {"id": "DFO_3850", "source": "Global Flood Database (MODIS, 250 m)",
                  "began": "2011-08-05", "ended": "2012-01-09"},
        "n_training_cells": int(len(y)), "flooded_share": float(pos_rate),
        "excluded_features": [FEATURE_LABELS.get(c, c) for c in exclude],
        "cv": {"block_m": BLOCK_M, "folds": N_FOLDS,
               "summary": [{"method": m, "roc_auc": r[("roc_auc", "mean")], "roc_auc_std": r[("roc_auc", "std")],
                            "pr_auc": r[("pr_auc", "mean")], "pr_auc_std": r[("pr_auc", "std")]}
                           for m, r in summary.iterrows()],
               "folds_detail": rows},
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
    (out_dir / C.ML_REPORT_FILE.name).write_text(json.dumps(report, indent=2))
    model.booster_.save_model(str(model_path))

    print("\nTop features (mean |SHAP|):")
    for f in sorted(report["features"], key=lambda f: -f["mean_abs_shap"])[:6]:
        print(f"  {f['label']:36s} {f['mean_abs_shap']:.3f}  ({'+' if f['direction'] > 0 else '-'} flood)")
    print(f"\nWrote {out_dir / C.ML_PROB_FILE.name}, {out_dir / C.ML_REPORT_FILE.name}, {model_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=C.FLOOD_2011_FILE)
    ap.add_argument("--out-dir", type=Path, default=C.PROCESSED_DIR)
    ap.add_argument("--model-path", type=Path, default=C.ML_MODEL_FILE)
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help="feature names to leave out (default: %(default)s; give none to use all)")
    args = ap.parse_args()
    if not args.labels.exists():
        raise SystemExit(f"{args.labels} not found. Run scripts/prepare_ml_data.py first.")
    main(args.labels, args.out_dir, args.model_path, args.exclude)
