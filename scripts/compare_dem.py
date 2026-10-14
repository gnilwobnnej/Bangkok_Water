"""Does FABDEM (buildings and trees removed) make the bathtub model better than Copernicus (a surface model)?

    python scripts/compare_dem.py [--gee-project ID]

Builds both elevation grids the way prepare_data.py does (FABDEM is downloaded from Earth Engine the first time),
runs the bathtub model on each, with and without flood defences, and scores them on every flood we have labels
for. The bathtub model isn't trained, so every event is a fair test:
    2011 flood (MODIS), wet seasons 2017-2025 (radar), every 2026 radar pass, and the Traffy Fondue street-flood
    labels for each year (Bangkok's 50 districts and the dense city, where satellites can't see water).
Each FABDEM score comes with a 95% block bootstrap interval for its difference from the same model on Copernicus,
so a change is only called real when that interval excludes zero.

Output: data/processed/dem_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_current import score  # noqa: E402
from prepare_data import build_dem  # noqa: E402
from src import config as C  # noqa: E402
from src.current_flood import load_pass, radar_labels  # noqa: E402
from src.data_loader import load_study_area  # noqa: E402
from src.validation import block_bootstrap  # noqa: E402
from train_model import BLOCK_M, bathtub_onset, event_info, spatial_blocks  # noqa: E402

OUT_FILE = C.PROCESSED_DIR / "dem_comparison.json"
MIN_CELLS = 20
PAIRS = [("Bathtub", False), ("Bathtub + defences", True)]


def read_labels(path: Path) -> np.ndarray:
    """1 flooded, 0 dry, -1 unknown from a 1/0/255 label file."""
    with rasterio.open(path) as src:
        lab = src.read(1).astype("int16")
    lab[lab == C.LABEL_NODATA] = -1
    return lab


def events(bkk: np.ndarray, dense: np.ndarray) -> list:
    """(event id, name, source kind, scope, labels with -1 outside the scope)."""
    out = []
    for path in [C.FLOOD_2011_FILE] + sorted(C.FLOOD_LABELS_DIR.glob("radar_*.tif")):
        info = event_info(path)
        out.append((info["id"], info["name"], "river", "Greater Bangkok", read_labels(path)))
    summary = json.loads(C.RADAR_SUMMARY_FILE.read_text())
    for p in summary["passes"]:
        out.append((f"pass_{p['date']}", f"{p['date']} radar pass", "river", "Greater Bangkok",
                    radar_labels(load_pass(p["date"])).astype("int16")))
    for path in sorted(C.FLOOD_LABELS_DIR.glob("traffy_*.tif")):
        info = event_info(path)
        lab = read_labels(path)
        for scope, mask in [("Bangkok (50 districts)", bkk), ("Dense city", dense)]:
            out.append((info["id"], info["name"], "street", scope, np.where(mask, lab, -1)))
    return out


def main(project: str | None):
    area = load_study_area()
    bkk = area.district_ids > 0
    with rasterio.open(C.LANDCOVER_FILE) as src:
        built_up = src.read(1 + list(src.descriptions).index("built_up"))
    dense = bkk & (built_up >= C.DENSE_BUILT_UP_PCT)
    blocks = spatial_blocks().reshape(bkk.shape)

    dems = {"Copernicus": build_dem("copernicus"), "FABDEM": build_dem("fabdem", project)}
    diff = dems["FABDEM"] - dems["Copernicus"]
    print("FABDEM minus Copernicus (m), median / p5 / p95:")
    for name, mask in [("everywhere", np.ones_like(bkk)), ("Bangkok", bkk), ("dense city", dense)]:
        d = diff[mask]
        print(f"  {name:12s} {np.median(d):+.2f} / {np.percentile(d, 5):+.2f} / {np.percentile(d, 95):+.2f}")

    methods = {}
    for dem_name, dem in dems.items():
        a = replace(area, dem=dem.astype("float32"))
        for label, defended in PAIRS:
            if defended and area.defences is None:
                continue
            methods[f"{label} ({dem_name})"] = -bathtub_onset(a, defences=defended)
            print(f"  ran {label} on {dem_name}")

    results = []
    for eid, name, kind, scope, labels in events(bkk, dense):
        known = labels >= 0
        y = labels[known].astype(int)
        if y.sum() < MIN_CELLS or (y == 0).sum() < MIN_CELLS:
            continue
        rec = {"event": eid, "name": name, "kind": kind, "scope": scope, "n_cells": int(known.sum()),
               "flooded_share": float(y.mean()), "methods": {}}
        for label, _ in PAIRS:
            pair = {d: methods.get(f"{label} ({d})") for d in dems}
            if any(s is None for s in pair.values()):
                continue
            ref = f"{label} (Copernicus)"
            ci = block_bootstrap(y, {f"{label} ({d})": s[known] for d, s in pair.items()}, blocks[known],
                                 reference=ref)
            for d, s in pair.items():
                m = f"{label} ({d})"
                rec["methods"][m] = {**score(y, s[known]), **ci[m]}
        results.append(rec)
        line = "  ".join(f"{m.replace('Bathtub', 'BT')}: {v['roc_auc']:.3f}" for m, v in rec["methods"].items())
        print(f"  {name:30s} {scope:22s} {line}")

    OUT_FILE.write_text(json.dumps({
        "dems": {"Copernicus": "Copernicus GLO-30 (surface model)",
                 "FABDEM": f"FABDEM V1-2, Earth Engine {C.RAW_DIR.name}/fabdem_bangkok.tif (Hawker et al. 2022)"},
        "elevation_difference_m": {
            name: {"median": float(np.median(diff[mask])), "p5": float(np.percentile(diff[mask], 5)),
                   "p95": float(np.percentile(diff[mask], 95))}
            for name, mask in [("Greater Bangkok", np.ones_like(bkk)), ("Bangkok (50 districts)", bkk),
                               ("Dense city", dense)]},
        "bootstrap": {"block_m": BLOCK_M, "interval_pct": 95, "difference": "FABDEM minus Copernicus"},
        "results": results,
    }, indent=2), encoding="utf-8")

    print("\nFABDEM vs Copernicus, ROC-AUC change (95% interval); * = interval excludes zero")
    for label, _ in PAIRS:
        m = f"{label} (FABDEM)"
        rows = [r for r in results if m in r["methods"]]
        if not rows:
            continue
        better = sum(r["methods"][m]["roc_auc_diff_ci"][0] > 0 for r in rows)
        worse = sum(r["methods"][m]["roc_auc_diff_ci"][1] < 0 for r in rows)
        print(f"\n{label}: FABDEM clearly better on {better}, clearly worse on {worse}, of {len(rows)} tests")
        for r in rows:
            f, c = r["methods"][m], r["methods"][f"{label} (Copernicus)"]
            lo, hi = f["roc_auc_diff_ci"]
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {r['name']:30s} {r['scope']:22s} {c['roc_auc']:.3f} -> {f['roc_auc']:.3f} "
                  f"({f['roc_auc'] - c['roc_auc']:+.3f}, {lo:+.3f} to {hi:+.3f}){star}")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gee-project", help="Google Cloud project for Earth Engine (first FABDEM download only)")
    main(ap.parse_args().gee_project)
