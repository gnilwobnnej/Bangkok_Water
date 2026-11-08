---
name: retrain
description: Retrain the river-flood and/or street-flood ML models, re-score them on the test years, and compare the new metrics with the committed ones before anything is committed. Use when the user wants to retrain, or after changing features, labels or training code.
disable-model-invocation: true
argument-hint: "[river|street|both] (default both)"
---
Retrain and compare. Which model: `$ARGUMENTS` (`river`, `street` or `both`; empty means both).

1. **Save a baseline** from git, not from disk: copy `git show HEAD:<file>` for
   `data/processed/ml_report.json`, `data/processed/ml_street_report.json`,
   `data/processed/radar_flood/evaluation.json` and `data/processed/traffy/evaluation.json` into a temp folder.
2. **Train.** `train_model.py` needs `shap` (`pip install shap` if it's missing). Each run takes a while, so run it
   in the background.
   - river: `python scripts/train_model.py`
   - street: `python scripts/train_model.py --kind street`
3. **Re-score on the test years** (2026+, never trained on):
   - river: `python scripts/evaluate_current.py`
   - street: `python scripts/evaluate_street.py`
4. **Compare** old and new with a short Python script: the pooled CV and leave-one-event-out scores in the
   `ml_*report.json` files (`cv`, `leave_one_event_out`), and every method's scores and 95% intervals in the
   two `evaluation.json` files. Show a table: metric, old, new, change, and whether the new value is outside the
   old 95% interval.
5. **Verdict:** better / same / worse. Treat a change inside the bootstrap intervals as noise. Flag any score that
   got worse on the dense city or the city zone (the street model's reason to exist).
6. Run `pytest`, because the smoke test loads the new models.

Don't commit. The models (`models/*.txt`), rasters and reports are all part of the change, so leave the decision to
the user. If the result is worse, tell them how to restore the old files:
`git restore models data/processed/ml_*`, `git restore data/processed/radar_flood/evaluation.json` and
`git restore data/processed/traffy/evaluation.json`.
