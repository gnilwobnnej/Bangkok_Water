"""Train the text model that recognises flood reports, for when Traffy doesn't say which reports are floods.

    python scripts/train_traffy_text.py

Since October 2026 Traffy Fondue's public sources no longer carry a report's category (the old GeoJSON API is
gone, and the share API and CSV export leave it empty for 2026), so scripts/fetch_traffy.py --daily recognises
flood reports from their text instead. The model: TF-IDF on character 2-4-grams (Thai has no spaces between
words) and logistic regression, trained on reports that do have Traffy's own category: every report on the
sampled days cached by scripts/fetch_traffy.py --all.

Test: a model trained on the years before config.TEST_YEARS_FROM is scored on the test years' categorised reports,
and the precision and recall at each cut-off are printed and saved. The saved model is then trained on every
categorised report, the test years' too (it learns words, not places, so the street-flood test isn't affected).

It is saved as plain arrays (vocabulary, idf, weights), not a pickle, so it loads with any scikit-learn version.

Outputs:
    models/traffy_flood_text.npz    the model (src/traffy.TextModel)
    models/traffy_flood_text.json   cut-off, test scores, training data
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.traffy import TextModel  # noqa: E402

CUTOFF = 0.9  # a report counts as a flood report at this probability or above (best balance on the test year)
MAX_FEATURES = 60000
VECTORISER = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=5, sublinear_tf=True)


def load_categorised(folder: Path) -> tuple:
    """(days, texts, is_flood) of every cached report, from the sampled days fetched with --all."""
    days, texts, labels = [], [], []
    for f in sorted(folder.glob("*.jsonl.gz")):
        day = f.name[:10]
        with gzip.open(f, "rt", encoding="utf-8") as h:
            for line in h:
                r = json.loads(line)
                days.append(day)
                texts.append(r.get("description") or "")
                labels.append(int(bool(r.get("is_flood"))))
    return np.array(days), texts, np.array(labels)


def fit(texts: list, y: np.ndarray, max_features: int = MAX_FEATURES) -> TextModel:
    vec = TfidfVectorizer(max_features=max_features, **VECTORISER)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=4, class_weight="balanced").fit(X, y)
    terms = np.empty(len(vec.vocabulary_), dtype=object)
    for t, i in vec.vocabulary_.items():
        terms[i] = t
    return TextModel(terms=terms.astype(str), idf=vec.idf_.astype("float32"),
                     coef=clf.coef_[0].astype("float32"), intercept=float(clf.intercept_[0]))


def test_scores(y: np.ndarray, prob: np.ndarray) -> list:
    return [{"cutoff": c, "precision": float(precision_score(y, prob >= c)),
             "recall": float(recall_score(y, prob >= c))} for c in (0.5, 0.6, 0.7, 0.8, 0.9)]


def main():
    days, texts, y = load_categorised(C.RAW_DIR / "traffy" / "all")
    if not len(y):
        sys.exit("No cached reports: run scripts/fetch_traffy.py --all first")
    test_from = f"{C.TEST_YEARS_FROM}-01-01"
    train, test = days < test_from, days >= test_from
    print(f"{train.sum():,} training reports ({y[train].mean():.1%} floods), "
          f"{test.sum():,} test reports ({y[test].mean():.1%} floods)")
    model = fit([texts[i] for i in np.flatnonzero(train)], y[train])
    prob = model.predict_proba([texts[i] for i in np.flatnonzero(test)])
    scores = test_scores(y[test], prob)
    wet = days[test] >= f"{C.TEST_YEARS_FROM}-08-01"  # the wet season, when most flood reports come in
    scores_wet = test_scores(y[test][wet], prob[wet])
    for s, w in zip(scores, scores_wet):
        print(f"  cut-off {s['cutoff']:.1f}: {s['precision']:.0%} of flagged reports are floods, "
              f"{s['recall']:.0%} of floods found (from August: {w['precision']:.0%}, {w['recall']:.0%})")

    final = fit(texts, y)
    final.save(C.TRAFFY_TEXT_MODEL_FILE)
    C.TRAFFY_TEXT_MODEL_FILE.with_suffix(".json").write_text(json.dumps({
        "cutoff": CUTOFF,
        "test": {"trained_before": test_from, "tested_on": [min(days[test]), max(days[test])],
                 "n_reports": int(test.sum()), "flood_share": float(y[test].mean()), "scores": scores,
                 "from_august": {"n_reports": int(wet.sum()), "flood_share": float(y[test][wet].mean()),
                                 "scores": scores_wet}},
        "final_model": {"days": [min(days), max(days)], "n_reports": int(len(y)),
                        "features": len(final.terms), **VECTORISER, "ngram_range": list(VECTORISER["ngram_range"])},
    }, indent=2))
    print(f"Wrote {C.TRAFFY_TEXT_MODEL_FILE} ({C.TRAFFY_TEXT_MODEL_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
