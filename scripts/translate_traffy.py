"""Translate the Traffy Fondue flood reports from Thai into English, for the map.

    python scripts/translate_traffy.py              # every report not yet translated (~2 h the first time on a CPU)
    python scripts/translate_traffy.py --limit 30   # a quick sample, printed Thai -> English

Needs `pip install transformers sentencepiece` (and torch); the app itself doesn't.

The machine translation is Helsinki-NLP/opus-mt-th-en (MarianMT, Apache-2.0), free and run locally. It gives the
gist of a report, not a polished translation: place names and slang often come out wrong. Two things help it:
- the commonest short reports ("น้ำท่วม", "ท่อตัน" ...) and "เขต: <district>" labels have exact translations;
- the flood form's own answers at the end of most reports ("water depth: knee-deep, flooded for less than 3
  hours") are translated exactly by src/traffy.split_description, not by the model;
- the free text is translated line by line, because the model tends to drop the rest of a long multi-line text.
Pieces without Thai (already English, numbers, emoji) are kept as they are.

Output: data/processed/traffy/translations.parquet, one row per distinct Thai description (description,
description_en). Texts already there are skipped, so the daily update only translates new reports.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.traffy import THAI, fixed_en, join_english, load_reports, load_translations, split_description  # noqa: E402

MODEL = "Helsinki-NLP/opus-mt-th-en"
CHUNK = 512  # descriptions per save, so a long first run can be stopped and resumed


def pending(descriptions: pd.Series, done: dict) -> list:
    """Distinct, non-blank descriptions without a translation yet, in first-seen order."""
    d = descriptions.dropna()
    d = d[d.str.strip() != ""].drop_duplicates()
    return [t for t in d if t not in done]


def translate_texts(texts: list, translate) -> dict:
    """{Thai description: English}. `translate` maps a list of Thai pieces to English (the model, or a stub)."""
    split = [split_description(t) for t in texts]
    pieces = {p for ps, _ in split for p in ps}
    en = {p: f for p in pieces if (f := fixed_en(p)) is not None}
    todo = sorted((p for p in pieces if p not in en and THAI.search(p)), key=len)  # similar lengths batch well
    if todo:
        en.update(zip(todo, translate(todo)))
    return {t: join_english([en.get(p, p) for p in pieces], form) for t, (pieces, form) in zip(texts, split)}


def model_translator(batch_size: int):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).eval()

    def translate(pieces: list) -> list:
        out = []
        with torch.inference_mode():
            for i in range(0, len(pieces), batch_size):
                batch = tok(pieces[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                            max_length=256)
                ids = model.generate(**batch, num_beams=1, max_new_tokens=160,
                                     no_repeat_ngram_size=3)  # greedy: ~2.5x faster than beams
                out += tok.batch_decode(ids, skip_special_tokens=True)
        return out
    return translate


def save(done: dict) -> None:
    C.TRAFFY_TRANSLATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.TRAFFY_TRANSLATIONS_FILE.with_suffix(".part")
    pd.DataFrame({"description": list(done), "description_en": list(done.values())}).to_parquet(tmp, index=False)
    tmp.replace(C.TRAFFY_TRANSLATIONS_FILE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="translate only this many new texts, print them and don't save")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    reports = load_reports()
    if reports is None:
        sys.exit("No reports: run scripts/fetch_traffy.py first")
    done = load_translations()
    texts = pending(reports["description"], done)
    print(f"{len(done):,} texts translated, {len(texts):,} to go")
    if args.limit:
        texts = texts[:args.limit]
    if not texts:
        return
    translate = model_translator(args.batch_size)
    start = time.time()
    for i in range(0, len(texts), CHUNK):
        new = translate_texts(texts[i:i + CHUNK], translate)
        if args.limit:
            for th, en in new.items():
                print(f"TH: {th}\nEN: {en}\n")
            continue
        done.update(new)
        save(done)
        n = min(i + CHUNK, len(texts))
        left = (time.time() - start) / n * (len(texts) - n)
        print(f"  {n:,}/{len(texts):,} translated, ~{left / 60:.0f} min left", flush=True)
    if not args.limit:
        print(f"Wrote {C.TRAFFY_TRANSLATIONS_FILE}")


if __name__ == "__main__":
    main()
