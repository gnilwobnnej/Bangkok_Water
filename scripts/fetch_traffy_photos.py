"""Find the residents' photos of the Traffy Fondue flood reports, for the map's report popups.

    python scripts/fetch_traffy_photos.py            # every day not done yet (~30-45 min the first time)
    python scripts/fetch_traffy_photos.py --days 2   # a quick test: only the first 2 days still to do

Only reports from config.TRAFFY_PHOTOS_FROM on (September 2026) get photos, to keep the fetch small. Needs
scripts/fetch_traffy.py first: photos are matched to its flood reports by ticket id.

Source: Traffy Fondue's public "share" search API (no key). Its category field is empty for 2026 reports, so it
can't be asked for floods only: every report of each day is paged through (1,000 per call, ~2-4k a day) and the
flood reports' photo links kept. Only the links are saved; the photos stay on Traffy's public bucket and the map
loads one when a report is opened. Days run in UTC, like the API's timestamps; the last few days are always
fetched again, for reports that came in late. Saved after every day, so a stopped run carries on where it was.

Outputs:
    data/processed/traffy/photos.parquet     ticket_id, photo_url
    data/processed/traffy/photos_days.json   the UTC days already searched
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.traffy import load_photos, load_reports, safe_photo_url, share_day  # noqa: E402

REFETCH_DAYS = 3  # the latest days are searched again on every run
DAYS_FILE = C.TRAFFY_PHOTOS_FILE.with_name("photos_days.json")


def flood_photos(results: list, flood_ids: set) -> dict:
    """{ticket_id: photo URL} for the flood reports among one page's results that have a usable photo."""
    out = {}
    for r in results:
        url = safe_photo_url(r.get("photo_url"))
        if r.get("ticket_id") in flood_ids and url:
            out[r["ticket_id"]] = url
    return out


def days_to_do(first: dt.date, today: dt.date, done: set) -> list:
    """Every UTC day from `first` to `today` not done yet, plus the last REFETCH_DAYS days in any case."""
    days = [first + dt.timedelta(n) for n in range((today - first).days + 1)]
    recent = today - dt.timedelta(REFETCH_DAYS - 1)
    return [d.isoformat() for d in days if d.isoformat() not in done or d >= recent]


def save(photos: dict, done: set) -> None:
    C.TRAFFY_PHOTOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.TRAFFY_PHOTOS_FILE.with_suffix(".part")
    pd.DataFrame({"ticket_id": list(photos), "photo_url": list(photos.values())}).to_parquet(tmp, index=False)
    tmp.replace(C.TRAFFY_PHOTOS_FILE)
    DAYS_FILE.write_text(json.dumps(sorted(done), indent=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, help="search only this many days (a test run)")
    args = ap.parse_args()

    reports = load_reports()
    if reports is None:
        sys.exit("No reports: run scripts/fetch_traffy.py first")
    recent = reports[reports["time"] >= pd.Timestamp(C.TRAFFY_PHOTOS_FROM)]
    flood_ids = set(recent["ticket_id"].dropna())
    photos = load_photos()
    done = set(json.loads(DAYS_FILE.read_text())) if DAYS_FILE.exists() else set()
    # Bangkok is UTC+7: the first Bangkok morning is the day before in UTC
    first = dt.date.fromisoformat(C.TRAFFY_PHOTOS_FROM) - dt.timedelta(1)
    todo = days_to_do(first, dt.datetime.now(dt.timezone.utc).date(), done)[:args.days]
    print(f"{len(flood_ids):,} flood reports since {C.TRAFFY_PHOTOS_FROM}, {len(photos):,} photos found so far; "
          f"{len(todo)} days to search")
    for day in todo:
        new = flood_photos(share_day(day), flood_ids)
        photos.update(new)
        done.add(day)
        save(photos, done)
        print(f"  {day}: {len(new):,} photos", flush=True)
    have = sum(t in photos for t in flood_ids)
    print(f"{have:,} of {len(flood_ids):,} flood reports since {C.TRAFFY_PHOTOS_FROM} have a photo; "
          f"wrote {C.TRAFFY_PHOTOS_FILE}")


if __name__ == "__main__":
    main()
