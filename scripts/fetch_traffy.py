"""Download Traffy Fondue citizen reports (Bangkok's complaint app) as ground truth for street flooding.

    python scripts/fetch_traffy.py                    # every flood report, 2022 to now (~1 h once)
    python scripts/fetch_traffy.py --all              # all reports on every 3rd day, for the "dry" class (~2 h once)
    python scripts/fetch_traffy.py --months 2026-09 2026-10 --refresh

Source: the public Traffy Fondue API (no key), the GeoJSON endpoint behind Bangkok's open dashboard. It returns
at most 1,000 reports per call (10-20 s each; bigger pages time out), and paging deep into a month of all reports
(30-60k) keeps failing, so --all fetches one day at a time, and by default only every 3rd day: where people report
problems at all changes slowly, so a third of the days gives the same picture (counts are scaled up to a full
year). Times are Bangkok local, for both the filter and the reports. Finished months and days are cached, so a
rerun only fetches what's missing plus the current month.

Outputs:
    data/raw/traffy/flood/YYYY-MM.jsonl.gz            cached flood reports per month (the fields kept, one per line)
    data/raw/traffy/all/YYYY-MM-DD.jsonl.gz           cached reports of every kind on the sampled days
    data/processed/traffy/flood_reports.parquet       every flood report: point, time, district, photo, text
    data/processed/traffy/other_counts.tif            estimated reports of anything else per ~65 m cell and year
                                                      (float32, one band per year with every sampled day cached)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import PROFILE  # noqa: E402
from src import config as C  # noqa: E402
from src.traffy import FLOOD_TYPE, count_grid, report_row  # noqa: E402

API = "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1"
PAGE = 1000           # the API's maximum per call
RETRIES = (10, 30, 60, 120, 300)
RAW_DIR = C.RAW_DIR / "traffy"
FIRST_MONTH = "2022-01"
FIRST_DAY = dt.date(2022, 1, 1)
SAMPLE_EVERY = 3  # --all fetches every n-th day


def get_page(start: str, end: str, offset: int, flood_only: bool) -> dict:
    params = {"start": start, "end": end, "limit": PAGE, "offset": offset}
    if flood_only:
        params["problem_type"] = FLOOD_TYPE
    for wait in (*RETRIES, None):
        try:
            r = requests.get(API, params=params, timeout=180)
            r.raise_for_status()
            d = r.json()
            if "features" in d:
                return d
            err = d.get("message") or "no features"
        except (requests.RequestException, ValueError) as e:
            err = str(e)[:200]
        if wait is None:
            raise RuntimeError(f"{start}..{end} offset {offset}: {err}")
        print(f"    retry in {wait} s ({err})")
        time.sleep(wait)


def fetch_period(start: str, end: str, flood_only: bool) -> list:
    """Every report from start to end (dates, inclusive), as dicts from report_row, deduplicated."""
    rows, offset, total = {}, 0, None
    while total is None or offset < total:
        d = get_page(start, end, offset, flood_only)
        total = int(d.get("total") or 0)
        feats = d["features"]
        for f in feats:
            r = report_row(f)
            if r and r["ticket_id"]:
                rows[r["ticket_id"]] = r
        offset += PAGE
        if not feats:
            break
        if total > PAGE:
            print(f"    {start}: {min(offset, total):,} / {total:,}", flush=True)
    if total and len(rows) < 0.98 * total:
        raise RuntimeError(f"{start}..{end}: only {len(rows):,} unique reports of {total:,}; try again later")
    return list(rows.values())


def cache_path(kind: str, period: str) -> Path:
    return RAW_DIR / kind / f"{period}.jsonl.gz"


def read_month(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_month(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def months_between(first: str, last: str) -> list:
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def sampled_days(months: list, every: int = SAMPLE_EVERY) -> list:
    """The days --all fetches in these months: every n-th day counted from FIRST_DAY, up to yesterday."""
    yesterday = dt.date.today() - dt.timedelta(days=1)
    days = []
    for m in months:
        p = pd.Period(m, "M")
        for d in pd.date_range(p.start_time, p.end_time.normalize(), freq="D"):
            d = d.date()
            if d <= yesterday and (d - FIRST_DAY).days % every == 0:
                days.append(d)
    return days


def update_flood_cache(months: list, refresh: bool) -> None:
    current = str(pd.Period(dt.date.today(), "M"))
    for m in months:
        path = cache_path("flood", m)
        if path.exists() and not refresh and m != current:
            continue
        t0 = time.time()
        p = pd.Period(m, "M")
        rows = fetch_period(str(p.start_time.date()), str(p.end_time.date()), flood_only=True)
        write_month(path, rows)
        print(f"  flood {m}: {len(rows):,} reports ({time.time() - t0:.0f} s)", flush=True)


def update_all_cache(days: list, refresh: bool) -> None:
    todo = [d for d in days if refresh or not cache_path("all", str(d)).exists()]
    print(f"Every report on {len(days)} sampled days: {len(todo)} to fetch")
    for i, d in enumerate(todo, start=1):
        if not refresh and cache_path("all", str(d)).exists():  # fetched meanwhile by a second run
            continue
        t0 = time.time()
        rows = fetch_period(str(d), str(d), flood_only=False)
        write_month(cache_path("all", str(d)), rows)
        print(f"  [{i}/{len(todo)}] all {d}: {len(rows):,} reports ({time.time() - t0:.0f} s)", flush=True)


def write_flood_reports(months: list) -> pd.DataFrame:
    """flood_reports.parquet from the monthly flood cache."""
    rows = {}
    for m in months:
        p = cache_path("flood", m)
        if p.exists():
            rows.update({r["ticket_id"]: r for r in read_month(p)})
    df = pd.DataFrame(list(rows.values()))
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df = df.drop(columns=["is_flood"]).sort_values("time").reset_index(drop=True)
    df["types"] = df["types"].map(lambda t: ",".join(t))
    C.TRAFFY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.TRAFFY_FLOOD_FILE, index=False)
    print(f"Wrote {C.TRAFFY_FLOOD_FILE} ({len(df):,} flood reports, {df['time'].min():%Y-%m-%d} to "
          f"{df['time'].max():%Y-%m-%d})")
    return df


def write_other_counts(every: int = SAMPLE_EVERY) -> None:
    """other_counts.tif: estimated non-flood reports per cell for each year whose sampled days are all cached
    (the sampled count x `every`)."""
    bands, years = [], []
    for y in C.TRAFFY_YEARS:
        last = min(pd.Period(f"{y}-12", "M"), pd.Period(dt.date.today(), "M"))
        paths = [cache_path("all", str(d)) for d in sampled_days(months_between(f"{y}-01", str(last)), every)]
        if not paths or not all(p.exists() for p in paths):
            continue
        lon, lat = [], []
        for p in paths:
            for r in read_month(p):
                if not r["is_flood"]:
                    lon.append(r["lon"])
                    lat.append(r["lat"])
        bands.append((count_grid(lon, lat) * every).astype("float32"))
        years.append(y)
        print(f"  {y}: {len(lon):,} other reports on {len(paths)} days, {(bands[-1] > 0).sum():,} cells with any")
    if not bands:
        print("No year with every sampled day of --all reports yet; other_counts.tif not written")
        return
    with rasterio.open(C.TRAFFY_OTHER_FILE, "w", dtype="float32", count=len(bands), nodata=None,
                       **{k: v for k, v in PROFILE.items() if k != "count"}) as dst:
        dst.update_tags(sample_every_days=every)
        for i, (y, b) in enumerate(zip(years, bands), start=1):
            dst.write(b, i)
            dst.set_band_description(i, str(y))
    print(f"Wrote {C.TRAFFY_OTHER_FILE} (years {years})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="fetch reports of every kind on sampled days, instead of flood reports (slow)")
    ap.add_argument("--every", type=int, default=SAMPLE_EVERY, help="with --all, fetch every n-th day (default 3)")
    ap.add_argument("--months", nargs="+", help="YYYY-MM months to fetch (default: 2022-01 to now)")
    ap.add_argument("--refresh", action="store_true", help="fetch cached months again")
    ap.add_argument("--reverse", action="store_true",
                    help="with --all, newest days first (run a second copy with it to fetch from both ends)")
    args = ap.parse_args()

    months = args.months or months_between(FIRST_MONTH, str(pd.Period(dt.date.today(), "M")))
    if args.all:
        days = sampled_days(months, args.every)
        update_all_cache(days[::-1] if args.reverse else days, args.refresh)
    else:
        update_flood_cache(months, args.refresh)
    write_flood_reports(months_between(FIRST_MONTH, str(pd.Period(dt.date.today(), "M"))))
    write_other_counts(args.every)


if __name__ == "__main__":
    main()
