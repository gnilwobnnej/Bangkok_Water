"""Download Traffy Fondue citizen reports (Bangkok's complaint app) as ground truth for street flooding.

    python scripts/fetch_traffy.py                    # every flood report, 2022 to now (~1 h once)
    python scripts/fetch_traffy.py --all              # all reports on every 3rd day, for the "dry" class (~2 h once)
    python scripts/fetch_traffy.py --months 2026-09 2026-10 --refresh
    python scripts/fetch_traffy.py --daily            # the Update data workflow: new reports only (~1 min a day)

Source: the public Traffy Fondue API (no key), the GeoJSON endpoint behind Bangkok's open dashboard (moved to
api.traffy.in.th on 30 Oct 2026). It returns at most 300 reports per call and can't page, and its flood filter
misses all but the last day or two, so every report is fetched two hours at a time (~45 s a day; see API) and the
flood reports are picked out by their category. --all fetches by default only every 3rd day: where people report
problems at all changes slowly, so a third of the days gives the same picture (counts are scaled up to a full
year). Times are Bangkok local, for both the windows and the reports. Finished months and days are cached, so a
rerun only fetches what's missing plus the current month.

--daily works without that cache (data/raw isn't kept between GitHub Actions runs): it fetches every report from a
few days before the newest saved one (late reports and status changes) to today, saves the flood reports and their
photo links, and adds the sampled days since each year's band of other_counts.tif was last counted (its "through"
tag) onto that band.

Outputs:
    data/raw/traffy/flood/YYYY-MM.jsonl.gz            cached flood reports per month (the fields kept, one per line)
    data/raw/traffy/all/YYYY-MM-DD.jsonl.gz           cached reports of every kind on the sampled days
    data/processed/traffy/flood_reports/YYYY-MM.parquet  every flood report: point, time, district, photo, text
    data/processed/traffy/other_counts.tif            estimated reports of anything else per ~65 m cell and year
                                                      (float32, one band per year with every sampled day cached,
                                                      tagged with the last sampled day counted)
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
from src.traffy import (FLOOD_TYPE, TEXT_MODEL, count_grid, load_text_model, merge_photos, report_files,  # noqa: E402
                        report_row, share_day, share_rows)

# Since 30 Oct 2026 (the old publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1 redirects here). It returns
# at most PAGE reports per call, newest first, and can't page (any offset > 0 gives null); its problem_type filter
# finds nothing older than a day or two, and its "total" is off. So every report is fetched, WINDOW at a time
# (split in two while a window comes back full), and the flood reports are picked out by their category.
API = "https://api.traffy.in.th/postgrest/rpc/get_bkk_geojson_feed"
PAGE = 300            # the most it returns per call
FULL = PAGE - 20      # a window with this many reports may have been cut off (a call has returned 280 of 328)
WINDOW = dt.timedelta(hours=2)
MIN_WINDOW = dt.timedelta(minutes=5)
RETRIES = (10, 30, 60, 120, 300)
DAILY_LOOKBACK = 3    # --daily fetches again the days from this many days before the newest saved report
RAW_DIR = C.RAW_DIR / "traffy"
FIRST_MONTH = "2022-01"
FIRST_DAY = dt.date(2022, 1, 1)
SAMPLE_EVERY = 3  # --all fetches every n-th day


def get_page(start: dt.datetime, end: dt.datetime) -> list:
    """The GeoJSON features of every report from start to end (Bangkok times, inclusive), at most PAGE."""
    params = {"start": f"{start:%Y-%m-%d %H:%M:%S}", "end": f"{end:%Y-%m-%d %H:%M:%S}"}
    for wait in (*RETRIES, None):
        try:
            r = requests.get(API, params=params, timeout=180)
            r.raise_for_status()
            d = r.json()
            if d is None:  # what it answers for no reports at all
                return []
            if "features" in d:
                return d["features"] or []
            err = d.get("message") or "no features"
        except (requests.RequestException, ValueError, AttributeError) as e:
            err = str(e)[:200]
        if wait is None:
            raise RuntimeError(f"{params['start']}..{params['end']}: {err}")
        print(f"    retry in {wait} s ({err})")
        time.sleep(wait)


def fetch_window(start: dt.datetime, end: dt.datetime) -> list:
    """Every report's feature from start to end, in two halves while a call comes back full."""
    feats = get_page(start, end)
    if len(feats) < FULL:
        return feats
    if end - start < MIN_WINDOW:
        raise RuntimeError(f"{start}..{end}: {len(feats)} reports in a few minutes; the API may have changed")
    mid = start + (end - start) / 2
    mid = mid.replace(microsecond=0)
    return fetch_window(start, mid) + fetch_window(mid + dt.timedelta(seconds=1), end)


def fetch_features(start: str, end: str) -> dict:
    """{ticket_id: feature} of every report from start to end (dates, inclusive, Bangkok time)."""
    feats = {}
    t, last = dt.datetime.fromisoformat(start), dt.datetime.fromisoformat(end) + dt.timedelta(days=1)
    while t < last:
        for f in fetch_window(t, min(t + WINDOW, last) - dt.timedelta(seconds=1)):
            ticket = (f.get("properties") or {}).get("ticket_id")
            if ticket:
                feats[ticket] = f
        t += WINDOW
    return feats


def fetch_period(start: str, end: str, flood_only: bool) -> list:
    """Every report from start to end (dates, inclusive), as dicts from report_row, deduplicated."""
    rows = [r for r in map(report_row, fetch_features(start, end).values()) if r and r["ticket_id"]]
    return [r for r in rows if r["is_flood"]] if flood_only else rows


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


def write_flood_reports(months: list) -> None:
    """flood_reports/YYYY-MM.parquet for each of these months with cached flood reports."""
    C.TRAFFY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for m in months:
        p = cache_path("flood", m)
        rows = read_month(p) if p.exists() else []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], format="mixed")
        df = df.drop(columns=["is_flood"]).sort_values(["time", "ticket_id"]).reset_index(drop=True)
        df["types"] = df["types"].map(lambda t: ",".join(t))
        df.to_parquet(C.TRAFFY_REPORTS_DIR / f"{m}.parquet", index=False)
        written += len(df)
    print(f"Wrote {written:,} flood reports to {C.TRAFFY_REPORTS_DIR} ({len(report_files())} months in all)")


def read_other_counts() -> tuple:
    """({year: band}, {year: last sampled day counted}, sample_every) from other_counts.tif."""
    with rasterio.open(C.TRAFFY_OTHER_FILE) as src:
        years = [int(d) for d in src.descriptions]
        bands = {y: src.read(i) for i, y in enumerate(years, start=1)}
        through = {y: src.tags(i).get("through") for i, y in enumerate(years, start=1)}
        every = int(src.tags().get("sample_every_days", SAMPLE_EVERY))
    if not all(through.values()):
        raise SystemExit(f"{C.TRAFFY_OTHER_FILE} has no 'through' tags: run scripts/fetch_traffy.py --all once "
                         "with the full download cache to rebuild it")
    return bands, {y: dt.date.fromisoformat(d) for y, d in through.items()}, every


def save_other_counts(bands: dict, through: dict, every: int) -> None:
    years = sorted(bands)
    with rasterio.open(C.TRAFFY_OTHER_FILE, "w", dtype="float32", count=len(years), nodata=None,
                       **{k: v for k, v in PROFILE.items() if k != "count"}) as dst:
        dst.update_tags(sample_every_days=every)
        for i, y in enumerate(years, start=1):
            dst.write(bands[y].astype("float32"), i)
            dst.set_band_description(i, str(y))
            dst.update_tags(i, through=str(through[y]))
    print(f"Wrote {C.TRAFFY_OTHER_FILE} (years {years}, counted through {max(through.values())})")


def write_other_counts(every: int = SAMPLE_EVERY) -> None:
    """other_counts.tif: estimated non-flood reports per cell for each year whose sampled days are all cached
    (the sampled count x `every`)."""
    bands, through = {}, {}
    for y in C.TRAFFY_YEARS:
        last = min(pd.Period(f"{y}-12", "M"), pd.Period(dt.date.today(), "M"))
        days = sampled_days(months_between(f"{y}-01", str(last)), every)
        paths = [cache_path("all", str(d)) for d in days]
        if not paths or not all(p.exists() for p in paths):
            continue
        lon, lat = [], []
        for p in paths:
            for r in read_month(p):
                if not r["is_flood"]:
                    lon.append(r["lon"])
                    lat.append(r["lat"])
        bands[y], through[y] = count_grid(lon, lat) * every, days[-1]
        print(f"  {y}: {len(lon):,} other reports on {len(paths)} days, {(bands[y] > 0).sum():,} cells with any")
    if not bands:
        print("No year with every sampled day of --all reports yet; other_counts.tif not written")
        return
    save_other_counts(bands, through, every)


def days_to_add(through: dict, every: int) -> list:
    """Sampled days not yet counted: after each year's "through" day, from the newest year with a band on (a
    new year starts a new band)."""
    first = max(through)
    months = months_between(f"{first}-01", str(pd.Period(dt.date.today(), "M")))
    return [d for d in sampled_days(months, every) if d > through.get(d.year, dt.date(d.year - 1, 12, 31))]


def bangkok_today() -> dt.date:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).date()


def replace_flood_reports(rows: list, since: dt.date) -> int:
    """Make the month files hold exactly these flood reports from `since` on (all of them refetched, with Traffy's
    category), and keep the saved ones from before. Returns how many are saved from `since` on."""
    new = pd.DataFrame(rows, columns=["ticket_id", "lon", "lat", "time", "types", "is_flood", "state", "district",
                                      "has_photo", "description"]).drop(columns=["is_flood"])
    new["time"] = pd.to_datetime(new["time"])
    new["types"] = new["types"].map(lambda t: ",".join(t))
    new["source"], new["flood_prob"] = "category", np.nan
    C.TRAFFY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(since)
    for m in months_between(str(pd.Period(since, "M")), str(pd.Period(bangkok_today(), "M"))):
        path = C.TRAFFY_REPORTS_DIR / f"{m}.parquet"
        old = pd.read_parquet(path) if path.exists() else new.iloc[:0]
        if "source" not in old:
            old["source"], old["flood_prob"] = "category", np.nan
        part = new[new["time"].dt.to_period("M").astype(str) == m]
        df = pd.concat([old[old["time"] < cut], part], ignore_index=True)
        if df.empty:
            path.unlink(missing_ok=True)
            continue
        df.sort_values(["time", "ticket_id"]).reset_index(drop=True).to_parquet(path, index=False)
    return len(new)


def daily_feed() -> None:
    """--daily: every report of each Bangkok day from a few days before the newest saved flood report (and from
    the oldest one the text model picked out, so Traffy's own category replaces its guesses, and the oldest
    sampled day still to count) to today. Saves the flood reports and their photo links, and counts the others
    onto other_counts.tif."""
    bands, through, every = read_other_counts()
    count_days = days_to_add(through, every)
    saved = pd.concat([pd.read_parquet(f) for f in report_files()], ignore_index=True) if report_files() else None
    starts = [FIRST_DAY if saved is None else saved["time"].max().date() - dt.timedelta(days=DAILY_LOOKBACK)]
    if saved is not None and "source" in saved and (saved["source"] == TEXT_MODEL).any():
        starts.append(saved.loc[saved["source"] == TEXT_MODEL, "time"].min().date())
    first, today = min(starts + count_days), bangkok_today()
    days = [first + dt.timedelta(days=n) for n in range((today - first).days + 1)]
    print(f"Every report on {len(days)} days ({days[0]} to {days[-1]}), flood reports by Traffy's category")
    rows, photos = [], {}
    for d in days:
        t0 = time.time()
        feats = fetch_features(str(d), str(d))
        if not feats and d < today:  # a Bangkok day always has hundreds; nothing means the API is broken
            raise RuntimeError(f"{d}: no reports at all; nothing saved")
        day_rows = [r for r in map(report_row, feats.values()) if r and r["ticket_id"]]
        rows += day_rows
        flood = [r for r in day_rows if r["is_flood"]]
        photos.update((r["ticket_id"], feats[r["ticket_id"]]["properties"].get("photo_url")) for r in flood
                      if r["time"] >= C.TRAFFY_PHOTOS_FROM)
        if d in count_days:
            write_month(cache_path("all", str(d)), day_rows)
        print(f"  {d}: {len(day_rows):,} reports, {len(flood):,} flood ({time.time() - t0:.0f} s)", flush=True)

    n = replace_flood_reports([r for r in rows if r["is_flood"]], first)
    print(f"Saved {n:,} flood reports from {first} on to {C.TRAFFY_REPORTS_DIR}")
    print(f"Added {merge_photos(photos):,} photo links to {C.TRAFFY_PHOTOS_FILE}")
    by_day = {}
    for r in rows:
        if not r["is_flood"]:
            by_day.setdefault(r["time"][:10], []).append(r)
    for d in count_days:
        other = by_day.get(str(d), [])
        bands[d.year] = bands.get(d.year, 0) + count_grid([r["lon"] for r in other], [r["lat"] for r in other]) * every
        through[d.year] = d
        print(f"  counted {d}: {len(other):,} other reports")
    if count_days:
        save_other_counts(bands, through, every)


# ---------- fallback: the share API and the text model (when the GeoJSON API is down, as 14-30 Oct 2026) ----------

def api_up() -> bool:
    """False when the GeoJSON API errors or can't be reached (one quick call, no retries)."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).replace(tzinfo=None, microsecond=0)
    try:
        r = requests.get(API, params={"start": f"{now - dt.timedelta(hours=1):%Y-%m-%d %H:%M:%S}",
                                      "end": f"{now:%Y-%m-%d %H:%M:%S}"}, timeout=60)
        return r.ok and (r.json() is None or "features" in r.json())
    except (requests.RequestException, ValueError, TypeError):
        return False


def merge_flood_reports(rows: list) -> int:
    """Add flood reports to the month files. A ticket already saved is kept as it is, so a report with Traffy's own
    category is never replaced by the text model's guess. Returns how many were added."""
    if not rows:
        return 0
    new = pd.DataFrame(rows).drop(columns=["is_flood"])
    new["time"] = pd.to_datetime(new["time"])
    new["types"] = new["types"].map(lambda t: ",".join(t))
    C.TRAFFY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    for m, part in new.groupby(new["time"].dt.to_period("M").astype(str)):
        path = C.TRAFFY_REPORTS_DIR / f"{m}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=part.columns)
        if "source" not in old:
            old["source"], old["flood_prob"] = "category", np.nan
        part = part[~part["ticket_id"].isin(old["ticket_id"])]
        if part.empty:
            continue
        df = pd.concat([old, part], ignore_index=True).sort_values(["time", "ticket_id"]).reset_index(drop=True)
        df.to_parquet(path, index=False)
        added += len(part)
    return added


def daily_share() -> None:
    """--daily without the GeoJSON API: every report of each UTC day since the newest saved flood report (and
    the oldest sampled day still to count), from the share API; the text model picks out the flood reports.
    Adds them to the month files, counts the others onto other_counts.tif, and saves the photo links."""
    model = load_text_model()
    if model is None:
        raise SystemExit("No text model: run scripts/train_traffy_text.py first")
    bands, through, every = read_other_counts()
    count_days = days_to_add(through, every)
    saved = pd.concat([pd.read_parquet(f) for f in report_files()], ignore_index=True) if report_files() else None
    newest = saved["time"].max().date() if saved is not None else FIRST_DAY
    # Model guesses are only added after the last report Traffy categorised itself
    categorised = saved if saved is None or "source" not in saved else saved[saved["source"].fillna("category")
                                                                               == "category"]
    last_category = str(categorised["time"].max()) if categorised is not None else ""
    # A Bangkok day starts at 17:00 UTC the day before
    first = min([newest] + count_days) - dt.timedelta(days=1)
    today = dt.datetime.now(dt.timezone.utc).date()
    days = [first + dt.timedelta(days=n) for n in range((today - first).days + 1)]
    print(f"Share API: every report on {len(days)} UTC days ({days[0]} to {days[-1]}), flood reports picked out "
          f"by the text model (cut-off {model.cutoff})")
    rows, photos = [], {}
    for d in days:
        t0 = time.time()
        results = share_day(str(d))
        day_rows = share_rows(results, model)
        rows += day_rows
        flood_ids = {r["ticket_id"] for r in day_rows if r["is_flood"]}
        photos.update((r["ticket_id"], r.get("photo_url")) for r in results if r["ticket_id"] in flood_ids)
        print(f"  {d}: {len(day_rows):,} reports, {len(flood_ids):,} flood ({time.time() - t0:.0f} s)", flush=True)

    flood = [r for r in rows if r["is_flood"] and r["time"] > last_category]
    print(f"Added {merge_flood_reports(flood):,} new flood reports to {C.TRAFFY_REPORTS_DIR}")
    keep = {r["ticket_id"] for r in flood if r["time"] >= C.TRAFFY_PHOTOS_FROM}
    print(f"Added {merge_photos({t: u for t, u in photos.items() if t in keep}):,} photo links to "
          f"{C.TRAFFY_PHOTOS_FILE}")

    by_day = {}
    for r in rows:
        if not r["is_flood"]:
            by_day.setdefault(r["time"][:10], []).append(r)
    for d in count_days:
        other = by_day.get(str(d), [])
        bands[d.year] = bands.get(d.year, 0) + count_grid([r["lon"] for r in other], [r["lat"] for r in other]) * every
        through[d.year] = d
        print(f"  counted {d}: {len(other):,} other reports")
    if count_days:
        save_other_counts(bands, through, every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="fetch reports of every kind on sampled days, instead of flood reports (slow)")
    ap.add_argument("--every", type=int, default=SAMPLE_EVERY, help="with --all, fetch every n-th day (default 3)")
    ap.add_argument("--months", nargs="+", help="YYYY-MM months to fetch (default: 2022-01 to now)")
    ap.add_argument("--refresh", action="store_true", help="fetch cached months again")
    ap.add_argument("--reverse", action="store_true",
                    help="with --all, newest days first (run a second copy with it to fetch from both ends)")
    ap.add_argument("--daily", action="store_true",
                    help="only what's new since the saved data (for the daily workflow; needs no download cache)")
    args = ap.parse_args()

    if args.daily and not api_up():
        print("The Traffy GeoJSON API is down: using the share API and the text model")
        daily_share()
        return
    if args.daily:
        daily_feed()
        return
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
