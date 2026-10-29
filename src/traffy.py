"""Traffy Fondue citizen reports: Bangkok's complaint app, as ground truth for street flooding.

Satellites can't see water between buildings; residents can. Each report is a point with a time (Bangkok local),
categories, a description and usually a photo. Flood reports (category "น้ำท่วม") show where streets flooded;
reports of anything else show where people report things at all, which is what the labels need for their
"dry" class (see street_labels).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config as C

FLOOD_TYPE = "น้ำท่วม"  # "flood": the category Traffy reports carry, and the API's problem_type filter


def report_row(feature: dict) -> dict | None:
    """The fields kept from one GeoJSON feature of the Traffy API (None without a usable point)."""
    p = feature.get("properties") or {}
    try:
        lon, lat = (float(v) for v in feature["geometry"]["coordinates"][:2])
    except (KeyError, TypeError, ValueError):
        return None
    types = [t for t in (p.get("problem_type_fondue") or []) if t]
    return {
        "ticket_id": p.get("ticket_id"),
        "lon": lon,
        "lat": lat,
        "time": p.get("timestamp"),  # Bangkok local time, no zone
        "types": types,
        "is_flood": FLOOD_TYPE in types,
        "state": p.get("state"),
        "district": p.get("district"),
        "has_photo": bool(p.get("photo_url")),
        "description": (p.get("description") or "")[:300],
    }


def cell_index(lon, lat):
    """(row, col, inside) on the analysis grid for arrays of points; row/col are -1 outside the study area."""
    lon, lat = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    col = np.floor((lon - C.WEST) / C.RES).astype("int64")
    row = np.floor((C.NORTH - lat) / C.RES).astype("int64")
    inside = (col >= 0) & (col < C.WIDTH) & (row >= 0) & (row < C.HEIGHT)
    return np.where(inside, row, -1), np.where(inside, col, -1), inside


def count_grid(lon, lat, shape=None) -> np.ndarray:
    """Reports per grid cell (int32), points outside the study area dropped."""
    shape = shape or (C.HEIGHT, C.WIDTH)
    row, col, inside = cell_index(lon, lat)
    grid = np.zeros(shape, dtype="int32")
    np.add.at(grid, (row[inside], col[inside]), 1)
    return grid


def storm_days(flood_times: pd.Series, rain: pd.Series | None = None, min_reports: int = None,
               spike_ratio: float = None, min_rain_mm: float = None) -> pd.DatetimeIndex:
    """Days when streets flooded city-wide: many more flood reports than usual for the time of year.

    A day counts when its flood reports are >= `min_reports` and >= `spike_ratio` x the median daily count of the
    surrounding 61 days. With `rain` (daily mm, Bangkok dates), the day must also have >= `min_rain_mm` on it or
    the day before, so a backlog of drain complaints after a quiet day doesn't count as a storm.
    """
    min_reports = C.TRAFFY_STORM_MIN_REPORTS if min_reports is None else min_reports
    spike_ratio = C.TRAFFY_STORM_SPIKE if spike_ratio is None else spike_ratio
    min_rain_mm = C.TRAFFY_STORM_MIN_RAIN_MM if min_rain_mm is None else min_rain_mm
    days = pd.to_datetime(flood_times).dt.normalize()
    if days.empty:
        return pd.DatetimeIndex([])
    daily = days.value_counts().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
    typical = daily.rolling(61, center=True, min_periods=1).median().clip(lower=1)
    storm = (daily >= min_reports) & (daily >= spike_ratio * typical)
    if rain is not None:  # days without rain data (the archive lags a few days) aren't ruled out
        r = rain.reindex(daily.index)
        wet = (r >= min_rain_mm) | (r.shift(1) >= min_rain_mm) | (r.isna() & r.shift(1).isna())
        storm &= wet.to_numpy()
    return daily.index[storm.to_numpy()]


def street_labels(n_storm_flood: np.ndarray, n_flood: np.ndarray, n_other: np.ndarray,
                  min_flood: int = None, min_other: int = None) -> np.ndarray:
    """1 flooded, 0 dry, LABEL_NODATA unknown for one year, by case-control.

    flooded  >= `min_flood` flood reports on storm days
    dry      no flood report at all, but >= `min_other` reports of other problems: people here use the app,
             so a flood would likely have been reported (this keeps "dry" in the same busy, built-up places as
             "flooded", instead of in empty farmland where nobody reports anything)
    unknown  everything else: no reports, few reports, or flood reports only on ordinary days (often chronic
             drain complaints rather than a flood that day)
    """
    min_flood = C.TRAFFY_MIN_STORM_REPORTS if min_flood is None else min_flood
    min_other = C.TRAFFY_MIN_OTHER_REPORTS if min_other is None else min_other
    labels = np.full(n_flood.shape, C.LABEL_NODATA, dtype="uint8")
    labels[(n_flood == 0) & (n_other >= min_other)] = 0
    labels[n_storm_flood >= min_flood] = 1
    return labels


def load_reports() -> pd.DataFrame | None:
    """Every flood report (scripts/fetch_traffy.py), oldest first, or None if not fetched. `description_en` is the
    English text (scripts/translate_traffy.py), NaN where it hasn't been translated."""
    files = report_files()
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    en = load_translations()
    df["description_en"] = df["description"].map(en) if en else np.nan
    photos = load_photos()
    df["photo_url"] = df["ticket_id"].map(photos) if photos and "ticket_id" in df else np.nan
    return df


def load_translations() -> dict:
    """{Thai description: English}, empty if nothing has been translated."""
    if not C.TRAFFY_TRANSLATIONS_FILE.exists():
        return {}
    t = pd.read_parquet(C.TRAFFY_TRANSLATIONS_FILE)
    return dict(zip(t["description"], t["description_en"]))


# ---------- photos ----------
PHOTO_HOST = "https://storage.googleapis.com/traffy_public_bucket/"  # where Traffy keeps residents' photos


def load_photos() -> dict:
    """{ticket_id: photo URL} (scripts/fetch_traffy_photos.py), empty if not fetched."""
    if not C.TRAFFY_PHOTOS_FILE.exists():
        return {}
    p = pd.read_parquet(C.TRAFFY_PHOTOS_FILE)
    return dict(zip(p["ticket_id"], p["photo_url"]))


def merge_photos(new: dict) -> int:
    """Add {ticket_id: URL} to photos.parquet (unsafe links dropped); returns how many were new."""
    photos = load_photos()
    add = {t: u for t, u in new.items() if t not in photos and safe_photo_url(u)}
    if add:
        photos.update(add)
        C.TRAFFY_PHOTOS_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticket_id": list(photos), "photo_url": list(photos.values())}).to_parquet(
            C.TRAFFY_PHOTOS_FILE, index=False)
    return len(add)


def safe_photo_url(url) -> str | None:
    """The URL if it is a plain https link into Traffy's public photo bucket, else None. The links come from a
    public API, so anything else is never put into the map's HTML."""
    if not isinstance(url, str) or not url.startswith(PHOTO_HOST):
        return None
    if re.search(r"[\s\"'<>\\`]", url):
        return None
    return url


# ---------- English text ----------
# Traffy's flood form ends every report with "ความสูงระดับ <depth> น้ำท่วมมาแล้ว <duration>" ("water depth ...
# flooded for ..."). Those parts are translated exactly here; the machine translation only gets the free text.
DEPTH_MARK, DURATION_MARK = "ความสูงระดับ", "น้ำท่วมมาแล้ว"
DEPTHS = [("สูงกว่าเอว", "above the waist"), ("เอว", "waist-deep"), ("เข่า", "knee-deep"),
          ("แข้ง", "shin-deep"), ("ตาตุ่ม", "ankle-deep"), ("ข้อเท้า", "ankle-deep")]  # first match wins
UNITS = {"ชั่วโมง": "hour", "ชม": "hour", "วัน": "day", "สัปดาห์": "week", "อาทิตย์": "week", "เดือน": "month"}
QUALIFIERS = {"น้อยกว่า": "less than ", "มากกว่า": "more than ", "ประมาณ": "about ", "": ""}
POLITE = re.compile(r"(นะคะ|นะครับ|ค่ะ|คะ|ครับ|จ้า|ค่า)+$")
THAI = re.compile(r"[฀-๿]")


def _clean(s: str) -> str:
    return POLITE.sub("", s.strip()).strip()


def depth_en(s: str) -> str | None:
    """The form's depth answer in English, or None if it isn't one of the form's answers."""
    s = _clean(s)
    about = s.startswith("ประมาณ")
    s = s.removeprefix("ประมาณ").strip()
    if not s or len(s) > 12:
        return None
    for th, en in DEPTHS:
        if th in s:
            return ("about " if about else "") + en
    return None


def duration_en(s: str) -> str | None:
    """The form's duration answer in English ("" if it was cut off), or None if it isn't one."""
    s = _clean(s)
    if s.startswith("ตั้งแต่เมื่อคืน"):
        return "since last night"
    m = re.fullmatch(r"(น้อยกว่า|มากกว่า|ประมาณ)?\s*(\d+)\s*(ชั่วโมง|ชม\.?|วัน|สัปดาห์|อาทิตย์|เดือน)", s)
    if m:
        n, unit = int(m[2]), UNITS[m[3].rstrip(".")]
        return f"{QUALIFIERS[m[1] or '']}{n} {unit}{'s' if n != 1 else ''}"
    if re.fullmatch(r"(น้อยกว่า|มากกว่า)?\s*\d*", s) or any(q.startswith(s) for q in ("น้อยกว่า", "มากกว่า")):
        return ""  # cut off by Traffy's 255-character limit
    return None


def split_description(text: str) -> tuple[list, str]:
    """(free-text pieces to machine-translate, the form's depth and duration already in English)."""
    body, form = text, []
    i = text.rfind(DEPTH_MARK)
    if i >= 0:
        body, tail = text[:i], text[i + len(DEPTH_MARK):]
        depth, _, duration = tail.partition(DURATION_MARK)
        d = depth_en(depth)
        if d is None:
            body += " " + depth
        else:
            form.append(f"Water: {d}.")
        if duration.strip():
            u = duration_en(duration)
            if u is None:
                body += " " + duration
            elif u:
                form.append(f"Flooded for {u}.")
    pieces = [p.strip() for p in re.split(r"\n+", body) if p.strip()]
    return pieces, " ".join(form)


# The commonest short reports, translated exactly (polite endings such as ค่ะ/ครับ removed first)
PHRASES = {
    "น้ำท่วม": "Flooding", "นำ้ท่วม": "Flooding", "ไม่แสดงรายละเอียด": "No details given",
    "น้ำท่วมขัง": "Standing water", "น้ำขัง": "Standing water", "น้ำท่วมสูง": "Deep flooding",
    "น้ำท่วมหนัก": "Heavy flooding", "น้ำท่วมถนน": "Flooded road", "น้ำท่วมซอย": "Flooded soi (side street)",
    "น้ำท่วมขังในซอย": "Standing water in the soi (side street)", "น้ำท่วมบ้าน": "Flooded house",
    "น้ำท่วมเข้าบ้าน": "Water coming into the house", "น้ำท่วมฟุตบาท": "Flooded pavement",
    "ฝนตกน้ำท่วม": "Flooding after rain", "ฝนตกน้ำท่วมขัง": "Standing water after rain",
    "ท่อระบายน้ำตัน": "Blocked drain", "ท่อตัน": "Blocked drain", "ถนน": "Road",
    "สวัสดี": "Hello", "ขอบคุณ": "Thank you",
}
# Bangkok's 50 districts (English as in districts.geojson), and the neighbouring ones that reports also name
DISTRICTS = {
    "พระนคร": "Phra Nakhon", "ดุสิต": "Dusit", "หนองจอก": "Nong Chok", "บางรัก": "Bang Rak", "บางเขน": "Bang Khen",
    "บางกะปิ": "Bang Kapi", "ปทุมวัน": "Pathum Wan", "ป้อมปราบศัตรูพ่าย": "Pom Prap Sattru Phai",
    "พระโขนง": "Phra Khanong", "มีนบุรี": "Min Buri", "ลาดกระบัง": "Lat Krabang", "ยานนาวา": "Yan Nawa",
    "สัมพันธวงศ์": "Samphanthawong", "พญาไท": "Phaya Thai", "ธนบุรี": "Thon Buri", "บางกอกใหญ่": "Bangkok Yai",
    "ห้วยขวาง": "Huai Khwang", "คลองสาน": "Khlong San", "ตลิ่งชัน": "Taling Chan", "บางกอกน้อย": "Bangkok Noi",
    "บางขุนเทียน": "Bang Khun Thian", "ภาษีเจริญ": "Phasi Charoen", "หนองแขม": "Nong Khaem",
    "ราษฎร์บูรณะ": "Rat Burana", "บางพลัด": "Bang Phlat", "ดินแดง": "Din Daeng", "บึงกุ่ม": "Bueng Kum",
    "สาทร": "Sathon", "บางซื่อ": "Bang Sue", "จตุจักร": "Chatuchak", "บางคอแหลม": "Bang Kho Laem",
    "ประเวศ": "Prawet", "คลองเตย": "Khlong Toei", "สวนหลวง": "Suan Luang", "จอมทอง": "Chom Thong",
    "ดอนเมือง": "Don Mueang", "ราชเทวี": "Ratchathewi", "ลาดพร้าว": "Lat Phrao", "วัฒนา": "Vadhana",
    "บางแค": "Bang Khae", "หลักสี่": "Lak Si", "สายไหม": "Sai Mai", "คันนายาว": "Khan Na Yao",
    "สะพานสูง": "Saphan Sung", "วังทองหลาง": "Wang Thonglang", "คลองสามวา": "Khlong Sam Wa", "บางนา": "Bang Na",
    "ทวีวัฒนา": "Thawi Watthana", "ทุ่งครุ": "Thung Khru", "บางบอน": "Bang Bon",
    # outside Bangkok
    "กระทุ่มแบน": "Krathum Baen", "บางพลี": "Bang Phli", "ลำลูกกา": "Lam Luk Ka", "ปากเกร็ด": "Pak Kret",
    "เมืองสมุทรปราการ": "Mueang Samut Prakan", "เมืองสมุทรสาคร": "Mueang Samut Sakhon",
}
LABELS = {"เขต": "District", "แขวง": "Subdistrict", "ถนน": "Road", "ซอย": "Soi", "สถานที่": "Place",
          "เวลา": "Time", "วันที่": "Date"}


def fixed_en(piece: str) -> str | None:
    """An exact English version of a stock phrase or a "เขต: <district>" label, or None to machine-translate."""
    s = _clean(piece)
    if s in PHRASES:
        return PHRASES[s]
    m = re.fullmatch(r"(\S+?)\s*:\s*(.+)", s)
    if m and m[1] in LABELS:
        value = m[2].strip()
        if m[1] == "เขต" and value.removeprefix("เขต").strip() in DISTRICTS:
            return f"District: {district_en(value)}"
        if not THAI.search(value):
            return f"{LABELS[m[1]]}: {value}"
    return None


def district_en(thai) -> str:
    """A report's district in English (spelled as in districts.geojson), or the Thai name if it isn't listed."""
    if not isinstance(thai, str) or not thai.strip():
        return ""
    s = thai.strip().removeprefix("เขต").strip()
    return DISTRICTS.get(s, s)


def join_english(pieces_en: list, form_en: str) -> str:
    text = " ".join(p.strip().rstrip(".") + "." for p in pieces_en if p.strip())
    return " ".join(s for s in (text, form_en) if s)


def report_files() -> list:
    """The monthly flood-report files, oldest first."""
    return sorted(C.TRAFFY_REPORTS_DIR.glob("*.parquet")) if C.TRAFFY_REPORTS_DIR.exists() else []


def load_summary() -> dict | None:
    """Storm days and per-year label counts (scripts/build_traffy_labels.py), or None."""
    if not C.TRAFFY_STORMS_FILE.exists():
        return None
    import json
    return json.loads(C.TRAFFY_STORMS_FILE.read_text(encoding="utf-8"))


def reports_in(reports: pd.DataFrame, period: str, storms=()) -> pd.DataFrame:
    """Flood reports of a period: "YYYY" = that year's reports on storm days; "YYYY-MM-DD" = that storm day and
    the day after (reports trail the rain)."""
    day = reports["time"].dt.normalize()
    if len(period) == 4:
        return reports[(reports["time"].dt.year == int(period)) & day.isin(pd.to_datetime(list(storms)))]
    d = pd.Timestamp(period)
    return reports[day.isin([d, d + pd.Timedelta(days=1)])]


# ---------- the share API, and recognising flood reports from their text ----------
# Traffy's GeoJSON API (scripts/fetch_traffy.py) has returned 404 since 14 Oct 2026. The share search API still
# works, a UTC day at a time, but leaves the category empty for 2026 reports and ignores every filter: so each
# day's reports are all fetched, and TextModel decides which are flood reports.
SHARE_API = "https://publicapi.traffy.in.th/share/teamchadchart/search"
SHARE_PAGE = 1000  # the most it returns per call
SHARE_RETRIES = (10, 30, 60, 120, 300)
TEXT_MODEL = "text model"  # `source` of a flood report recognised by TextModel; "category" = Traffy's own
# About flooding, but not a new flood: cases filed later by staff ("received the case on 27 Sep ...") and claims
# for flood-relief payments. Rare among Traffy's own flood reports (189 of 44,443), but common after a storm.
NOT_A_NEW_FLOOD = re.compile(r"(?:ได้)?รับแจ้งเคส|แจ้งเคส\s*เมื่อ|เยียวยา")


def share_page(day: str, offset: int) -> list:
    """One page of every report on a UTC day (YYYY-MM-DD), newest first."""
    import time

    import requests
    params = {"start": day, "end": day, "limit": SHARE_PAGE, "offset": offset}
    for wait in (*SHARE_RETRIES, None):
        try:
            r = requests.get(SHARE_API, params=params, timeout=180)
            r.raise_for_status()
            d = r.json()
            if isinstance(d.get("results"), list):
                return d["results"]
            err = d.get("message") or "no results"
        except (requests.RequestException, ValueError) as e:
            err = str(e)[:200]
        if wait is None:
            raise RuntimeError(f"share API {day} offset {offset}: {err}")
        print(f"    retry in {wait} s ({err})")
        time.sleep(wait)


def share_day(day: str) -> list:
    """Every report on a UTC day from the share API (raw results, one per ticket)."""
    rows, offset = {}, 0
    while True:
        page = share_page(day, offset)
        rows.update((r["ticket_id"], r) for r in page if r.get("ticket_id"))
        offset += SHARE_PAGE
        if len(page) < SHARE_PAGE:
            return list(rows.values())


class TextModel:
    """Chance that a report is about flooding, from its text: TF-IDF on character 2-4-grams (sublinear tf, L2
    norm) and logistic regression, trained by scripts/train_traffy_text.py and kept as plain arrays."""

    def __init__(self, terms, idf, coef, intercept: float, cutoff: float = 0.9):
        self.terms, self.idf, self.coef = np.asarray(terms), np.asarray(idf), np.asarray(coef)
        self.intercept, self.cutoff = float(intercept), float(cutoff)
        self._counter = None

    def predict_proba(self, texts) -> np.ndarray:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import normalize
        if self._counter is None:
            self._counter = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                            vocabulary={t: i for i, t in enumerate(self.terms)})
        X = self._counter.transform(["" if t is None else str(t) for t in texts]).astype("float64")
        X.data = np.log(X.data) + 1  # sublinear tf
        X = normalize(X.multiply(self.idf).tocsr())
        return 1 / (1 + np.exp(-(X @ self.coef + self.intercept)))

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, terms=self.terms, idf=self.idf, coef=self.coef,
                            intercept=np.array(self.intercept))


def load_text_model() -> TextModel | None:
    """The trained TextModel with its cut-off, or None if scripts/train_traffy_text.py hasn't run."""
    path = C.TRAFFY_TEXT_MODEL_FILE
    if not path.exists():
        return None
    import json
    d = np.load(path, allow_pickle=False)
    meta = path.with_suffix(".json")
    cutoff = json.loads(meta.read_text())["cutoff"] if meta.exists() else 0.9
    return TextModel(d["terms"], d["idf"], d["coef"], float(d["intercept"]), cutoff)


def share_row(result: dict, flood_prob: float, cutoff: float) -> dict | None:
    """A share-API result as a report_row dict (time in Bangkok local), flagged a flood report when the text
    model's chance is at least the cut-off and it isn't about an earlier flood (NOT_A_NEW_FLOOD). None without a
    usable point."""
    try:
        lon, lat = (float(v) for v in result["coords"][:2])
        when = pd.Timestamp(result["timestamp"]).tz_convert("Asia/Bangkok")
    except (KeyError, TypeError, ValueError):
        return None
    m = re.search(r"เขต\s*([^\s,]+)", result.get("address") or "")
    is_flood = flood_prob >= cutoff and not NOT_A_NEW_FLOOD.search(result.get("description") or "")
    return {
        "ticket_id": result.get("ticket_id"),
        "lon": lon,
        "lat": lat,
        "time": when.strftime("%Y-%m-%d %H:%M:%S"),
        "types": [FLOOD_TYPE] if is_flood else [],
        "is_flood": bool(is_flood),
        "state": result.get("state"),
        "district": m[1] if m else None,
        "has_photo": bool(result.get("photo_url")),
        "description": (result.get("description") or "")[:300],
        "source": TEXT_MODEL,
        "flood_prob": round(float(flood_prob), 3),
    }


def share_rows(results: list, model: TextModel) -> list:
    """report_row dicts for a day of share-API results, each classified by the text model."""
    prob = model.predict_proba([r.get("description") for r in results]) if results else []
    rows = [share_row(r, p, model.cutoff) for r, p in zip(results, prob)]
    return [r for r in rows if r and r["ticket_id"]]
