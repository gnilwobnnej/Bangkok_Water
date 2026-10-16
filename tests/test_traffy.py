"""Traffy Fondue street-flood reports: parsing, gridding, storm days and labels (src/traffy.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_traffy as ft  # noqa: E402
import train_model as tm  # noqa: E402
from src import config as C  # noqa: E402
from src.traffy import FLOOD_TYPE, cell_index, count_grid, report_row, storm_days, street_labels  # noqa: E402

U = C.LABEL_NODATA


def feature(lon, lat, types, ticket="T1", **props):
    return {"geometry": {"coordinates": [lon, lat]},
            "properties": {"ticket_id": ticket, "timestamp": "2025-09-26 18:30:00", "problem_type_fondue": types,
                           "photo_url": "https://x/p.jpg", "description": "น้ำท่วมขังหน้าบ้าน", **props}}


def test_report_row_keeps_fields_and_flags_floods():
    r = report_row(feature("100.5", "13.7", [FLOOD_TYPE, "ถนน"]))
    assert (r["lon"], r["lat"], r["is_flood"], r["has_photo"]) == (100.5, 13.7, True, True)
    assert r["time"] == "2025-09-26 18:30:00"  # Bangkok local, as the API gives it
    assert not report_row(feature(100.5, 13.7, ["ถนน", None]))["is_flood"]
    assert report_row({"geometry": None, "properties": {}}) is None


def test_points_land_in_their_grid_cell():
    # the centre of cell (row 2, col 3), a point just inside the north-west corner, and one outside the box
    lon = [C.WEST + 3.5 * C.RES, C.WEST + 1e-9, C.WEST - 0.01]
    lat = [C.NORTH - 2.5 * C.RES, C.NORTH - 1e-9, C.NORTH - 0.01]
    row, col, inside = cell_index(lon, lat)
    assert list(inside) == [True, True, False]
    assert (row[0], col[0], row[1], col[1]) == (2, 3, 0, 0)
    grid = count_grid(lon + [lon[0]], lat + [lat[0]])
    assert grid.shape == (C.HEIGHT, C.WIDTH) and grid[2, 3] == 2 and grid.sum() == 3


def test_storm_days_need_a_spike_and_rain():
    days = pd.date_range("2025-08-01", "2025-09-30", freq="D")
    times = [d + pd.Timedelta(hours=9) for d in days for _ in range(2)]  # 2 reports a day, every day
    times += [pd.Timestamp("2025-09-10 20:00")] * 30                  # a storm evening
    times += [pd.Timestamp("2025-09-20 10:00")] * 30                  # a burst of complaints on a dry day
    times = pd.Series(times)
    rain = pd.Series(0.0, index=days)
    rain["2025-09-09"] = 40  # rain the day before still counts
    assert list(storm_days(times, rain, min_reports=15, spike_ratio=3, min_rain_mm=10)) == \
        [pd.Timestamp("2025-09-10")]
    # without rain data both bursts count; without rain on the archive's missing days, too
    assert len(storm_days(times, None, min_reports=15, spike_ratio=3)) == 2
    assert len(storm_days(times, rain.where(rain.index < "2025-09-15"), min_reports=15, spike_ratio=3,
                          min_rain_mm=10)) == 2
    # too few reports is never a storm
    assert len(storm_days(times, rain, min_reports=50, spike_ratio=3, min_rain_mm=10)) == 0


def test_street_labels_case_control():
    # cells: storm flood | flood only on ordinary days | busy, no flood | quiet, no flood | nothing at all
    n_storm = np.array([1, 0, 0, 0, 0])
    n_flood = np.array([3, 2, 0, 0, 0])
    n_other = np.array([0, 9, 5, 4, 0])
    assert list(street_labels(n_storm, n_flood, n_other, min_flood=1, min_other=5)) == [1, U, 0, U, U]


def test_event_info_names_street_years_and_keeps_them_out_of_the_river_model(tmp_path, monkeypatch):
    info = tm.event_info(Path("x/traffy_2023.tif"))
    assert (info["id"], info["year"]) == ("traffy_2023", 2023) and "street" in info["name"]
    for name in ["radar_2020.tif", "traffy_2023.tif", f"traffy_{C.TEST_YEARS_FROM}.tif"]:
        (tmp_path / name).touch()
    monkeypatch.setattr(C, "FLOOD_LABELS_DIR", tmp_path)
    assert [p.name for p in tm.street_label_files()] == ["traffy_2023.tif"]  # the test year is held back
    assert all("traffy" not in p.name for p in tm.default_label_files())
    with pytest.raises(SystemExit, match="test year"):
        tm.load_events([tmp_path / f"traffy_{C.TEST_YEARS_FROM}.tif"])


def test_sampled_days_are_every_nth_day_and_stop_before_today():
    import datetime as dt
    days = ft.sampled_days(["2022-01", "2022-02"], every=3)
    assert days[:3] == [dt.date(2022, 1, 1), dt.date(2022, 1, 4), dt.date(2022, 1, 7)]
    assert all((d - ft.FIRST_DAY).days % 3 == 0 for d in days) and len(days) == 20  # 59 days / 3
    current = ft.months_between(f"{dt.date.today():%Y-%m}", f"{dt.date.today():%Y-%m}")
    assert all(d < dt.date.today() for d in ft.sampled_days(current, every=1))


def test_daily_adds_only_uncounted_days_and_starts_a_band_each_new_year():
    import datetime as dt
    today = dt.date.today()
    counted = today - dt.timedelta(days=4)
    through = {counted.year: counted}
    assert ft.days_to_add(through, every=1) == [today - dt.timedelta(days=k) for k in (3, 2, 1)]
    new_year = ft.days_to_add({today.year - 1: dt.date(today.year - 1, 12, 31)}, every=1)
    assert all(d.year == today.year for d in new_year) and len(new_year) == today.timetuple().tm_yday - 1


def test_other_counts_keep_their_through_day(tmp_path, monkeypatch):
    import datetime as dt
    monkeypatch.setattr(C, "TRAFFY_OTHER_FILE", tmp_path / "other_counts.tif")
    bands = {2025: np.ones((C.HEIGHT, C.WIDTH)), 2026: np.zeros((C.HEIGHT, C.WIDTH))}
    through = {2025: dt.date(2025, 12, 30), 2026: dt.date(2026, 10, 13)}
    ft.save_other_counts(bands, through, every=3)
    got_bands, got_through, every = ft.read_other_counts()
    assert got_through == through and every == 3 and got_bands[2025].sum() == C.HEIGHT * C.WIDTH


def test_flood_reports_are_saved_and_loaded_by_month(tmp_path, monkeypatch):
    from src.traffy import load_reports
    monkeypatch.setattr(ft, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(C, "TRAFFY_REPORTS_DIR", tmp_path / "reports")
    rows = {"2026-09": [report_row(feature(100.5, 13.7, [FLOOD_TYPE], ticket="A", timestamp="2026-09-30 23:00:00"))],
            "2026-10": [report_row(feature(100.5, 13.7, [FLOOD_TYPE], ticket=t, timestamp=f"2026-10-0{d} 08:00:00"))
                        for t, d in (("C", 2), ("B", 1))]}
    for m, r in rows.items():
        ft.write_month(ft.cache_path("flood", m), r)
    ft.write_flood_reports(["2026-08", "2026-09", "2026-10"])  # August has no cache: no file
    assert [p.name for p in sorted((tmp_path / "reports").iterdir())] == ["2026-09.parquet", "2026-10.parquet"]
    df = load_reports()
    assert list(df["ticket_id"]) == ["A", "B", "C"] and "is_flood" not in df and df["types"][0] == FLOOD_TYPE


def test_flood_form_answers_are_translated_exactly():
    from src.traffy import duration_en, split_description
    pieces, form = split_description("น้ำท่วมขังหน้าบ้าน\nช่วยด้วยค่ะ ความสูงระดับหัวเข่าค่ะ น้ำท่วมมาแล้วน้อยกว่า 3 ชั่วโมง ")
    assert pieces == ["น้ำท่วมขังหน้าบ้าน", "ช่วยด้วยค่ะ"]  # line by line for the model
    assert form == "Water: knee-deep. Flooded for less than 3 hours."
    assert duration_en("2วัน") == "2 days" and duration_en("มากกว่า 1 สัปดาห์") == "more than 1 week"
    assert duration_en("น้อยก") == ""  # cut off by the 255-character limit: dropped
    # an answer that isn't one of the form's goes to the model instead
    pieces, form = split_description("ท่อตัน ความสูงระดับน้ำขัง ไม่ลงท่อระบายน้ำ")
    assert form == "" and pieces == ["ท่อตัน  น้ำขัง ไม่ลงท่อระบายน้ำ"]


def test_stock_phrases_and_districts_skip_the_model():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import translate_traffy as tt
    from src.traffy import district_en, fixed_en
    assert fixed_en("น้ำท่วมครับ") == "Flooding" and fixed_en("เขต: บางกะปิ") == "District: Bang Kapi"
    assert district_en("เขตประเวศ") == "Prawet" and district_en("บางพลี") == "Bang Phli" and district_en(None) == ""
    seen = []

    def stub(pieces):
        seen.extend(pieces)
        return [f"<{p}>" for p in pieces]
    got = tt.translate_texts(["น้ำท่วม", "ซอยลึก\nOK 123 ความสูงระดับข้อเท้า น้ำท่วมมาแล้ว 1 วัน"], stub)
    assert seen == ["ซอยลึก"]  # stock phrase, English text and form answers never reach the model
    assert got == {"น้ำท่วม": "Flooding.", "ซอยลึก\nOK 123 ความสูงระดับข้อเท้า น้ำท่วมมาแล้ว 1 วัน":
                   "<ซอยลึก>. OK 123. Water: ankle-deep. Flooded for 1 day."}
    assert tt.pending(pd.Series(["ก", "ข", "ก", " ", None]), {"ข": "b"}) == ["ก"]


def test_reports_carry_their_english_text(tmp_path, monkeypatch):
    from src.traffy import load_reports
    monkeypatch.setattr(C, "TRAFFY_REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(C, "TRAFFY_TRANSLATIONS_FILE", tmp_path / "translations.parquet")
    (tmp_path / "reports").mkdir()
    pd.DataFrame({"time": pd.to_datetime(["2026-09-26 18:00", "2026-09-26 19:00"]),
                  "description": ["น้ำท่วม", "ใหม่"]}).to_parquet(tmp_path / "reports" / "2026-09.parquet")
    assert load_reports()["description_en"].isna().all()  # nothing translated yet
    pd.DataFrame({"description": ["น้ำท่วม"], "description_en": ["Flooding."]}).to_parquet(
        tmp_path / "translations.parquet")
    assert load_reports()["description_en"].tolist()[0] == "Flooding." and pd.isna(load_reports()["description_en"][1])


def test_report_popup_escapes_text_and_hides_the_rest_behind_read_more():
    from src.viz import report_popup_html, report_text, shorten
    short = report_popup_html("26 Sep 2026 18:00", "Prawet", "Water <b>knee</b> deep")
    assert "&lt;b&gt;knee&lt;/b&gt;" in short and "<details>" not in short and "Machine-translated" in short
    long_text = "word " * 60
    pop = report_popup_html("t", "", long_text.strip(), translated=False, preview=50)
    assert "<details>" in pop and "Read more" in pop and "Not translated yet" in pop
    assert shorten("one two three four", 10, "!") == "one two…!" and shorten("short", 10) == "short"
    r = pd.DataFrame({"description": ["ไทย", "ไทย2"], "description_en": ["English", None]})
    assert report_text(r).tolist() == ["English", "ไทย2"]
