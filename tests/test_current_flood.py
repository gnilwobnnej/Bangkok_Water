import json

import numpy as np
import pandas as pd
import pytest

from src import current_flood as cf

RAIN_COLS, RIVER_COLS = ["date", "mm"], ["date", "m3s", "normal_m3s"]


def passes(*dates):
    return pd.DataFrame({"date": list(dates), "flooded_km2": [1.0] * len(dates)})


@pytest.fixture
def rain():
    return pd.DataFrame({"date": [f"2026-09-0{d}" for d in range(1, 6)], "mm": [1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.fixture
def river():
    return pd.DataFrame({"date": ["2026-09-03", "2026-09-05"], "m3s": [200.0, 150.0], "normal_m3s": [100.0, 100.0]})


# ---------- pass_weather ----------

def test_rain_on_the_day_and_over_three_days(rain, river):
    out = cf.pass_weather(passes("2026-09-03", "2026-09-05"), rain, river)
    assert list(out["rain_day_mm"]) == [3.0, 5.0]
    assert list(out["rain_3day_mm"]) == [1 + 2 + 3, 3 + 4 + 5]


def test_river_flow_and_share_of_normal(rain, river):
    out = cf.pass_weather(passes("2026-09-03", "2026-09-05"), rain, river)
    assert list(out["discharge_m3s"]) == [200.0, 150.0]
    assert list(out["discharge_pct_normal"]) == [200.0, 150.0]


def test_incomplete_three_day_window_is_missing_not_a_partial_sum(rain, river):
    out = cf.pass_weather(passes("2026-09-02"), rain, river)
    assert out["rain_day_mm"].iloc[0] == 2.0
    assert np.isnan(out["rain_3day_mm"].iloc[0]), "only 2 of the 3 days have rain data"


def test_dates_without_weather_are_missing(rain, river):
    out = cf.pass_weather(passes("2026-09-04", "2026-10-01"), rain, river)
    assert np.isnan(out["discharge_m3s"]).all()  # no river data on either date
    assert np.isnan(out.loc[1, ["rain_day_mm", "rain_3day_mm"]].astype(float)).all()


def test_no_weather_at_all():
    out = cf.pass_weather(passes("2026-09-03"), pd.DataFrame(columns=RAIN_COLS), pd.DataFrame(columns=RIVER_COLS))
    for col in ("rain_day_mm", "rain_3day_mm", "discharge_m3s", "discharge_pct_normal"):
        assert np.isnan(out[col].iloc[0])


def test_input_is_not_modified(rain, river):
    p = passes("2026-09-03")
    cf.pass_weather(p, rain, river)
    assert list(p.columns) == ["date", "flooded_km2"]


# ---------- load_summary ----------

def test_old_summary_without_weather_still_loads(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"passes": [{"date": "2026-09-03", "flooded_km2": 5.0}]}))
    monkeypatch.setattr(cf.C, "RADAR_SUMMARY_FILE", path)
    passes_df, rain_df, river_df, summary = cf.load_summary()
    assert rain_df.empty and river_df.empty
    assert np.isnan(passes_df["rain_day_mm"].iloc[0])
    assert summary["passes"][0]["flooded_km2"] == 5.0


def test_missing_summary_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cf.C, "RADAR_SUMMARY_FILE", tmp_path / "nope.json")
    assert cf.load_summary() is None


# ---------- radar_labels ----------

def test_radar_labels():
    radar = {
        #                    flooded  dry  partial  unseen  permanent
        "flood_pct": np.array([80.0, 5.0, 30.0, 90.0, 90.0]),
        "valid_pct": np.array([100.0, 100.0, 100.0, 20.0, 100.0]),
        "permanent_pct": np.array([0.0, 0.0, 0.0, 0.0, 80.0]),
    }
    assert list(cf.radar_labels(radar)) == [1, 0, -1, -1, -1]


def test_radar_label_thresholds_are_inclusive_for_flooded():
    radar = {"flood_pct": np.array([cf.FLOODED_PCT, cf.DRY_PCT]),
             "valid_pct": np.array([50.0, 50.0]), "permanent_pct": np.array([0.0, 0.0])}
    assert list(cf.radar_labels(radar)) == [1, -1]


# ---------- district_flooded_pct ----------

def test_district_flooded_pct(area):
    flood = np.zeros(area.dem.shape, dtype="float32")
    flood[area.district_ids == 1] = 100
    radar = {"flood_pct": flood, "valid_pct": np.full(area.dem.shape, 100.0), "permanent_pct": np.zeros(area.dem.shape)}
    out = cf.district_flooded_pct(area, radar).set_index("district_id")
    assert out.loc[1, "flooded_now_pct"] == pytest.approx(100)
    assert out.loc[2, "flooded_now_pct"] == 0
    assert out.loc[1, "flooded_now_km2"] == pytest.approx(area.cell_area_km2[area.district_ids == 1].sum())


def test_district_unseen_by_radar_is_zero_not_an_error(area):
    radar = {"flood_pct": np.full(area.dem.shape, 100.0), "valid_pct": np.zeros(area.dem.shape),
             "permanent_pct": np.zeros(area.dem.shape)}
    out = cf.district_flooded_pct(area, radar)
    assert (out["flooded_now_pct"] == 0).all()
