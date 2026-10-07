"""History-keeping and login logic of scripts/fetch_current_flood.py (no network or Earth Engine calls)."""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_current_flood as fetch  # noqa: E402


def recs(*pairs):
    return [{"date": d, "v": v} for d, v in pairs]


def test_merge_keeps_old_adds_new_and_sorts():
    old = recs(("2026-09-03", 1), ("2026-08-20", 2))
    new = recs(("2026-10-01", 3))
    assert [r["date"] for r in fetch.merge_by_date(old, new)] == ["2026-08-20", "2026-09-03", "2026-10-01"]


def test_merge_new_values_replace_old_for_the_same_date():
    merged = fetch.merge_by_date(recs(("2026-09-03", 1)), recs(("2026-09-03", 9)))
    assert merged == recs(("2026-09-03", 9))


def test_merge_with_nothing_new_keeps_everything():
    old = recs(("2026-09-03", 1))
    assert fetch.merge_by_date(old, []) == old


@pytest.fixture
def summary_file(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    monkeypatch.setattr(fetch.C, "RADAR_SUMMARY_FILE", path)
    return path


def test_existing_summary_is_reused_when_the_baseline_matches(summary_file):
    summary_file.write_text(json.dumps({"method": {"baseline": "seasonal"}, "passes": recs(("2026-09-03", 1))}))
    assert len(fetch.load_existing_summary("seasonal")["passes"]) == 1


def test_different_baseline_starts_a_new_archive(summary_file):
    summary_file.write_text(json.dumps({"method": {"baseline": "dry"}, "passes": recs(("2026-09-03", 1))}))
    assert fetch.load_existing_summary("seasonal") == {}


def test_no_summary_yet(summary_file):
    assert fetch.load_existing_summary("seasonal") == {}


def test_weather_merges_with_older_days_and_survives_a_failed_download(monkeypatch):
    old = {"rainfall": [{"date": "2026-06-01", "mm": 4.0}],
           "river": [{"date": "2026-06-01", "m3s": 900.0, "normal_m3s": 1000.0}]}
    monkeypatch.setattr(fetch, "fetch_rainfall", lambda start: [{"date": "2026-10-01", "mm": 7.0}])
    monkeypatch.setattr(fetch, "fetch_river_discharge", lambda start: [])  # e.g. the API was down
    weather = fetch.fetch_weather("2026-09-01", old)
    assert [r["date"] for r in weather["rainfall"]] == ["2026-06-01", "2026-10-01"]
    assert weather["river"] == old["river"]


class FakeEE(types.SimpleNamespace):
    """Records how init_earth_engine logs in."""

    def __init__(self):
        super().__init__(calls=[])

    def ServiceAccountCredentials(self, email, key_data=None):
        return ("credentials", email, json.loads(key_data)["project_id"])

    def Initialize(self, credentials=None, project=None):
        self.calls.append((credentials, project))


@pytest.fixture
def fake_ee(monkeypatch):
    ee = FakeEE()
    monkeypatch.setitem(sys.modules, "ee", ee)
    return ee


def test_service_account_login_when_the_key_is_set(fake_ee, monkeypatch):
    key = {"client_email": "ci@proj-123.iam.gserviceaccount.com", "project_id": "proj-123"}
    monkeypatch.setenv(fetch.KEY_ENV, json.dumps(key))
    fetch.init_earth_engine(None)
    assert fake_ee.calls == [(("credentials", key["client_email"], "proj-123"), "proj-123")]


def test_explicit_project_overrides_the_keys_project(fake_ee, monkeypatch):
    monkeypatch.setenv(fetch.KEY_ENV, json.dumps({"client_email": "a@b", "project_id": "from-key"}))
    fetch.init_earth_engine("explicit")
    assert fake_ee.calls[0][1] == "explicit"


def test_user_login_without_a_key(fake_ee, monkeypatch):
    monkeypatch.delenv(fetch.KEY_ENV, raising=False)
    fetch.init_earth_engine("my-project")
    assert fake_ee.calls == [(None, "my-project")]
