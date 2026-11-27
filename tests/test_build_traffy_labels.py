"""Open-Meteo retries and kept rain in scripts/build_traffy_labels.py (no network calls)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_traffy_labels as btl  # noqa: E402


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self.body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self.body


@pytest.fixture
def fake_get(monkeypatch):
    """Serve the queued answers (a FakeResponse or an exception) in turn; record the waits instead of sleeping."""
    queue, waits = [], []

    def get(*args, **kwargs):
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(btl.requests, "get", get)
    monkeypatch.setattr(btl.time, "sleep", waits.append)
    return queue, waits


def test_retries_rate_limit_then_returns_json(fake_get):
    queue, waits = fake_get
    queue += [FakeResponse(429), FakeResponse(503), requests.ConnectionError("reset"), FakeResponse(200, {"ok": 1})]
    assert btl.get_archive({}, retries=(1, 2, 3)) == {"ok": 1}
    assert waits == [1, 2, 3]


def test_gives_up_after_last_retry(fake_get):
    queue, waits = fake_get
    queue += [FakeResponse(429)] * 3
    with pytest.raises(RuntimeError, match="HTTP 429 after 3 tries"):
        btl.get_archive({}, retries=(1, 2))
    assert waits == [1, 2]


def test_other_client_errors_are_not_retried(fake_get):
    queue, waits = fake_get
    queue += [FakeResponse(400)]
    with pytest.raises(requests.HTTPError):
        btl.get_archive({}, retries=(1, 2))
    assert waits == []


@pytest.fixture
def fake_download(monkeypatch):
    """download_rain() answers 1 mm a day (2 mm from 2026-11-01, as if the archive revised them); records calls."""
    calls = []

    def download(start, end):
        calls.append((start, end))
        days = pd.date_range(start, end)
        return pd.Series(np.where(days >= "2026-11-01", 2.0, 1.0), index=days)

    monkeypatch.setattr(btl, "download_rain", download)
    monkeypatch.setattr(btl.C, "TRAFFY_RAIN_REFETCH_DAYS", 5)
    return calls


def test_city_rain_downloads_everything_without_kept_file(tmp_path, fake_download):
    path = tmp_path / "city_rain.csv"
    rain = btl.city_rain("2026-10-01", "2026-10-20", path)
    assert fake_download == [("2026-10-01", "2026-10-20")]
    assert len(rain) == 20 and path.exists()
    assert btl.read_kept_rain(path).equals(rain)


def test_city_rain_refetches_only_recent_days(tmp_path, fake_download):
    path = tmp_path / "city_rain.csv"
    btl.save_kept_rain(pd.Series(1.0, index=pd.date_range("2026-10-01", "2026-10-31")), path)
    rain = btl.city_rain("2026-10-01", "2026-11-05", path)
    assert fake_download == [("2026-10-26", "2026-11-05")]
    assert rain.index.min() == pd.Timestamp("2026-10-01") and rain.index.max() == pd.Timestamp("2026-11-05")
    assert (rain[:"2026-10-31"] == 1.0).all() and (rain["2026-11-01":] == 2.0).all()


def test_city_rain_keeps_old_value_where_download_is_missing(tmp_path, monkeypatch):
    path = tmp_path / "city_rain.csv"
    btl.save_kept_rain(pd.Series([3.0, 4.0], index=pd.date_range("2026-10-01", periods=2)), path)
    monkeypatch.setattr(btl, "download_rain",
                        lambda s, e: pd.Series([np.nan, 5.0], index=pd.date_range(s, periods=2)))
    rain = btl.city_rain("2026-10-01", "2026-10-02", path)
    assert rain.tolist() == [3.0, 5.0]


def test_city_rain_redownloads_when_kept_file_starts_late(tmp_path, fake_download):
    path = tmp_path / "city_rain.csv"
    btl.save_kept_rain(pd.Series(1.0, index=pd.date_range("2026-10-10", "2026-10-20")), path)
    btl.city_rain("2026-10-01", "2026-10-20", path)
    assert fake_download == [("2026-10-01", "2026-10-20")]
