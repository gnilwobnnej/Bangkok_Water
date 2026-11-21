"""Open-Meteo retries in scripts/build_traffy_labels.py (no network calls)."""
import sys
from pathlib import Path

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
