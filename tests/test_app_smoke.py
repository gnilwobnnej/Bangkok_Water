"""Run the whole Streamlit app headlessly on the real processed data and fail on any exception."""
import json
import os
from pathlib import Path

import pytest

from src import config as C
from src.data_loader import missing_files

APP = Path(__file__).resolve().parent.parent / "app.py"

pytestmark = pytest.mark.skipif(bool(missing_files()), reason="data/processed not prepared")


def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=600).run()
    assert not at.exception, [e.message for e in at.exception]
    tab_labels = [t.label for t in at.tabs]
    assert "Current flood" in tab_labels


@pytest.mark.skipif(not C.RADAR_SUMMARY_FILE.exists(), reason="no radar summary")
def test_updated_radar_data_shows_without_a_restart(tmp_path, monkeypatch):
    """Streamlit Cloud reruns pushed code in the same process, so cached data must not outlive a data update."""
    from streamlit.testing.v1 import AppTest

    def pass_dates(at):
        return next(s for s in at.selectbox if s.label == "Radar pass date").options

    full = json.loads(C.RADAR_SUMMARY_FILE.read_text())
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({**full, "passes": full["passes"][:3]}))
    monkeypatch.setattr(C, "RADAR_SUMMARY_FILE", summary)

    first = AppTest.from_file(str(APP), default_timeout=600).run()
    assert not first.exception and len(pass_dates(first)) == 3

    summary.write_text(json.dumps(full))  # "new passes arrive"
    stat = summary.stat()
    os.utime(summary, (stat.st_atime, stat.st_mtime + 10))  # ensure a new mtime even on coarse filesystems
    second = AppTest.from_file(str(APP), default_timeout=600).run()
    assert not second.exception, [e.message for e in second.exception]
    assert len(pass_dates(second)) == len(full["passes"])
