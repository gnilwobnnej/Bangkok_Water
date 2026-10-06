"""Run the whole Streamlit app headlessly on the real processed data and fail on any exception."""
from pathlib import Path

import pytest

from src.data_loader import missing_files

APP = Path(__file__).resolve().parent.parent / "app.py"

pytestmark = pytest.mark.skipif(bool(missing_files()), reason="data/processed not prepared")


def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=600).run()
    assert not at.exception, [e.message for e in at.exception]
    tab_labels = [t.label for t in at.tabs]
    assert "Current flood" in tab_labels
