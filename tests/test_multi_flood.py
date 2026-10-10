"""Wet-season labels (scripts/build_flood_archive.py) and multi-flood training helpers (scripts/train_model.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_flood_archive as archive  # noqa: E402
import train_model as tm  # noqa: E402
from src import config as C  # noqa: E402

U = C.LABEL_NODATA


def test_season_labels():
    # cells:          never seen | dry every time | dry once, partly flooded once | flooded once | flooded twice
    n_seen = np.array([0, 5, 2, 4, 6])
    n_flooded = np.array([0, 0, 0, 1, 2])
    n_dry = np.array([0, 5, 1, 3, 4])
    assert list(archive.season_labels(n_seen, n_flooded, n_dry)) == [U, 0, U, U, 1]
    # a single flooded pass is enough when asked
    assert list(archive.season_labels(n_seen, n_flooded, n_dry, min_flood_passes=1)) == [U, 0, U, 1, 1]


def test_archive_refuses_test_years():
    with pytest.raises(SystemExit, match="test years"):
        archive.main(None, [2025, C.TEST_YEARS_FROM], refresh=False, min_flood_passes=2)


def test_event_info_names_radar_seasons_and_2011():
    assert tm.event_info(Path("x/radar_2019.tif"))["name"] == "2019 wet season (radar)"
    assert tm.event_info(Path("x/radar_2019.tif"))["year"] == 2019
    assert tm.event_info(C.FLOOD_2011_FILE)["id"] == "2011"


def test_load_events_refuses_test_years(tmp_path):
    with pytest.raises(SystemExit, match="test year"):
        tm.load_events([tmp_path / f"radar_{C.TEST_YEARS_FROM}.tif"])


def event(labels):
    return {"labels": np.asarray(labels, dtype="uint8")}


def test_training_set_samples_each_event_equally():
    big = event([1, 0] * 50)           # 100 labelled cells
    small = event([1, 0, U, U] * 5)    # 10 labelled cells
    cells, y, w = tm.training_set([big, small], cap=20)
    assert len(cells) == 30            # 20 sampled from the big event, all 10 of the small one
    w_big, w_small = w[:20], w[20:]
    assert w_big.sum() == pytest.approx(w_small.sum())  # each event carries the same total weight
    assert w.mean() == pytest.approx(1.0)
    assert set(small["labels"][cells[20:]]) == {0, 1}   # no unknown cells
    assert np.array_equal(y[20:], small["labels"][cells[20:]])


def test_training_set_respects_the_cv_mask():
    ev = event([1, 0, 1, 0, 1, 0])
    ok = np.array([True, True, False, False, True, True])
    cells, _, _ = tm.training_set([ev], cell_ok=ok)
    assert set(cells) == {0, 1, 4, 5}


def test_block_folds_keep_each_block_in_one_fold():
    blocks = np.repeat(np.arange(20), 7)
    fold = tm.block_folds(blocks)
    assert set(fold) == set(range(1, tm.N_FOLDS + 1))
    for b in range(20):
        assert len(set(fold[blocks == b])) == 1
