"""The double-bounce street-flood test (scripts/double_bounce.py): choosing passes and gridding the classes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import double_bounce as db  # noqa: E402
from src import config as C  # noqa: E402


def passes(*local_times):
    local = pd.to_datetime(list(local_times))
    return pd.DataFrame({"id": local.strftime("%Y-%m-%dT%H"), "utc": local - pd.Timedelta(hours=7), "local": local,
                         "direction": "ASCENDING", "orbit": 172})


def test_reports_near_counts_both_sides_of_the_pass():
    times = pd.Series(pd.to_datetime(["2026-09-27 11:00", "2026-09-27 13:00", "2026-09-27 23:30",
                                      "2026-09-28 01:00"]))
    assert list(db.reports_near(times, pd.Timestamp("2026-09-27 18:00"))) == [False, True, True, False]


def test_wet_passes_need_a_storm_and_reports_and_controls_are_quiet_dry_season_passes():
    p = passes("2026-09-27 18:00", "2026-09-15 18:00", "2026-08-01 18:00", "2026-02-03 06:00", "2026-02-15 06:00")
    times = pd.Series(pd.to_datetime(["2026-09-27 17:00"] * 30 + ["2026-08-01 17:00"] * 30
                                     + ["2026-02-15 07:00"] * 5))
    storms = {pd.Timestamp("2026-09-27"), pd.Timestamp("2026-09-14")}  # 09-15 is the day after a storm
    chosen = db.choose_passes(p, times, storms)
    kinds = dict(zip(chosen["id"], chosen["kind"]))
    assert kinds == {"2026-09-27T18": "wet", "2026-02-03T06": "control"}  # 09-15 had no reports, 08-01 no storm


def test_to_grid_is_the_bright_share_of_seen_built_up_ground(tmp_path):
    # one analysis cell (0.0006 deg) as 3 x 3 source pixels: 2 bright, 4 not, 3 not built-up
    codes = np.full((3, 3), db.NA, dtype="uint8")
    codes.flat[:6] = [db.BRIGHT, db.BRIGHT, db.NOT_BRIGHT, db.NOT_BRIGHT, db.NOT_BRIGHT, db.NOT_BRIGHT]
    path = tmp_path / "pass.tif"
    with rasterio.open(path, "w", driver="GTiff", height=3, width=3, count=1, dtype="uint8", crs=C.CRS,
                       transform=from_origin(C.WEST, C.NORTH, C.RES / 3, C.RES / 3)) as dst:
        dst.write(codes, 1)
    bright, seen = db.to_grid(path)
    assert np.isclose(seen[0, 0], 600 / 9, atol=0.5) and np.isclose(bright[0, 0], 100 / 3, atol=0.5)
    assert np.isnan(bright[5, 5]) and seen[5, 5] == 0
