"""Elevation uncertainty (src/uncertainty.py, scripts/dem_uncertainty.py) and the event context script."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import dem_uncertainty as du  # noqa: E402
import event_context as ec  # noqa: E402
from src.uncertainty import (  # noqa: E402
    DemUncertainty, correlated_noise, mode_key, nearest_prob_level,
)
from src.viz import level_curve_chart  # noqa: E402


def test_noise_has_the_requested_spread_and_correlation():
    rng = np.random.default_rng(0)
    cell = 50.0
    corr_cells = 6  # 300 m
    corrs = []
    for _ in range(5):
        f = correlated_noise((300, 300), (cell, cell), sigma_m=0.7, corr_m=corr_cells * cell, rng=rng)
        assert f.std() == pytest.approx(0.7, rel=1e-4)
        assert abs(f.mean()) < 0.1
        corrs.append([np.corrcoef(f[:, :-k].ravel(), f[:, k:].ravel())[0, 1] for k in (1, corr_cells, 4 * corr_cells)])
    neighbour, at_corr, far = np.mean(corrs, axis=0)
    assert neighbour > 0.9                       # neighbouring cells are wrong together
    assert at_corr == pytest.approx(np.exp(-1), abs=0.08)  # 1/e at the correlation distance
    assert abs(far) < 0.1                         # and independent far apart


def test_mode_keys_and_nearest_level():
    assert mode_key(True, False) == "river"
    assert mode_key(True, True) == "river_defences"
    assert mode_key(False, True) == "anywhere"  # defences only apply with connectivity
    assert nearest_prob_level(0.1) == 0.5
    assert nearest_prob_level(1.2) == 1.0
    assert nearest_prob_level(1.3) == 1.5
    assert nearest_prob_level(5.0) == 3.0


def test_run_stats_match_the_simulation(area):
    from src.simulate import level_curve, simulate
    district_area = np.bincount(area.district_ids.ravel(), weights=area.cell_area_km2.ravel())
    onset = du.onset_levels(area, connected=True, defences=False)
    stats = du.run_stats(area, onset, district_area)
    curve = level_curve(area, du.LEVELS, connected=True)
    assert stats[:, 0] == pytest.approx(curve["area_km2"].to_numpy(), rel=1e-5)
    assert stats[:, 1] == pytest.approx(curve["people"].to_numpy(), rel=1e-5)
    for i, lv in enumerate(du.LEVELS):
        res = simulate(area, float(lv))
        assert stats[i, 2] == (res.by_district["pct_flooded"] > 25).sum()
    # sealed pocket: rain-ponding mode floods it at 0.5 m, river mode doesn't
    i05 = int(np.flatnonzero(np.isclose(du.LEVELS, 0.5))[0])
    anywhere = du.run_stats(area, du.onset_levels(area, False, False), district_area)
    assert anywhere[i05, 0] > stats[i05, 0]


def test_uncertainty_lookup_and_curve_band():
    rows = [{"level": lv, **{f"{q}_p{p}": lv * k for q, k in [("area_km2", 100), ("people", 1e5),
                                                              ("districts_25", 3)] for p in (5, 50, 95)}}
            for lv in du.LEVELS]
    u = DemUncertainty(meta={}, curves={"river": pd.DataFrame(rows)})
    assert u.at(True, False, 1.04)["level"] == pytest.approx(1.0)
    assert u.at(True, True, 1.0) is None
    curve = pd.DataFrame({"level": du.LEVELS, "area_km2": du.LEVELS * 100, "people": du.LEVELS * 1e5})
    fig = level_curve_chart(curve, 1.0, u.curve(True, False))
    assert len(fig.data) == 5  # two bands, two lines, hover markers
    assert len(level_curve_chart(curve, 1.0).data) == 2


def test_year_context():
    days = pd.date_range("2020-01-01", "2020-12-31")
    rain = pd.Series(0.0, index=days)
    rain["2020-09-10":"2020-09-12"] = 50.0
    rain["2020-12-01"] = 500.0  # outside August-November: ignored
    river = pd.Series(1000.0, index=days)
    river["2020-10-01":"2020-10-05"] = 3000.0
    normal = pd.Series(1000.0, index=days.strftime("%m-%d"))
    c = ec.year_context(2020, rain, river, normal, rain_normal=300.0)
    assert c["rain_total_mm"] == 150 and c["rain_pct_normal"] == 50 and c["rain_max_3day_mm"] == 150
    assert c["river_peak_m3s"] == 3000 and c["river_peak_date"] == "2020-10-01"
    assert c["river_peak_pct_normal"] == 300 and c["river_days_above_alert"] == 5
