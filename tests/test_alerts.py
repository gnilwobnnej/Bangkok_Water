import pandas as pd
import pytest

from src import config as C
from src.current_flood import conditions_alert

TODAY = "2026-10-05"
NO_RAIN = pd.DataFrame(columns=["date", "mm"])
NO_RIVER = pd.DataFrame(columns=["date", "m3s", "normal_m3s"])


def passes(*km2, last_date="2026-10-01"):
    """Passes 4 days apart ending on `last_date`, with these flooded areas."""
    end = pd.Timestamp(last_date)
    dates = [str((end - pd.Timedelta(days=4 * i)).date()) for i in range(len(km2))][::-1]
    return pd.DataFrame({"date": dates, "flooded_km2": list(km2)})


def rain(*mm, end=TODAY):
    dates = pd.date_range(end=end, periods=len(mm)).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "mm": list(mm)})


def river(pct, date=TODAY):
    return pd.DataFrame({"date": [date], "m3s": [pct * 10.0], "normal_m3s": [1000.0]})


def test_calm_conditions_give_no_alert():
    assert conditions_alert(passes(10, 12, 11, 13), rain(5, 5, 5), river(100)) == []


def test_flood_jump_alerts():
    alerts = conditions_alert(passes(10, 12, 11, 30), NO_RAIN, NO_RIVER, today=TODAY)
    assert len(alerts) == 1 and "30 km²" in alerts[0] and "2.7×" in alerts[0]


def test_flood_just_below_the_ratio_does_not_alert():
    typical = 10
    below = C.ALERT_FLOOD_RATIO * typical - 0.1
    assert conditions_alert(passes(10, 10, 10, below), NO_RAIN, NO_RIVER, today=TODAY) == []


def test_old_flood_pass_does_not_alert():
    old = passes(10, 12, 11, 30, last_date="2026-09-01")
    assert conditions_alert(old, NO_RAIN, NO_RIVER, today=TODAY) == []


def test_too_few_passes_to_judge():
    few = passes(*[10] * C.ALERT_MIN_EARLIER_PASSES)[:-1]
    few = pd.concat([few, passes(50)], ignore_index=True)  # big latest pass, but too little history
    assert conditions_alert(few, NO_RAIN, NO_RIVER, today=TODAY) == []


def test_pass_order_does_not_matter():
    shuffled = passes(10, 12, 11, 30).sample(frac=1, random_state=1)
    assert len(conditions_alert(shuffled, NO_RAIN, NO_RIVER, today=TODAY)) == 1


@pytest.mark.parametrize("pct,expected", [(C.ALERT_RIVER_PCT, 1), (C.ALERT_RIVER_PCT - 1, 0), (150, 1)])
def test_river_threshold(pct, expected):
    alerts = conditions_alert(passes(), NO_RAIN, river(pct))
    assert len(alerts) == expected
    if expected:
        assert f"{pct}% of normal" in alerts[0]


def test_river_uses_the_latest_day():
    flows = pd.concat([river(150, "2026-10-01"), river(100, "2026-10-04")], ignore_index=True)
    assert conditions_alert(passes(), NO_RAIN, flows) == []


def test_heavy_rain_alerts():
    alerts = conditions_alert(passes(), rain(0, 0, 60, 30, 20), NO_RIVER)
    assert len(alerts) == 1 and "110 mm" in alerts[0]


def test_rain_outside_the_3_day_window_is_ignored():
    assert conditions_alert(passes(), rain(200, 0, 0, 0, 0), NO_RIVER) == []


def test_missing_rain_day_means_no_rain_alert():
    gap = rain(80, 80, 80).drop(index=1)
    assert conditions_alert(passes(), gap, NO_RIVER) == []


def test_several_alerts_at_once():
    alerts = conditions_alert(passes(10, 12, 11, 30), rain(50, 50, 50), river(140))
    assert len(alerts) == 3


def test_no_data_at_all():
    assert conditions_alert(passes(), NO_RAIN, NO_RIVER) == []
