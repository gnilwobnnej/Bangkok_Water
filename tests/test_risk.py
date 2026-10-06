import numpy as np
import pytest

from src.risk import compute_factors, district_mean, district_risk, risk_index
from tests.conftest import HIGH, LOWLAND, POCKET


@pytest.fixture
def factors(area):
    return compute_factors(area)


@pytest.mark.parametrize("weights", [(0.5, 0.3, 0.2), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 2, 2)])
def test_index_stays_between_0_and_1(factors, weights):
    risk = risk_index(factors, *weights)
    assert risk.min() >= 0 and risk.max() <= 1


def test_zero_weights_give_zero_risk(factors):
    assert (risk_index(factors, 0, 0, 0) == 0).all()


def test_weights_are_normalised(factors):
    assert np.allclose(risk_index(factors, 1, 1, 1), risk_index(factors, 0.2, 0.2, 0.2))


def test_elevation_weight_ranks_low_ground_higher(factors):
    risk = risk_index(factors, 1, 0, 0)
    assert np.allclose(risk, factors.low_elevation)
    assert risk[:, list(POCKET)].min() > risk[:, list(HIGH)].max()


def test_water_weight_ranks_cells_near_the_river_higher(factors):
    risk = risk_index(factors, 0, 1, 0)
    assert risk[0, 0] == pytest.approx(1.0)  # on the river
    assert np.all(np.diff(risk[0]) < 0), "proximity should fall steadily away from the river"


def test_population_weight_ranks_dense_cells_higher(factors):
    risk = risk_index(factors, 0, 0, 1)
    assert risk[0, -1] == pytest.approx(1.0)  # most populated column
    assert risk[0, 0] == 0  # nobody lives on the river


def test_changing_weights_reorders_districts(area, factors):
    by_elevation = district_risk(area, risk_index(factors, 1, 0, 0))
    by_people = district_risk(area, risk_index(factors, 0, 0, 1))
    # The west is low and riverside; the east is where people live
    assert by_elevation.idxmax() == 1
    assert by_people.idxmax() == 2


def test_district_mean_matches_a_direct_average(area):
    values = np.random.default_rng(0).random(area.dem.shape)
    means = district_mean(area, values, "v")
    assert means.name == "v" and list(means.index) == [1, 2]
    for d in (1, 2):
        assert means[d] == pytest.approx(values[area.district_ids == d].mean())


def test_lowland_is_as_low_risk_as_the_pocket_on_elevation(factors):
    """Same height, same elevation factor: the risk index has no notion of connectivity."""
    assert np.allclose(factors.low_elevation[:, list(LOWLAND)], factors.low_elevation[0, POCKET[0]])
