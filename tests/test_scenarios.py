import pytest

from app.simulation.scenarios import resolve_scenario


def test_randomized_scenario_is_seeded_and_bounded():
    first = resolve_scenario("randomized", 41)
    repeated = resolve_scenario("randomized", 41)
    different = resolve_scenario("randomized", 42)

    assert first == repeated
    assert first != different
    assert 0.46 <= first.object_position[0] <= 0.50
    assert -0.22 <= first.object_position[1] <= -0.18
    assert 0.55 <= first.destination_position[0] <= 0.59
    assert 0.20 <= first.destination_position[1] <= 0.24


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="Unsupported scenario"):
        resolve_scenario("warehouse", 7)
