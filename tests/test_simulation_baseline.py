from pathlib import Path

import numpy as np
import pytest

from app.agent.executor import TaskExecutor
from app.control.motion import inverse_kinematics
from app.simulation.environment import SimulationEnvironment
from app.simulation.scenarios import resolve_scenario

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
PANDA_MODEL = ASSET_ROOT / "franka_emika_panda" / "panda.xml"

pytestmark = pytest.mark.skipif(
    not PANDA_MODEL.is_file(),
    reason="Run `python scripts/fetch_panda_assets.py` to install Panda assets.",
)


def test_deterministic_pick_and_place_baseline():
    environment = SimulationEnvironment(
        asset_root=ASSET_ROOT,
        robot_name="panda",
        headless=True,
    )
    executor = TaskExecutor(environment, planner=None, max_retries=0)
    try:
        result = executor.run(
            "Pick up the red block and place it inside the blue tray.",
            scenario="default",
        )
    finally:
        environment.close()

    assert result["success"] is True
    assert result["state"] == "succeeded"
    assert result["attempts"] == 1
    assert result["result"]["trace"][-1]["action"] == "verify"
    assert result["result"]["trace"][-1]["success"] is True


def test_seeded_randomized_pick_and_place_baseline():
    environment = SimulationEnvironment(
        asset_root=ASSET_ROOT,
        robot_name="panda",
        headless=True,
    )
    executor = TaskExecutor(environment, planner=None, max_retries=0)
    try:
        result = executor.run(
            "Pick up the red block and place it inside the blue tray.",
            scenario="randomized",
            seed=23,
        )
    finally:
        environment.close()

    expected = resolve_scenario("randomized", 23)
    assert result["success"] is True
    assert result["scenario_config"] == expected.to_dict()
    assert result["result"]["object_start"] == pytest.approx(expected.object_position)
    assert result["result"]["destination_position"] == pytest.approx(
        expected.destination_position
    )


def test_unreachable_ik_fails_without_mutating_simulation_state():
    environment = SimulationEnvironment(
        asset_root=ASSET_ROOT,
        robot_name="panda",
        headless=True,
    )
    try:
        environment.load()
        environment.reset()
        qpos_before = environment.data.qpos.copy()
        qvel_before = environment.data.qvel.copy()
        ctrl_before = environment.data.ctrl.copy()

        with pytest.raises(ValueError, match="IK did not converge"):
            inverse_kinematics(
                environment,
                [5.0, 5.0, 5.0],
                max_iterations=5,
            )

        np.testing.assert_allclose(environment.data.qpos, qpos_before)
        np.testing.assert_allclose(environment.data.qvel, qvel_before)
        np.testing.assert_allclose(environment.data.ctrl, ctrl_before)
    finally:
        environment.close()
