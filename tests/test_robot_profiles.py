import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_default_profile_is_configurable_and_complete():
    from app.robots.registry import load_robot_profile

    profile = load_robot_profile("panda")
    assert profile["id"] == "panda"
    assert len(profile["arm_joint_names"]) == len(profile["home_positions"])
    assert profile["end_effector_body"]
    assert profile["gripper_actuators"]


def test_training_registry_reads_profile_without_runtime_imports():
    from training.robots.registry import available_robots, load_robot_profile

    profile = load_robot_profile("panda")
    assert profile["id"] == "panda"
    assert "panda" in {robot["id"] for robot in available_robots()}
    import training.robots.registry as registry

    assert "app" not in registry.__dict__


def test_unknown_robot_profile_fails_with_actionable_message():
    from app.robots.registry import load_robot_profile

    with pytest.raises(ValueError, match="Unknown robot profile"):
        load_robot_profile("not_a_real_robot")


def test_template_is_not_listed_as_an_available_robot():
    from app.robots.registry import available_robots

    assert "custom_robot.template" not in {robot["id"] for robot in available_robots()}


def test_custom_profile_template_has_required_adapter_fields():
    template = json.loads(
        (ROOT / "configs/robots/custom_robot.template.json").read_text()
    )
    for key in (
        "asset_directory",
        "mjcf_file",
        "arm_joint_names",
        "home_positions",
        "end_effector_body",
        "gripper_actuators",
        "gripper_open_control",
        "gripper_closed_control",
        "task",
    ):
        assert key in template
