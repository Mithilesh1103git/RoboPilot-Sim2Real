"""Callable and CLI validation for a configured training robot profile."""

import argparse
from pathlib import Path

from training.robots.registry import PROJECT_ROOT, load_robot_profile


def validate_robot_profile(robot_name="panda"):
    profile = load_robot_profile(robot_name)
    robot_dir = (
        PROJECT_ROOT / "training" / "assets" / "robots" / profile["asset_directory"]
    )
    model = robot_dir / profile["mjcf_file"]
    if not model.is_file():
        raise FileNotFoundError(
            f"Training-owned robot MJCF missing: {model}. Copy the robot model/assets into this directory."
        )
    for key in ("object_name", "destination_name"):
        if not profile.get("task", {}).get(key):
            raise ValueError(f"Robot profile task is missing {key}.")
    return {
        "robot_name": robot_name,
        "display_name": profile.get("display_name", robot_name),
        "profile": "configs/robots/" + robot_name + ".json",
        "training_mjcf": str(model),
        "arm_joints": len(profile["arm_joint_names"]),
        "gripper_actuators": len(profile["gripper_actuators"]),
        "control_mode": profile.get("control_mode", "position"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default="panda")
    print(validate_robot_profile(parser.parse_args(argv).robot_name))


if __name__ == "__main__":
    main()
