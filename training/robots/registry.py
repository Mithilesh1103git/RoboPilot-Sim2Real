"""Training-owned robot profile reader; intentionally does not import app/."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = PROJECT_ROOT / "configs" / "robots"


def available_robots():
    return [
        {
            "id": p.stem,
            "display_name": json.loads(p.read_text(encoding="utf-8")).get(
                "display_name", p.stem
            ),
        }
        for p in sorted(PROFILE_DIR.glob("*.json"))
        if not p.name.endswith(".template.json")
    ]


def load_robot_profile(name=None):
    robot_id = (name or os.getenv("ROBOT_NAME", "panda")).lower()
    if not robot_id or Path(robot_id).name != robot_id:
        raise ValueError("Robot name must be a profile ID, not a path.")
    path = PROFILE_DIR / f"{robot_id}.json"
    if not path.is_file():
        raise ValueError(
            f"Unknown robot profile {robot_id!r}; add configs/robots/{robot_id}.json."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("id", robot_id) != robot_id:
        raise ValueError(f"Profile id must match its filename stem {robot_id!r}.")
    required = (
        "asset_directory",
        "mjcf_file",
        "arm_joint_names",
        "home_positions",
        "end_effector_body",
        "gripper_actuators",
        "gripper_open_control",
        "gripper_closed_control",
        "task",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Robot profile missing fields: {', '.join(missing)}")
    if Path(data["asset_directory"]).name != data["asset_directory"]:
        raise ValueError("asset_directory must be a single directory name.")
    if not data["arm_joint_names"] or len(data["arm_joint_names"]) != len(
        data["home_positions"]
    ):
        raise ValueError(
            "arm_joint_names and home_positions must have equal nonzero lengths."
        )
    if not data["gripper_actuators"]:
        raise ValueError(
            "Pick-and-place profiles require at least one configured gripper actuator."
        )
    if len(data["gripper_actuators"]) != len(data["gripper_open_control"]) or len(
        data["gripper_actuators"]
    ) != len(data["gripper_closed_control"]):
        raise ValueError(
            "Gripper actuator names and control arrays must have equal lengths."
        )
    if data.get("control_mode", "position") != "position":
        raise ValueError(
            "Built-in controller supports position-controlled arm actuators only; add a custom controller adapter for other modes."
        )
    if not isinstance(data.get("task"), dict):
        raise ValueError("Profile task must be a JSON object.")
    return data
