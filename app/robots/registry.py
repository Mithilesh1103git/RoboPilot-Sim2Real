"""Runtime robot profile loading."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import PROJECT_ROOT

PROFILE_DIR = PROJECT_ROOT / "configs" / "robots"


def available_robots():
    return [
        {"id": p.stem, "display_name": _load(p.stem).get("display_name", p.stem)}
        for p in sorted(PROFILE_DIR.glob("*.json"))
        if not p.name.endswith(".template.json")
    ]


def _load(name):
    if not name or Path(name).name != name:
        raise ValueError("Robot name must be a profile ID, not a path.")
    path = PROFILE_DIR / f"{name.lower()}.json"
    if not path.is_file():
        raise ValueError(
            f"Unknown robot profile {name!r}. Add {path.relative_to(PROJECT_ROOT)}."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("id", name.lower()) != name.lower():
        raise ValueError(f"Profile id must match its filename stem {name!r}.")
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
        raise ValueError(f"Robot profile {name!r} missing fields: {', '.join(missing)}")
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


def load_robot_profile(name=None):
    from app.config import settings

    return _load((name or settings.robot_name).lower())
