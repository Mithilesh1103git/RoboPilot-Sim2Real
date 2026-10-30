"""Scenario configuration and deterministic reset helpers."""

import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def default_scenario(robot_profile=None) -> dict:
    profile = robot_profile or {}
    task = profile.get("task", {})
    return {
        "name": "pick_and_place",
        "seed": 7,
        "instruction": f"Pick up {task.get('object_name','red_block')} and place it inside {task.get('destination_name','blue_tray')}.",
        "objects": [task.get("object_name", "red_block")],
        "destination": task.get("destination_name", "blue_tray"),
        "robot": profile.get("id", "panda"),
    }
