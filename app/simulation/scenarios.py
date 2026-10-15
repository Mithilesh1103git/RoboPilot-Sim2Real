"""Scenario configuration and deterministic reset helpers."""

import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def default_scenario() -> dict:
    return {
        "name": "default",
        "seed": 7,
        "instruction": "Pick up the red block and place it inside the blue tray.",
        "objects": ["red_block"],
        "destination": "blue_tray",
    }
