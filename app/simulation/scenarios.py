"""Reproducible tabletop scenario definitions."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    object_position: tuple[float, float, float]
    destination_position: tuple[float, float, float]
    instruction: str = "Pick up the red block and place it inside the blue tray."
    object_name: str = "red_block"
    destination_name: str = "blue_tray"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_scenario(name: str = "default", seed: int = 7) -> Scenario:
    normalized = name.strip().lower()
    if normalized in {"default", "panda_pick_place", "pick_and_place"}:
        return Scenario(
            name="default",
            seed=seed,
            object_position=(0.48, -0.20, 0.425),
            destination_position=(0.57, 0.22, 0.405),
        )
    if normalized == "randomized":
        rng = random.Random(seed)
        return Scenario(
            name="randomized",
            seed=seed,
            object_position=(
                rng.uniform(0.46, 0.50),
                rng.uniform(-0.22, -0.18),
                0.425,
            ),
            destination_position=(
                rng.uniform(0.55, 0.59),
                rng.uniform(0.20, 0.24),
                0.405,
            ),
        )
    raise ValueError(f"Unsupported scenario: {name}")


def default_scenario() -> dict:
    return resolve_scenario().to_dict()
