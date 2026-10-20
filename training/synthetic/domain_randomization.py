"""Generate reproducible MuJoCo scene-parameter manifests.

This writes sampled parameters; the task environment must explicitly apply them
during reset. It does not silently mutate an MJCF model.
"""

import argparse
import json
import random
from pathlib import Path


def sample_scene(
    seed,
    xy_range=(-0.2, 0.2),
    friction_range=(0.7, 1.3),
    mass_range=(0.8, 1.2),
    damping_range=(0.8, 1.2),
):
    r = random.Random(seed)
    return {
        "seed": seed,
        "object_xy": [r.uniform(*xy_range), r.uniform(*xy_range)],
        "object_yaw": r.uniform(-3.14159265, 3.14159265),
        "friction_scale": r.uniform(*friction_range),
        "mass_scale": r.uniform(*mass_range),
        "damping_scale": r.uniform(*damping_range),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="training/datasets/scenes.jsonl")
    a = p.parse_args()
    if a.count < 1:
        p.error("--count must be >= 1")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(a.count):
            f.write(json.dumps(sample_scene(a.seed + i)) + "\n")
    print(f"Wrote {a.count} scene manifests to {out.resolve()}")


if __name__ == "__main__":
    main()
