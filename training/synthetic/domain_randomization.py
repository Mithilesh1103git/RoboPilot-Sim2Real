"""Generate reproducible scene-parameter manifests."""

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
    rng = random.Random(seed)
    return {
        "seed": seed,
        "object_xy": [rng.uniform(*xy_range), rng.uniform(*xy_range)],
        "object_yaw": rng.uniform(-3.14159265, 3.14159265),
        "friction_scale": rng.uniform(*friction_range),
        "mass_scale": rng.uniform(*mass_range),
        "damping_scale": rng.uniform(*damping_range),
    }


def generate_scene_manifest(count=100, seed=7, output="training/datasets/scenes.jsonl"):
    if count < 1:
        raise ValueError("count must be >= 1")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scenes = [sample_scene(seed + i) for i in range(count)]
    with out.open("w", encoding="utf-8") as f:
        for scene in scenes:
            f.write(json.dumps(scene) + "\n")
    return {"output": str(out), "count": count, "scenes": scenes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="training/datasets/scenes.jsonl")
    result = generate_scene_manifest(**vars(p.parse_args(argv)))
    print(
        f"Wrote {result['count']} scene manifests to {Path(result['output']).resolve()}"
    )


if __name__ == "__main__":
    main()
