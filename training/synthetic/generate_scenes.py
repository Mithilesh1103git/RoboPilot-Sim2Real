"""Generate reproducible scene-parameter examples; callable API plus CLI."""

import argparse
import json
import random
from pathlib import Path


def sample(seed):
    rng = random.Random(seed)
    return {
        "seed": seed,
        "object_xy": [
            round(rng.uniform(-0.2, 0.2), 4),
            round(rng.uniform(-0.2, 0.2), 4),
        ],
        "object_yaw": round(rng.uniform(-3.14159, 3.14159), 4),
        "friction_scale": round(rng.uniform(0.7, 1.3), 3),
        "note": "Example metadata; apply values in your task environment.",
    }


def generate_scenes(count=100, seed=7, output="training/datasets/scenes.jsonl"):
    if count < 1:
        raise ValueError("count must be >= 1")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scenes = [sample(seed + i) for i in range(count)]
    with out.open("w", encoding="utf-8") as f:
        for scene in scenes:
            f.write(json.dumps(scene) + "\n")
    return {"output": str(out), "count": count, "scenes": scenes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="training/datasets/scenes.jsonl")
    result = generate_scenes(**vars(p.parse_args(argv)))
    print(
        f"Wrote {result['count']} scene manifests to {Path(result['output']).resolve()}"
    )


if __name__ == "__main__":
    main()
