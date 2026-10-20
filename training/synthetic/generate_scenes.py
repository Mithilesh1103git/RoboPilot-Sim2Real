"""Generate randomized scene-parameter examples; wire these into your MuJoCo reset hook."""

import argparse
import json
import random
from pathlib import Path


def sample(seed):
    r = random.Random(seed)
    return {
        "seed": seed,
        "object_xy": [round(r.uniform(-0.2, 0.2), 4), round(r.uniform(-0.2, 0.2), 4)],
        "object_yaw": round(r.uniform(-3.14159, 3.14159), 4),
        "friction_scale": round(r.uniform(0.7, 1.3), 3),
        "note": "Example metadata; apply values in your task environment.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="training/datasets/scenes.jsonl")
    a = p.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(a.count):
            f.write(json.dumps(sample(a.seed + i)) + "\n")
    print(f"Wrote {a.count} samples to {out.resolve()}")


if __name__ == "__main__":
    main()
