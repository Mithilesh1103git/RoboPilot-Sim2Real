import argparse
from pathlib import Path
import numpy as np

REQUIRED = {
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
    "episode_ids",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="training/datasets/export.npz")
    a = p.parse_args()
    with np.load(a.input) as d:
        missing = REQUIRED - set(d.files)
        if missing:
            raise ValueError(f"Missing keys: {sorted(missing)}")
        data = {k: d[k].copy() for k in d.files}
    n = len(data["rewards"])
    if any(len(data[k]) != n for k in REQUIRED):
        raise ValueError("Transition array lengths differ")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"Exported {n} transitions to {out.resolve()}")


if __name__ == "__main__":
    main()
