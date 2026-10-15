import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="training/datasets/augmented.npz")
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    with np.load(a.input) as d:
        data = {k: d[k].copy() for k in d.files}
    for k in ("observations", "next_observations"):
        if k in data and a.noise_std:
            data[k] += rng.normal(0, a.noise_std, data[k].shape).astype(np.float32)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(out.resolve())


if __name__ == "__main__":
    main()
