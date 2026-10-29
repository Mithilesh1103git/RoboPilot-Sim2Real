"""Augment numeric transition datasets; callable API plus CLI."""

import argparse
from pathlib import Path

import numpy as np


def augment_dataset(
    input, output="training/datasets/augmented.npz", noise_std=0.01, seed=7
):
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")
    rng = np.random.default_rng(seed)
    with np.load(input) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    for key in ("observations", "next_observations"):
        if key in data and noise_std:
            data[key] += rng.normal(0, noise_std, data[key].shape).astype(np.float32)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return {
        "output": str(out),
        "arrays": sorted(data),
        "transitions": int(len(data.get("rewards", []))),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="training/datasets/augmented.npz")
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    print(augment_dataset(**vars(p.parse_args(argv))))


if __name__ == "__main__":
    main()
