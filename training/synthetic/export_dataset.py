"""Validate and export transition datasets; callable API plus CLI."""

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


def export_dataset(input, output="training/datasets/export.npz"):
    with np.load(input) as archive:
        missing = REQUIRED - set(archive.files)
        if missing:
            raise ValueError(f"Missing keys: {sorted(missing)}")
        data = {key: archive[key].copy() for key in archive.files}
    count = len(data["rewards"])
    if any(len(data[key]) != count for key in REQUIRED):
        raise ValueError("Transition array lengths differ")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return {"output": str(out), "transitions": count, "arrays": sorted(data)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="training/datasets/export.npz")
    print(export_dataset(**vars(p.parse_args(argv))))


if __name__ == "__main__":
    main()
