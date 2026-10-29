"""Summarize a transition dataset; callable API plus CLI."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_dataset(input, output="training/datasets/summary.json"):
    with np.load(input) as archive:
        summary = {
            key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
            for key in archive.files
        }
        if "rewards" in archive and archive["rewards"].size:
            summary.update(
                reward_mean=float(np.mean(archive["rewards"])),
                reward_min=float(np.min(archive["rewards"])),
                reward_max=float(np.max(archive["rewards"])),
            )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"output": str(out), "summary": summary}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="training/datasets/summary.json")
    print(summarize_dataset(**vars(p.parse_args(argv)))["output"])


if __name__ == "__main__":
    main()
