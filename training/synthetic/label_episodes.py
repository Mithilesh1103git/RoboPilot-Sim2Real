"""Create descriptive episode summaries from recorded simulator transitions.

This does not invent success labels: success remains unknown unless a separate,
explicit success signal is present in the source dataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def label_episodes(input, output="training/datasets/episode_labels.json"):
    with np.load(input) as archive:
        if "episode_ids" not in archive or "rewards" not in archive:
            raise ValueError("Dataset must include episode_ids and rewards")
        labels = []
        for eid in np.unique(archive["episode_ids"]):
            mask = archive["episode_ids"] == eid
            labels.append(
                {
                    "episode_id": int(eid),
                    "steps": int(mask.sum()),
                    "return": float(archive["rewards"][mask].sum()),
                    "terminated": (
                        bool(archive["terminated"][mask][-1])
                        if "terminated" in archive
                        else None
                    ),
                    "success": None,
                }
            )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return {"output": str(out), "episodes": len(labels), "labels": labels}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="training/datasets/episode_labels.json")
    print(label_episodes(**vars(p.parse_args(argv)))["output"])


if __name__ == "__main__":
    main()
