import argparse, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="training/datasets/episode_labels.json")
    a = p.parse_args()
    with np.load(a.input) as d:
        labels = []
        for eid in np.unique(d["episode_ids"]):
            m = d["episode_ids"] == eid
            labels.append(
                {
                    "episode_id": int(eid),
                    "steps": int(m.sum()),
                    "return": float(d["rewards"][m].sum()),
                    "terminated": bool(d["terminated"][m][-1]),
                    "success": None,
                }
            )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=2))
    print(out.resolve())


if __name__ == "__main__":
    main()
