import argparse, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="training/datasets/summary.json")
    a = p.parse_args()
    with np.load(a.input) as d:
        s = {k: {"shape": list(d[k].shape), "dtype": str(d[k].dtype)} for k in d.files}
        if "rewards" in d and d["rewards"].size:
            s["reward_mean"] = float(np.mean(d["rewards"]))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2))
    print(out.resolve())


if __name__ == "__main__":
    main()
