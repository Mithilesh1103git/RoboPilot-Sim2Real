"""Starter CLI for repeatable evaluation; connect to TaskExecutor."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="runs/trials.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder rows make the expected artifact format explicit.
    with output.open("w", encoding="utf-8") as f:
        for i in range(args.trials):
            row = {
                "trial_id": i,
                "seed": args.seed + i,
                "success": None,
                "recovered": None,
                "duration_seconds": None,
                "status": "not_implemented",
            }
            f.write(json.dumps(row) + "\n")
    print(
        f"Wrote placeholder trial records to {output}. Connect TaskExecutor to run real trials."
    )


if __name__ == "__main__":
    main()
