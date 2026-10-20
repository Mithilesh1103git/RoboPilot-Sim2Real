"""Run real deterministic pick-and-place trials and write JSONL metrics."""

import argparse
import json
import time
from pathlib import Path

from app.agent.executor import TaskExecutor
from app.config import settings
from app.simulation.environment import SimulationEnvironment


def run_trial(trial_id: int, seed: int, scenario: str = "default") -> dict:
    # The current baseline is deterministic. Seed is retained in the record so
    # randomized scenes can be introduced without changing the output schema.
    started = time.perf_counter()
    env = SimulationEnvironment(
        asset_root=settings.assets_dir, robot_name="panda", headless=True
    )
    executor = TaskExecutor(env, planner=None, max_retries=settings.max_retries)
    try:
        result = executor.run(
            "Pick up the red block and place it inside the blue tray.",
            scenario=scenario,
        )
        return {
            "trial_id": trial_id,
            "seed": seed,
            "success": bool(result.get("success")),
            "recovered": result.get("attempts", 1) > 1 and bool(result.get("success")),
            "duration_seconds": round(time.perf_counter() - started, 4),
            "status": result.get("state", "failed"),
            "message": result.get("message"),
            "result": result.get("result"),
        }
    except Exception as exc:
        return {
            "trial_id": trial_id,
            "seed": seed,
            "success": False,
            "recovered": False,
            "duration_seconds": round(time.perf_counter() - started, 4),
            "status": "failed",
            "message": str(exc),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="runs/trials.jsonl")
    parser.add_argument("--scenario", default="default")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [run_trial(i, args.seed + i, args.scenario) for i in range(args.trials)]
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    successes = sum(bool(row["success"]) for row in rows)
    print(
        f"Trials: {len(rows)} | successes: {successes} | success rate: {successes / len(rows):.1%} | output: {output}"
    )


if __name__ == "__main__":
    main()
