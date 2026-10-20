"""Training orchestration with explicit execution-frequency and queue controls.

Runs bounded collection/model-training/policy-training stages. GPU is selected by
PyTorch when available; MuJoCo itself is not assumed to be GPU accelerated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .replay_buffer import ReplayBuffer


class FlowController:
    def __init__(
        self,
        replay_capacity=100_000,
        model_update_every=1,
        rl_update_every=1,
        max_queue_size=100_000,
    ):
        if (
            min(replay_capacity, model_update_every, rl_update_every, max_queue_size)
            < 1
        ):
            raise ValueError("Flow control limits/frequencies must be positive")
        self.buffer = ReplayBuffer(replay_capacity)
        self.model_update_every = model_update_every
        self.rl_update_every = rl_update_every
        self.max_queue_size = max_queue_size
        self.events = []
        self.iteration = 0

    def ingest(self, dataset_path):
        before = len(self.buffer)
        self.buffer.extend_npz(dataset_path)
        if len(self.buffer) > self.max_queue_size:
            # ReplayBuffer already bounds capacity; this guard documents queue pressure.
            raise RuntimeError("Replay queue capacity exceeded")
        added = len(self.buffer) - before
        self.events.append(
            {
                "event": "ingest",
                "path": str(dataset_path),
                "added": added,
                "buffer_size": len(self.buffer),
            }
        )
        return added

    def should_update_world_model(self):
        return self.iteration % self.model_update_every == 0

    def should_update_policy(self):
        return self.iteration % self.rl_update_every == 0

    def next_iteration(self):
        self.iteration += 1
        return self.iteration

    def save_replay(self, path):
        self.buffer.save(path)

    def write_report(self, path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "iteration": self.iteration,
                    "buffer_size": len(self.buffer),
                    "events": self.events,
                },
                indent=2,
            )
        )
        return out


def run_pipeline(args):
    controller = FlowController(
        args.replay_capacity, args.world_model_every, args.rl_every, args.max_queue_size
    )
    for dataset in args.datasets:
        controller.ingest(dataset)
    if len(controller.buffer) == 0:
        raise ValueError("At least one transition dataset is required")
    merged = Path(args.output_dir) / "replay_buffer.npz"
    controller.save_replay(merged)
    controller.next_iteration()
    report = {
        "replay_buffer": str(merged),
        "transitions": len(controller.buffer),
        "world_model": None,
        "policy": None,
    }
    if controller.should_update_world_model():
        cmd = [
            sys.executable,
            "-m",
            "training.rl.world_model",
            "--dataset",
            str(merged),
            "--output",
            str(Path(args.output_dir) / "world_model.pt"),
            "--epochs",
            str(args.world_model_epochs),
            "--device",
            args.device,
        ]
        subprocess.run(cmd, check=True)
        report["world_model"] = str(Path(args.output_dir) / "world_model.pt")
        controller.events.append(
            {"event": "world_model_trained", "checkpoint": report["world_model"]}
        )
    if args.env_id and controller.should_update_policy():
        cmd = [
            sys.executable,
            "-m",
            "training.rl.train",
            "--env-id",
            args.env_id,
            "--algorithm",
            args.algorithm,
            "--timesteps",
            str(args.timesteps),
            "--output",
            str(Path(args.output_dir) / "policy"),
            "--device",
            args.device,
        ]
        subprocess.run(cmd, check=True)
        report["policy"] = str(Path(args.output_dir) / "policy")
        controller.events.append(
            {"event": "policy_trained", "checkpoint": report["policy"]}
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    controller.write_report(Path(args.output_dir) / "flow_report.json")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True, help="Transition NPZ files")
    p.add_argument("--output-dir", default="training/experiments/flow_run")
    p.add_argument("--replay-capacity", type=int, default=100000)
    p.add_argument("--max-queue-size", type=int, default=100000)
    p.add_argument("--world-model-every", type=int, default=1)
    p.add_argument("--rl-every", type=int, default=1)
    p.add_argument("--world-model-epochs", type=int, default=50)
    p.add_argument(
        "--env-id",
        default="",
        help="Optional registered Gymnasium env ID; omit to skip RL policy training",
    )
    p.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    run_pipeline(a)


if __name__ == "__main__":
    main()
