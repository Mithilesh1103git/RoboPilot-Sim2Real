"""Deployable training orchestration with optional Stage 4 generation.

The Flow Controller accepts local transition files downloaded from object storage,
can invoke the synthetic generator as part of the same run, bounds the replay
buffer, and schedules World Model / optional RL updates. CLI is an optional wrapper
around the same pipeline function used by the container service.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
        added = len(self.buffer) - before
        if len(self.buffer) > self.max_queue_size:
            raise RuntimeError("Replay queue capacity exceeded")
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
    """Run Stage 4 generation (if enabled), then Stage 5 WM/RL training.

    `datasets` are local paths downloaded by the deploy service from object storage.
    Generated data is written into this run's output directory and joins the same
    replay buffer before model training.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = list(getattr(args, "datasets", []) or [])
    synthetic_report = None
    if getattr(args, "generate_synthetic", False):
        from training.synthetic.generate_episodes import generate_dataset

        synthetic_path = output_dir / "synthetic" / "transitions.npz"
        synthetic_report = generate_dataset(
            env_id=getattr(args, "synthetic_env_id", "Pendulum-v1"),
            episodes=int(getattr(args, "synthetic_episodes", 10)),
            seed=int(getattr(args, "synthetic_seed", 7)),
            policy=getattr(args, "synthetic_policy", "random"),
            output=synthetic_path,
        )
        dataset_paths.append(str(synthetic_path))

    if not dataset_paths:
        raise ValueError(
            "No transition datasets supplied; provide object-storage datasets or enable synthetic generation"
        )

    controller = FlowController(
        args.replay_capacity, args.world_model_every, args.rl_every, args.max_queue_size
    )
    for dataset in dataset_paths:
        controller.ingest(dataset)
    if len(controller.buffer) == 0:
        raise ValueError(
            "At least one transition dataset containing transitions is required"
        )

    merged = output_dir / "replay_buffer.npz"
    controller.save_replay(merged)
    controller.next_iteration()
    report = {
        "replay_buffer": str(merged),
        "transitions": len(controller.buffer),
        "synthetic_generation": synthetic_report,
        "world_model": None,
        "policy": None,
    }
    if controller.should_update_world_model():
        checkpoint = output_dir / "world_model.pt"
        cmd = [
            sys.executable,
            "-m",
            "training.rl.world_model",
            "--dataset",
            str(merged),
            "--output",
            str(checkpoint),
            "--epochs",
            str(args.world_model_epochs),
            "--device",
            args.device,
        ]
        subprocess.run(cmd, check=True)
        report["world_model"] = str(checkpoint)
        controller.events.append(
            {"event": "world_model_trained", "checkpoint": str(checkpoint)}
        )
    if getattr(args, "env_id", "") and controller.should_update_policy():
        policy_path = output_dir / "policy"
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
            str(policy_path),
            "--device",
            args.device,
        ]
        subprocess.run(cmd, check=True)
        report["policy"] = str(policy_path)
        controller.events.append(
            {"event": "policy_trained", "checkpoint": str(policy_path)}
        )
    controller.events.append(
        {"event": "pipeline_completed", "synthetic_generation": bool(synthetic_report)}
    )
    controller.write_report(output_dir / "flow_report.json")
    (output_dir / "pipeline_summary.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets", nargs="*", default=[], help="Optional transition NPZ files"
    )
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
    p.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate Stage 4 transitions inside this pipeline run",
    )
    p.add_argument("--synthetic-env-id", default="Pendulum-v1")
    p.add_argument("--synthetic-episodes", type=int, default=10)
    p.add_argument("--synthetic-seed", type=int, default=7)
    p.add_argument("--synthetic-policy", choices=["random", "zero"], default="random")
    a = p.parse_args()
    run_pipeline(a)


if __name__ == "__main__":
    main()
