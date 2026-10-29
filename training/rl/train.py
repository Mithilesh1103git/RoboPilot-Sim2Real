"""Train PPO/SAC; run_training is the reusable Python API and main is a CLI adapter."""

import argparse
import json
from pathlib import Path


def run_training(
    algorithm="PPO",
    timesteps=50000,
    seed=7,
    device="auto",
    env_id="RoboPilotPickPlace-v0",
    robot_name="panda",
    output=None,
):
    if algorithm not in {"PPO", "SAC"}:
        raise ValueError("algorithm must be PPO or SAC")
    if timesteps < 1:
        raise ValueError("timesteps must be >= 1")
    from stable_baselines3.common.env_checker import check_env

    from training.rl.env import make_env
    from training.rl.policies import build_model

    env = make_env(env_id, robot_name=robot_name, seed=seed)
    try:
        check_env(env, warn=True)
        model = build_model(algorithm, env, seed, device)
        model.learn(total_timesteps=timesteps)
        out = Path(output or f"training/runs/{robot_name}_{algorithm.lower()}")
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        checkpoint = str(out) + ".zip"
        metadata = {
            "robot_name": robot_name,
            "algorithm": algorithm,
            "timesteps": timesteps,
            "env_id": env_id,
            "observation_shape": list(env.observation_space.shape),
            "action_shape": list(env.action_space.shape),
        }
        Path(checkpoint + ".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return {"checkpoint": checkpoint, **metadata}
    finally:
        env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    p.add_argument("--timesteps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--env-id", default="RoboPilotPickPlace-v0")
    p.add_argument("--robot-name", default="panda")
    p.add_argument("--output", default=None)
    result = run_training(**vars(p.parse_args(argv)))
    print(f"Saved {result['checkpoint']}")


if __name__ == "__main__":
    main()
