"""Train PPO/SAC in the training-owned Panda environment."""

import argparse
from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from training.rl.env import make_env
from training.rl.policies import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    p.add_argument("--timesteps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--env-id", default="RoboPilotPandaPickPlace-v0")
    p.add_argument("--output", default="training/runs/panda_ppo")
    a = p.parse_args()
    env = make_env(a.env_id, seed=a.seed)
    try:
        check_env(env, warn=True)
        model = build_model(a.algorithm, env, a.seed, a.device)
        model.learn(total_timesteps=a.timesteps)
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        print(f"Saved {out}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
