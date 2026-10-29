"""Evaluate a saved policy and print empirical metrics."""

import argparse

import numpy as np
from stable_baselines3 import PPO, SAC

from training.rl.env import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    a = p.parse_args()
    model = (PPO if a.algorithm == "PPO" else SAC).load(a.checkpoint)
    env = make_env(seed=a.seed)
    returns = []
    successes = []
    try:
        for ep in range(a.episodes):
            obs, _ = env.reset(seed=a.seed + ep)
            done = False
            ret = 0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                ret += reward
                done = term or trunc
            returns.append(ret)
            successes.append(bool(info.get("success", False)))
    finally:
        env.close()
    print(
        {
            "episodes": a.episodes,
            "mean_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "successes": int(sum(successes)),
        }
    )


if __name__ == "__main__":
    main()
