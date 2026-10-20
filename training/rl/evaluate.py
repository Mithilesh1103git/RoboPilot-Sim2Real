import argparse
import numpy as np
from stable_baselines3 import PPO, SAC

from .env import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Pendulum-v1")
    p.add_argument("--model", required=True)
    p.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    p.add_argument("--episodes", type=int, default=5)
    a = p.parse_args()
    model = (PPO if a.algorithm == "PPO" else SAC).load(a.model)
    env = make_env(a.env_id)
    returns = []
    try:
        for ep in range(a.episodes):
            obs, _ = env.reset(seed=5000 + ep)
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(action)
                total += float(reward)
                done = term or trunc
            returns.append(total)
    finally:
        env.close()
    print(
        f"mean={np.mean(returns):.3f}, std={np.std(returns):.3f}, returns={[round(x,3) for x in returns]}"
    )


if __name__ == "__main__":
    main()
