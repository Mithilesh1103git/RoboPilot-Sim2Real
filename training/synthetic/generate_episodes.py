"""Generate transition datasets from a Gymnasium environment.

This module is callable from the deployed Flow Controller as well as via CLI.
The default Pendulum environment is only a smoke-test environment; robot data
requires a registered RoboPilot environment adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def flat(value):
    if isinstance(value, dict):
        raise TypeError("Dict observations require a task-specific exporter.")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def generate_dataset(
    env_id: str, episodes: int, seed: int, policy: str, output: str | Path
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if policy not in {"random", "zero"}:
        raise ValueError("policy must be 'random' or 'zero'")
    import gymnasium as gym

    env = gym.make(env_id)
    obs_l, act_l, rew_l, nxt_l, term_l, trunc_l, ids = [], [], [], [], [], [], []
    try:
        for eid in range(episodes):
            obs, _ = env.reset(seed=seed + eid)
            done = False
            while not done:
                action = (
                    env.action_space.sample()
                    if policy == "random"
                    else np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                )
                nxt, rew, term, trunc, _ = env.step(action)
                obs_l.append(flat(obs))
                act_l.append(np.asarray(action, dtype=np.float32).reshape(-1))
                rew_l.append(float(rew))
                nxt_l.append(flat(nxt))
                term_l.append(bool(term))
                trunc_l.append(bool(trunc))
                ids.append(eid)
                obs, done = nxt, bool(term or trunc)
    finally:
        env.close()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=np.asarray(obs_l, dtype=np.float32),
        actions=np.asarray(act_l, dtype=np.float32),
        rewards=np.asarray(rew_l, dtype=np.float32),
        next_observations=np.asarray(nxt_l, dtype=np.float32),
        terminated=np.asarray(term_l),
        truncated=np.asarray(trunc_l),
        episode_ids=np.asarray(ids, dtype=np.int32),
    )
    return {
        "path": str(out),
        "env_id": env_id,
        "episodes": episodes,
        "transitions": len(rew_l),
        "seed": seed,
        "policy": policy,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Pendulum-v1")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--policy", choices=["random", "zero"], default="random")
    p.add_argument("--output", default="training/datasets/transitions.npz")
    a = p.parse_args()
    result = generate_dataset(a.env_id, a.episodes, a.seed, a.policy, a.output)
    print(f"Saved {result['transitions']} transitions to {Path(a.output).resolve()}")


if __name__ == "__main__":
    main()
