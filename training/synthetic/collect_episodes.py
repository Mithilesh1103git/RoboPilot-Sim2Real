"""Collect simulator-generated transitions; no language model is used."""

import argparse
from pathlib import Path

import numpy as np

from training.rl.env import make_env


def collect(episodes=20, seed=7, output="training/datasets/transitions.npz"):
    env = make_env(seed=seed)
    obs_all = []
    acts = []
    rewards = []
    next_all = []
    terms = []
    truncs = []
    episode_ids = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                action = env.action_space.sample()
                nxt, reward, term, trunc, info = env.step(action)
                obs_all.append(obs)
                acts.append(action)
                rewards.append(reward)
                next_all.append(nxt)
                terms.append(term)
                truncs.append(trunc)
                episode_ids.append(ep)
                obs = nxt
                done = term or trunc
    finally:
        env.close()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.asarray(obs_all, dtype=np.float32),
        actions=np.asarray(acts, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.asarray(next_all, dtype=np.float32),
        terminated=np.asarray(terms, dtype=np.bool_),
        truncated=np.asarray(truncs, dtype=np.bool_),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
    )
    print(f"Saved {len(rewards)} transitions to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="training/datasets/transitions.npz")
    a = p.parse_args()
    collect(a.episodes, a.seed, a.output)
