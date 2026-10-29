"""Collect transitions from a profile-selected training-owned MuJoCo robot environment."""

import argparse
from pathlib import Path

import numpy as np


def collect(
    episodes=20, seed=7, output="training/datasets/transitions.npz", robot_name="panda"
):
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    from training.rl.env import make_env

    env = make_env(robot_name=robot_name, seed=seed)
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
                obs_all.append(np.asarray(obs, dtype=np.float32))
                acts.append(np.asarray(action, dtype=np.float32))
                rewards.append(float(reward))
                next_all.append(np.asarray(nxt, dtype=np.float32))
                terms.append(bool(term))
                truncs.append(bool(trunc))
                episode_ids.append(ep)
                obs = nxt
                done = bool(term or trunc)
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
        robot_name=np.asarray(robot_name),
    )
    return {
        "output": str(path),
        "episodes": episodes,
        "transitions": len(rewards),
        "seed": seed,
        "robot_name": robot_name,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="training/datasets/transitions.npz")
    p.add_argument("--robot-name", default="panda")
    result = collect(**vars(p.parse_args(argv)))
    print(f"Saved {result['transitions']} transitions to {result['output']}")


if __name__ == "__main__":
    main()
