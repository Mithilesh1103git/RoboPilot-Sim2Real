import argparse
import gymnasium as gym
import numpy as np
from pathlib import Path


def flat(x):
    if isinstance(x, dict):
        raise TypeError("Dict observations require a task-specific exporter.")
    return np.asarray(x, dtype=np.float32).reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Pendulum-v1")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--policy", choices=["random", "zero"], default="random")
    p.add_argument("--output", default="training/datasets/transitions.npz")
    a = p.parse_args()
    env = gym.make(a.env_id)
    obs_l, act_l, rew_l, nxt_l, term_l, trunc_l, ids = [], [], [], [], [], [], []
    try:
        for eid in range(a.episodes):
            obs, _ = env.reset(seed=a.seed + eid)
            done = False
            while not done:
                action = (
                    env.action_space.sample()
                    if a.policy == "random"
                    else np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                )
                nxt, rew, term, trunc, _ = env.step(action)
                obs_l.append(flat(obs))
                act_l.append(np.asarray(action, dtype=np.float32).reshape(-1))
                rew_l.append(rew)
                nxt_l.append(flat(nxt))
                term_l.append(term)
                trunc_l.append(trunc)
                ids.append(eid)
                obs, done = nxt, term or trunc
    finally:
        env.close()
    out = Path(a.output)
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
    print(f"Saved {len(rew_l)} transitions to {out.resolve()}")


if __name__ == "__main__":
    main()
