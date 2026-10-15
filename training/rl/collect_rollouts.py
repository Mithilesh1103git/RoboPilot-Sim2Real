import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, SAC
from .env import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Pendulum-v1")
    p.add_argument("--model", required=True)
    p.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--output", default="training/datasets/policy_rollouts.npz")
    a = p.parse_args()
    model = (PPO if a.algorithm == "PPO" else SAC).load(a.model)
    env = make_env(a.env_id)
    o = []
    ac = []
    r = []
    no = []
    te = []
    tr = []
    ids = []
    try:
        for eid in range(a.episodes):
            obs, _ = env.reset(seed=1000 + eid)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                nxt, rew, term, trunc, _ = env.step(action)
                o.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                ac.append(np.asarray(action, dtype=np.float32).reshape(-1))
                r.append(rew)
                no.append(np.asarray(nxt, dtype=np.float32).reshape(-1))
                te.append(term)
                tr.append(trunc)
                ids.append(eid)
                obs = nxt
                done = term or trunc
    finally:
        env.close()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=o,
        actions=ac,
        rewards=r,
        next_observations=no,
        terminated=te,
        truncated=tr,
        episode_ids=ids,
    )
    print(f"Saved {len(r)} transitions to {out.resolve()}")


if __name__ == "__main__":
    main()
