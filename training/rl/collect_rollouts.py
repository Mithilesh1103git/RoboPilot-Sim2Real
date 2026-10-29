"""Collect policy rollouts into a transition dataset; callable API plus CLI."""

import argparse
import json
from pathlib import Path

import numpy as np


def collect_policy_rollouts(
    model,
    algorithm="PPO",
    env_id="RoboPilotPickPlace-v0",
    robot_name="panda",
    episodes=10,
    output="training/datasets/policy_rollouts.npz",
    seed=1000,
):
    if algorithm not in {"PPO", "SAC"}:
        raise ValueError("algorithm must be PPO or SAC")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    from stable_baselines3 import PPO, SAC

    from training.rl.env import make_env

    metadata_path = Path(str(model) + ".metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("robot_name") != robot_name:
            raise ValueError(
                f"Checkpoint robot profile is {metadata.get('robot_name')!r}, but rollout collection requested {robot_name!r}."
            )
        if metadata.get("algorithm") != algorithm:
            raise ValueError(
                f"Checkpoint algorithm is {metadata.get('algorithm')!r}, but {algorithm!r} was requested."
            )
    policy = (PPO if algorithm == "PPO" else SAC).load(str(model))
    env = make_env(env_id, robot_name=robot_name, seed=seed)
    observations = []
    actions = []
    rewards = []
    next_obs = []
    terms = []
    truncs = []
    ids = []
    try:
        for eid in range(episodes):
            obs, _ = env.reset(seed=seed + eid)
            done = False
            while not done:
                action, _ = policy.predict(obs, deterministic=True)
                nxt, rew, term, trunc, _ = env.step(action)
                observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
                rewards.append(float(rew))
                next_obs.append(np.asarray(nxt, dtype=np.float32).reshape(-1))
                terms.append(bool(term))
                truncs.append(bool(trunc))
                ids.append(eid)
                obs = nxt
                done = bool(term or trunc)
    finally:
        env.close()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminated=terms,
        truncated=truncs,
        episode_ids=ids,
        robot_name=np.asarray(robot_name),
    )
    return {
        "output": str(out),
        "episodes": episodes,
        "transitions": len(rewards),
        "robot_name": robot_name,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="RoboPilotPickPlace-v0")
    p.add_argument("--robot-name", default="panda")
    p.add_argument("--model", required=True)
    p.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--output", default="training/datasets/policy_rollouts.npz")
    p.add_argument("--seed", type=int, default=1000)
    args = vars(p.parse_args(argv))
    model = args.pop("model")
    print(collect_policy_rollouts(model=model, **args))


if __name__ == "__main__":
    main()
