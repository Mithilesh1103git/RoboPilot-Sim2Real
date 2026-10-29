"""Evaluate a saved PPO/SAC policy using the training-owned environment."""

import argparse

import numpy as np


def evaluate_policy(
    checkpoint,
    algorithm="PPO",
    episodes=10,
    seed=100,
    env_id="RoboPilotPickPlace-v0",
    robot_name="panda",
):
    if algorithm not in {"PPO", "SAC"}:
        raise ValueError("algorithm must be PPO or SAC")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    from stable_baselines3 import PPO, SAC

    from training.rl.env import make_env

    metadata_path = Path(str(checkpoint) + ".metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("robot_name") != robot_name:
            raise ValueError(
                f"Checkpoint robot profile is {metadata.get('robot_name')!r}, but evaluation requested {robot_name!r}."
            )
        if metadata.get("algorithm") != algorithm:
            raise ValueError(
                f"Checkpoint algorithm is {metadata.get('algorithm')!r}, but evaluation requested {algorithm!r}."
            )
    model = (PPO if algorithm == "PPO" else SAC).load(str(checkpoint))
    env = make_env(env_id, robot_name=robot_name, seed=seed)
    returns = []
    successes = []
    lengths = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ret = 0.0
            steps = 0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                ret += float(reward)
                steps += 1
                done = bool(term or trunc)
            returns.append(ret)
            successes.append(bool(info.get("success", False)))
            lengths.append(steps)
    finally:
        env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "successes": int(sum(successes)),
        "mean_episode_length": float(np.mean(lengths)),
        "algorithm": algorithm,
        "checkpoint": str(checkpoint),
        "env_id": env_id,
        "robot_name": robot_name,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--env-id", default="RoboPilotPickPlace-v0")
    p.add_argument("--robot-name", default="panda")
    print(evaluate_policy(**vars(p.parse_args(argv))))


if __name__ == "__main__":
    main()
