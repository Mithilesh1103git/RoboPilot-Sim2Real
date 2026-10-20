"""Bounded transition replay buffer for world-model and policy data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_ids: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._rows: list[tuple] = []

    def __len__(self):
        return len(self._rows)

    def add(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated=False,
        truncated=False,
        episode_id=0,
    ):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        nxt = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        if obs.size == 0 or act.size == 0 or nxt.size != obs.size:
            raise ValueError(
                "observations/actions must be non-empty and next observation shape must match"
            )
        row = (
            obs.copy(),
            act.copy(),
            float(reward),
            nxt.copy(),
            bool(terminated),
            bool(truncated),
            int(episode_id),
        )
        self._rows.append(row)
        if len(self._rows) > self.capacity:
            del self._rows[: len(self._rows) - self.capacity]

    def extend_npz(self, path: str | Path):
        with np.load(path, allow_pickle=False) as d:
            required = {
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "terminated",
                "truncated",
                "episode_ids",
            }
            missing = required - set(d.files)
            if missing:
                raise ValueError(f"Transition dataset missing keys: {sorted(missing)}")
            n = len(d["rewards"])
            if any(len(d[k]) != n for k in required):
                raise ValueError("Transition arrays have inconsistent lengths")
            for i in range(n):
                self.add(
                    d["observations"][i],
                    d["actions"][i],
                    d["rewards"][i],
                    d["next_observations"][i],
                    d["terminated"][i],
                    d["truncated"][i],
                    d["episode_ids"][i],
                )

    def arrays(self) -> TransitionBatch:
        if not self._rows:
            raise ValueError("Replay buffer is empty")
        columns = list(zip(*self._rows))
        return TransitionBatch(
            observations=np.stack(columns[0]).astype(np.float32),
            actions=np.stack(columns[1]).astype(np.float32),
            rewards=np.asarray(columns[2], dtype=np.float32),
            next_observations=np.stack(columns[3]).astype(np.float32),
            terminated=np.asarray(columns[4], dtype=np.bool_),
            truncated=np.asarray(columns[5], dtype=np.bool_),
            episode_ids=np.asarray(columns[6], dtype=np.int64),
        )

    def save(self, path: str | Path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        b = self.arrays()
        np.savez_compressed(
            out,
            observations=b.observations,
            actions=b.actions,
            rewards=b.rewards,
            next_observations=b.next_observations,
            terminated=b.terminated,
            truncated=b.truncated,
            episode_ids=b.episode_ids,
        )
        return out
