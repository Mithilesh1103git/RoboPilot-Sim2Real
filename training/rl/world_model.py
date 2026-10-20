"""Action-conditioned ensemble world model predicting next observations and rewards.

This is a learned dynamics model, not a replacement for MuJoCo physics. It trains on
transition NPZ files produced by the synthetic collector or rollout collector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .replay_buffer import ReplayBuffer


class DynamicsMember(nn.Module):
    def __init__(self, input_dim: int, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim + 1),
        )

    def forward(self, x):
        return self.net(x)


class WorldModelEnsemble(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, members: int = 5, hidden: int = 256
    ):
        super().__init__()
        if obs_dim < 1 or action_dim < 1 or members < 1:
            raise ValueError("Dimensions and member count must be positive")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.members = nn.ModuleList(
            [
                DynamicsMember(obs_dim + action_dim, obs_dim, hidden)
                for _ in range(members)
            ]
        )
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    def forward_member(self, index, obs, action):
        x = torch.cat(
            (
                (obs - self.obs_mean) / self.obs_std.clamp_min(1e-6),
                (action - self.action_mean) / self.action_std.clamp_min(1e-6),
            ),
            dim=-1,
        )
        out = self.members[index](x)
        delta = out[..., : self.obs_dim] * self.obs_std
        return obs + delta, out[..., self.obs_dim]

    @torch.no_grad()
    def predict(self, observations, actions, deterministic=True):
        obs = torch.as_tensor(
            observations, dtype=torch.float32, device=next(self.parameters()).device
        )
        act = torch.as_tensor(actions, dtype=torch.float32, device=obs.device)
        preds = [self.forward_member(i, obs, act) for i in range(len(self.members))]
        next_obs = torch.stack([p[0] for p in preds]).mean(0)
        rewards = torch.stack([p[1] for p in preds]).mean(0)
        uncertainty = torch.stack([p[0] for p in preds]).std(0, unbiased=False).mean(-1)
        return {
            "next_observations": next_obs.cpu().numpy(),
            "rewards": rewards.cpu().numpy(),
            "uncertainty": uncertainty.cpu().numpy(),
        }


def train_world_model(
    dataset: str,
    output: str,
    epochs: int = 50,
    batch_size: int = 256,
    members: int = 5,
    hidden: int = 256,
    learning_rate: float = 1e-3,
    device: str = "auto",
    seed: int = 7,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    buffer = ReplayBuffer()
    buffer.extend_npz(dataset)
    b = buffer.arrays()
    obs = torch.as_tensor(b.observations)
    actions = torch.as_tensor(b.actions)
    nxt = torch.as_tensor(b.next_observations)
    rewards = torch.as_tensor(b.rewards).reshape(-1, 1)
    if obs.shape[1] == 0 or actions.shape[1] == 0:
        raise ValueError("Dataset observations/actions cannot be empty")
    dev = torch.device(
        "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    model = WorldModelEnsemble(obs.shape[1], actions.shape[1], members, hidden).to(dev)
    model.obs_mean.copy_(obs.mean(0).to(dev))
    model.obs_std.copy_(obs.std(0, unbiased=False).clamp_min(1e-4).to(dev))
    model.action_mean.copy_(actions.mean(0).to(dev))
    model.action_std.copy_(actions.std(0, unbiased=False).clamp_min(1e-4).to(dev))
    # Predict state deltas; each member gets bootstrap resampling for ensemble diversity.
    xobs, xact, delta, rew = (
        obs.to(dev),
        actions.to(dev),
        (nxt - obs).to(dev),
        rewards.to(dev),
    )
    optimizers = [
        torch.optim.Adam(m.parameters(), lr=learning_rate) for m in model.members
    ]
    losses = []
    n = len(obs)
    for epoch in range(epochs):
        epoch_losses = []
        for i, (member, opt) in enumerate(zip(model.members, optimizers)):
            idx = torch.randint(0, n, (n,), device=dev)
            order = idx[torch.randperm(n, device=dev)]
            for start in range(0, n, batch_size):
                ids = order[start : start + batch_size]
                norm_obs = (xobs[ids] - model.obs_mean) / model.obs_std
                norm_act = (xact[ids] - model.action_mean) / model.action_std
                pred = member(torch.cat([norm_obs, norm_act], -1))
                # Predict normalized delta and reward to keep dimensions reasonably scaled.
                target_delta = delta[ids] / model.obs_std
                loss = nn.functional.mse_loss(
                    pred[:, : model.obs_dim], target_delta
                ) + nn.functional.mse_loss(pred[:, model.obs_dim :], rew[ids])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(member.parameters(), 10.0)
                opt.step()
                epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "members": members,
            "hidden": hidden,
            "seed": seed,
        },
        out,
    )
    metrics = {
        "dataset": str(dataset),
        "transitions": n,
        "epochs": epochs,
        "final_loss": losses[-1],
        "device": str(dev),
        "checkpoint": str(out),
    }
    out.with_suffix(out.suffix + ".metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return model, metrics


def load_world_model(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WorldModelEnsemble(
        ckpt["obs_dim"], ckpt["action_dim"], ckpt["members"], ckpt["hidden"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="training/rl/checkpoints/world_model.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    _, metrics = train_world_model(
        a.dataset,
        a.output,
        a.epochs,
        a.batch_size,
        a.members,
        a.hidden,
        device=a.device,
        seed=a.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
