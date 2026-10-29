"""Standalone ensemble-style dynamics model training from numeric transitions."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim + 1),
        )

    def forward(self, obs, action):
        out = self.net(torch.cat([obs, action], dim=-1))
        return out[..., : self.obs_dim], out[..., self.obs_dim :]


def train_world_model(dataset_path, output_path, epochs=30, batch_size=256, seed=7):
    torch.manual_seed(seed)
    data = np.load(dataset_path)
    obs = torch.as_tensor(data["observations"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)
    nxt = torch.as_tensor(data["next_observations"], dtype=torch.float32)
    rewards = torch.as_tensor(data["rewards"], dtype=torch.float32).unsqueeze(-1)
    if len(obs) < 10:
        raise ValueError("At least 10 transitions are required.")
    model = DynamicsModel(obs.shape[1], actions.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(obs))
        losses = []
        for ids in order.split(batch_size):
            pred, rew_pred = model(obs[ids], actions[ids])
            loss = nn.functional.mse_loss(
                pred, nxt[ids]
            ) + 0.1 * nn.functional.mse_loss(rew_pred, rewards[ids])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch+1} loss={sum(losses)/len(losses):.6f}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": obs.shape[1],
            "action_dim": actions.shape[1],
        },
        path,
    )
    return model
