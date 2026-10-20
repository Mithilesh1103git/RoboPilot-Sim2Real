from typing import Literal

from pydantic import BaseModel, Field


class StartRunRequest(BaseModel):
    run_id: str | None = None
    dataset_prefix: str = "unified-buffer/"
    artifact_prefix: str = "training-artifacts/"
    world_model_epochs: int = Field(default=20, ge=1, le=10000)
    world_model_every: int = Field(default=1, ge=1)
    rl_every: int = Field(default=1, ge=1)
    replay_capacity: int = Field(default=100000, ge=1)
    max_queue_size: int = Field(default=100000, ge=1)
    env_id: str | None = None
    algorithm: Literal["PPO", "SAC"] = "PPO"
    timesteps: int = Field(default=10000, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
