from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RLTrainRequest(BaseModel):
    algorithm: Literal["PPO", "SAC"] = "PPO"
    timesteps: int = Field(default=50000, ge=100, le=100_000_000)
    seed: int = Field(default=7, ge=0, le=2_147_483_647)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    robot_name: str = Field(default="panda", min_length=1, max_length=80)
    env_id: str = "RoboPilotPickPlace-v0"
    output: str = "training/runs/panda_ppo"


class SyntheticRequest(BaseModel):
    robot_name: str = Field(default="panda", min_length=1, max_length=80)
    episodes: int = Field(default=20, ge=1, le=10000)
    seed: int = Field(default=7, ge=0, le=2_147_483_647)
    output: str = "training/datasets/transitions.npz"


class EvaluationRequest(BaseModel):
    robot_name: str = Field(default="panda", min_length=1, max_length=80)
    env_id: str = "RoboPilotPickPlace-v0"
    checkpoint: str = Field(min_length=1, max_length=500)
    algorithm: Literal["PPO", "SAC"] = "PPO"
    episodes: int = Field(default=10, ge=1, le=1000)
    seed: int = Field(default=100, ge=0, le=2_147_483_647)


class WorldModelRequest(BaseModel):
    dataset: str = "training/datasets/transitions.npz"
    output: str = "training/runs/world_model.pt"
    epochs: int = Field(default=30, ge=1, le=10000)

    @field_validator("dataset", "output")
    @classmethod
    def nonempty_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Path cannot be empty")
        return value


class ArtifactImportRequest(BaseModel):
    source: str = Field(min_length=1, max_length=1000)
    destination_name: str = Field(
        min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$"
    )
