"""Canonical, source-independent demonstration dataset models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, Literal


class Transition(BaseModel):
    model_config = ConfigDict(extra="allow")
    timestamp: float | None = None
    observation: dict[str, Any] = Field(default_factory=dict)
    action: Any = None
    instruction: str = ""
    episode_id: str = "episode-0001"
    robot: str = "unknown"
    embodiment: str = "unknown"
    coordinate_frame: str = "world"
    task_metadata: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    success: bool | None = None


class DatasetEpisode(BaseModel):
    episode_id: str
    instruction: str = ""
    robot: str = "unknown"
    embodiment: str = "unknown"
    coordinate_frame: str = "world"
    transitions: list[Transition] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    schema_name: Literal["robopilot.demonstrations"] = "robopilot.demonstrations"
    schema_version: str = "1.0.0"
    dataset_id: str
    source_format: str
    source_name: str = "uploaded"
    created_at: str
    episode_count: int = 0
    transition_count: int = 0
    robots: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    source_format: str = Field(default="auto", min_length=1, max_length=40)
    source_name: str = Field(default="uploaded", min_length=1, max_length=120)
    records: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    coordinate_frame: str = "world"
    robot: str = "unknown"
    embodiment: str = "unknown"
    instruction: str = ""
    coordinate_transform: list[list[float]] | None = None
    embodiment_mapping: dict[str, str] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    dataset_id: str
    schema_name: str
    schema_version: str
    episode_count: int
    transition_count: int
    validation: dict[str, Any]
    manifest_path: str
