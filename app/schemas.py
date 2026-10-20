"""Pydantic schemas used by the public API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskState = Literal["queued", "running", "succeeded", "failed", "stopped"]


class TaskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    instruction: str = Field(min_length=3, max_length=500)
    scenario: str = Field(default="default", min_length=1, max_length=80)


class PlanStep(BaseModel):
    action: Literal[
        "approach_object",
        "grasp_object",
        "lift_object",
        "move_to_destination",
        "release_object",
        "verify",
    ]
    target_object: str | None = None
    destination: str | None = None
    rationale: str = ""


class TaskPlan(BaseModel):
    task: str = "pick_and_place"
    target_object: str
    destination: str
    steps: list[PlanStep] = Field(min_length=1, max_length=12)
    recovery: Literal["reobserve_and_replan", "stop_safely"] = "reobserve_and_replan"


class TaskStatus(BaseModel):
    task_id: str
    status: TaskState
    instruction: str = ""
    scenario: str = "default"
    message: str = ""
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthStatus(BaseModel):
    status: str
    project: str
    simulation_connected: bool
    robot: str
