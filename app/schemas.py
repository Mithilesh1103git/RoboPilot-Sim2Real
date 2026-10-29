"""Typed API and planner contracts. LLM output is untrusted input."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ActionName = Literal[
    "approach_object",
    "grasp_object",
    "lift_object",
    "move_to_destination",
    "release_object",
    "verify",
]


class TaskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    instruction: str = Field(min_length=5, max_length=500)
    scenario: Literal["panda_pick_place"] = "panda_pick_place"


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: ActionName
    target_object: str | None = None
    destination: str | None = None
    rationale: str = Field(default="", max_length=240)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["pick_and_place"] = "pick_and_place"
    target_object: str = Field(min_length=1, max_length=80)
    destination: str = Field(min_length=1, max_length=80)
    steps: list[PlanStep] = Field(min_length=1, max_length=8)
    recovery: Literal["reobserve_and_replan", "stop_safely"] = "stop_safely"


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["queued", "running", "succeeded", "failed", "stopped"]
    instruction: str
    scenario: str
    message: str = ""
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthStatus(BaseModel):
    status: str
    project: str
    planner_configured: bool
    simulation_engine: str = "MuJoCo"
