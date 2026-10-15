"""Validated API and planner schemas."""
from typing import Literal
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    instruction: str = Field(min_length=3, max_length=500)
    scenario: str = "default"


class PlanStep(BaseModel):
    action: Literal["approach_object", "grasp_object", "lift_object", "move_to_destination", "release_object", "verify"]
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
    status: Literal["idle", "queued", "running", "succeeded", "failed", "stopped"]
    message: str = ""
