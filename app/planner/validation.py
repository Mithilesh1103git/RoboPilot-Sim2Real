"""Strict JSON parsing and policy validation for planner output."""

import json
import re

from app.schemas import TaskPlan

REQUIRED_SEQUENCE = [
    "approach_object",
    "grasp_object",
    "lift_object",
    "move_to_destination",
    "release_object",
    "verify",
]


def _extract_json(content):
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Planner response did not contain a JSON object.")
    return text[start : end + 1]


def parse_plan(content):
    try:
        payload = json.loads(_extract_json(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner returned invalid JSON: {exc}") from exc
    return TaskPlan.model_validate(payload)


def validate_plan(plan, instruction):
    if plan.target_object != "red_block":
        raise ValueError(f"Unsupported target object: {plan.target_object!r}.")
    if plan.destination != "blue_tray":
        raise ValueError(f"Unsupported destination: {plan.destination!r}.")
    if [step.action for step in plan.steps] != REQUIRED_SEQUENCE:
        raise ValueError(
            "Plan must contain the exact supported pick-and-place action sequence."
        )
    for step in plan.steps:
        if step.target_object not in (None, "red_block"):
            raise ValueError("Plan step references unsupported object.")
        if step.destination not in (None, "blue_tray"):
            raise ValueError("Plan step references unsupported destination.")
    if plan.recovery != "stop_safely":
        raise ValueError("Only stop_safely recovery is currently permitted.")
    return plan
