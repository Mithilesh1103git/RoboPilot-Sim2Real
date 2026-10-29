"""Strict JSON parsing and robot-profile-aware policy validation."""

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

ACTION_ALIASES = {
    "approach": "approach_object",
    "grasp": "grasp_object",
    "lift": "lift_object",
    "move": "move_to_destination",
    "release": "release_object",
}

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
        raise ValueError(
            f"Planner returned invalid JSON: {exc}"
        ) from exc

    if isinstance(payload, dict):
        steps = payload.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    action = step.get("action")
                    if action in ACTION_ALIASES:
                        step["action"] = ACTION_ALIASES[action]

    return TaskPlan.model_validate(payload)


def validate_plan(plan, instruction, profile=None):
    task = (profile or {}).get("task", {})
    object_name = task.get("object_name", "red_block")
    destination_name = task.get("destination_name", "blue_tray")

    if plan.task != "pick_and_place":
        raise ValueError("Only pick_and_place is currently supported.")

    if plan.target_object != object_name:
        raise ValueError(
            f"Unsupported target object: {plan.target_object!r}; "
            f"expected {object_name!r}."
        )

    if plan.destination != destination_name:
        raise ValueError(
            f"Unsupported destination: {plan.destination!r}; "
            f"expected {destination_name!r}."
        )

    if [step.action for step in plan.steps] != REQUIRED_SEQUENCE:
        raise ValueError(
            "Plan must contain the exact supported pick-and-place action sequence."
        )

    for step in plan.steps:
        if step.target_object not in (None, object_name):
            raise ValueError("Plan step references unsupported target object.")

        if step.destination not in (None, destination_name):
            raise ValueError("Plan step references unsupported destination.")

    # Enforce references on actions that require them.
    for step in plan.steps:
        if (
            step.action
            in {
                "approach_object",
                "grasp_object",
                "lift_object",
            }
            and step.target_object != object_name
        ):
            raise ValueError(f"{step.action} must reference {object_name!r}.")

        if (
            step.action
            in {
                "move_to_destination",
                "release_object",
            }
            and step.destination != destination_name
        ):
            raise ValueError(f"{step.action} must reference {destination_name!r}.")

    if plan.recovery != "stop_safely":
        raise ValueError("Only stop_safely recovery is currently permitted.")

    return plan
