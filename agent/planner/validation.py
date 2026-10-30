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
        raise ValueError(f"Planner returned invalid JSON: {exc}") from exc

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

    # Bind required step references deterministically from the already-validated
    # top-level task entities. Nemotron may omit a repeated reference even when
    # the top-level target_object/destination are correct.
    #
    # Missing references are filled from the robot profile. Explicit conflicting
    # references are never silently rewritten; they are rejected.
    normalized_steps = []
    object_required_actions = {
        "approach_object",
        "grasp_object",
        "lift_object",
        "move_to_destination",
        "release_object",
        "verify",
    }
    destination_required_actions = {
        "move_to_destination",
        "release_object",
        "verify",
    }

    for step in plan.steps:
        updates = {}

        if step.action in object_required_actions:
            if step.target_object is None:
                updates["target_object"] = object_name
            elif step.target_object != object_name:
                raise ValueError(
                    f"{step.action} references unsupported target object "
                    f"{step.target_object!r}; expected {object_name!r}."
                )

        if step.action in destination_required_actions:
            if step.destination is None:
                updates["destination"] = destination_name
            elif step.destination != destination_name:
                raise ValueError(
                    f"{step.action} references unsupported destination "
                    f"{step.destination!r}; expected {destination_name!r}."
                )

        normalized_steps.append(
            step.model_copy(update=updates) if updates else step
        )

    if normalized_steps != plan.steps:
        plan = plan.model_copy(update={"steps": normalized_steps})

    if plan.recovery != "stop_safely":
        raise ValueError("Only stop_safely recovery is currently permitted.")

    return plan
