"""Strictly parse planner JSON and validate it with Pydantic."""
import json
from app.schemas import TaskPlan


def parse_plan(content: str) -> TaskPlan:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner returned invalid JSON: {exc}") from exc
    return TaskPlan.model_validate(payload)
