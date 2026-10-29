import pytest

from app.planner.mock import FakePlanner
from app.planner.validation import parse_plan, validate_plan


def test_supported_plan_validates():
    assert (
        validate_plan(
            FakePlanner().plan("pick red block", {}), "pick red block"
        ).target_object
        == "red_block"
    )


def test_rejects_unsupported_destination():
    plan = FakePlanner().plan("pick red block", {})
    plan.destination = "unknown"
    with pytest.raises(ValueError):
        validate_plan(plan, "pick it")


def test_rejects_wrong_action_order():
    plan = FakePlanner().plan("pick red block", {})
    plan.steps = plan.steps[::-1]
    with pytest.raises(ValueError):
        validate_plan(plan, "pick it")


def test_parses_json_fence():
    raw = '```json\n{"task":"pick_and_place","target_object":"red_block","destination":"blue_tray","steps":[{"action":"verify"}],"recovery":"stop_safely"}\n```'
    assert parse_plan(raw).steps[0].action == "verify"
