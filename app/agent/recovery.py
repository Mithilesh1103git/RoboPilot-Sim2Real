"""Bounded recovery policy scaffold."""


def choose_recovery(attempt: int, max_retries: int, failure_reason: str) -> str:
    if attempt >= max_retries:
        return "stop_safely"
    if failure_reason in {"missed_grasp", "object_slipped", "placement_missed", "unexpected_scene"}:
        return "reobserve_and_replan"
    return "stop_safely"
