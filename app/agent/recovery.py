"""Bounded recovery decisions for deterministic task execution."""


def choose_recovery(attempt: int, max_retries: int, failure_reason: str) -> str:
    if attempt > max_retries:
        return "stop_safely"
    if failure_reason in {
        "missed_grasp",
        "object_slipped",
        "placement_missed",
        "ik_not_converged",
        "unexpected_scene",
    }:
        return "reset_and_retry"
    return "stop_safely"
