"""Bounded recovery decisions."""


def choose_recovery(attempt, max_retries, failure_reason):
    if attempt >= max_retries:
        return "stop_safely"
    if failure_reason in {
        "missed_grasp",
        "object_slipped",
        "placement_missed",
        "ik_not_converged",
        "controller_failure",
    }:
        return "reset_and_retry"
    return "stop_safely"
