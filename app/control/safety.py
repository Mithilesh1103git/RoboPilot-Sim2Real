"""Safety validation for high-level plans and low-level motion targets."""

ALLOWED_ACTIONS = {
    "approach_object",
    "grasp_object",
    "lift_object",
    "move_to_destination",
    "release_object",
    "verify",
}


def validate_action_name(action: str) -> bool:
    return action in ALLOWED_ACTIONS


def validate_joint_targets(targets, lower_limits, upper_limits) -> None:
    if len(targets) != len(lower_limits) or len(targets) != len(upper_limits):
        raise ValueError("Joint target and limit dimensions do not match.")
    for i, (value, low, high) in enumerate(zip(targets, lower_limits, upper_limits)):
        if not low <= value <= high:
            raise ValueError(f"Joint target {i} is outside configured limits.")
