"""Suggested extensible knowledge categories, not mandatory schema fields."""

COMMON_CATEGORIES = {
    "kinematics",
    "dynamics",
    "physical_capabilities",
    "end_effector",
    "sensing",
    "control_interfaces",
    "safety",
    "operating_constraints",
    "locomotion",
    "balance",
    "manipulation",
    "navigation",
    "other_capabilities",
}

# Categories are suggestions only. Extractors may add robot-family-specific
# categories, e.g. "legged_locomotion" or "mobile_base".
