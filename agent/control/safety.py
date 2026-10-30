"""Joint-space safety gates."""

import numpy as np


def validate_joint_targets(targets, lower, upper, *, max_delta=3.5):
    targets = np.asarray(targets, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if targets.shape != lower.shape or targets.shape != upper.shape:
        raise ValueError("Joint target shape mismatch.")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Joint targets must be finite.")
    if np.any(targets < lower) or np.any(targets > upper):
        raise ValueError("Joint target violates joint limits.")
    if targets.size > 1 and np.any(np.abs(np.diff(targets)) > max_delta):
        raise ValueError("Adjacent joint target jump exceeds safety bound.")
    return targets
