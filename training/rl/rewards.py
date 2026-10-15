def reach_reward(distance, success, distance_weight=1.0, success_bonus=10.0):
    """Illustrative dense reach reward; implement actual rewards in the task environment."""
    return -distance_weight * float(distance) + (success_bonus if success else 0.0)


# Manipulation reward candidates: task progress, stable grasp, placement accuracy,
# collision penalty, time penalty, and a success bonus. Test for reward hacking.
