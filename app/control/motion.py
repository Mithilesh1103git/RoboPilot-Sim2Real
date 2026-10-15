"""Robot motion helpers. Replace stubs with IK and trajectory execution."""

from dataclasses import dataclass


@dataclass
class MotionResult:
    success: bool
    message: str = ""


def inverse_kinematics(target_xyz, robot_config=None):
    raise NotImplementedError("Implement IK for the selected MuJoCo robot model.")


def execute_joint_trajectory(
    environment, joint_targets, *, max_steps: int = 1000
) -> MotionResult:
    raise NotImplementedError(
        "Implement bounded trajectory execution and convergence checks."
    )
