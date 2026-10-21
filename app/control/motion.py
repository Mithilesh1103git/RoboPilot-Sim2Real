"""Damped-least-squares Cartesian IK and bounded joint trajectory execution."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.control.safety import validate_joint_targets


@dataclass
class MotionResult:
    success: bool
    message: str = ""
    steps: int = 0
    final_position: list[float] | None = None


def _robot_joint_data(environment):
    import mujoco

    model = environment.model
    qpos_ids, dof_ids, actuator_ids, lows, highs = [], [], [], [], []
    for i in range(1, 8):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        aid = environment._actuator_for_joint(jid)
        if jid < 0 or aid is None:
            raise RuntimeError(
                f"Panda arm joint/actuator {i} is missing from the model."
            )
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))
        actuator_ids.append(int(aid))
        lows.append(float(model.jnt_range[jid, 0]))
        highs.append(float(model.jnt_range[jid, 1]))
    return qpos_ids, dof_ids, actuator_ids, np.array(lows), np.array(highs)


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


def inverse_kinematics(
    environment,
    target_xyz,
    robot_config=None,
    *,
    target_rotation=None,
    max_iterations=None,
):
    """Solve Panda hand pose IK with damped least squares, without stepping physics.

    Returns a seven-element arm joint target or raises ValueError when unreachable.
    """
    import mujoco

    if environment.model is None or environment.data is None:
        raise RuntimeError("Load the simulation before solving IK.")
    target = np.asarray(target_xyz, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_xyz must contain three finite coordinates.")
    qpos_ids, dof_ids, _, lows, highs = _robot_joint_data(environment)
    hand_id = mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_id < 0:
        raise RuntimeError("The Panda model has no body named 'hand'.")
    original_qpos = environment.data.qpos.copy()
    original_qvel = environment.data.qvel.copy()
    original_ctrl = environment.data.ctrl.copy()
    desired_rotation = np.asarray(
        (
            target_rotation
            if target_rotation is not None
            else environment._cached_hand_rotation
        ),
        dtype=float,
    )
    if desired_rotation.shape != (3, 3):
        desired_rotation = environment.data.xmat[hand_id].reshape(3, 3).copy()
    q = original_qpos[qpos_ids].copy()
    damping = 0.06
    iterations = max_iterations or settings.ik_iterations
    try:
        for _ in range(iterations):
            environment.data.qpos[qpos_ids] = q
            mujoco.mj_forward(environment.model, environment.data)
            current_pos = environment.data.xpos[hand_id].copy()
            current_rot = environment.data.xmat[hand_id].reshape(3, 3).copy()
            pos_error = target - current_pos
            rot_error = _rotation_error(current_rot, desired_rotation)
            if np.linalg.norm(pos_error) < 0.012 and np.linalg.norm(rot_error) < 0.18:
                return q.copy()
            jacp = np.zeros((3, environment.model.nv))
            jacr = np.zeros((3, environment.model.nv))
            mujoco.mj_jacBody(environment.model, environment.data, jacp, jacr, hand_id)
            jp, jr = jacp[:, dof_ids], jacr[:, dof_ids]
            # Orientation has a lower weight than translation but prevents the
            # hand from rotating unpredictably during a grasp.
            jac = np.vstack((jp, 0.25 * jr))
            error = np.concatenate((pos_error, 0.25 * rot_error))
            delta = jac.T @ np.linalg.solve(
                jac @ jac.T + damping * damping * np.eye(6), error
            )
            q = np.clip(q + np.clip(delta, -0.12, 0.12), lows + 1e-4, highs - 1e-4)
        raise ValueError(
            f"IK did not converge to target {target.tolist()}; final position error={float(np.linalg.norm(target - environment.data.xpos[hand_id])):.3f} m"
        )
    finally:
        environment.data.qpos[:] = original_qpos
        environment.data.qvel[:] = original_qvel
        environment.data.ctrl[:] = original_ctrl
        mujoco.mj_forward(environment.model, environment.data)


def execute_joint_trajectory(
    environment, joint_targets, *, max_steps: int = 1000, stop_event=None
) -> MotionResult:
    """Move the Panda arm to joint targets using the model's position-like actuators."""
    import mujoco

    if environment.model is None or environment.data is None:
        raise RuntimeError("Load the simulation before executing a trajectory.")
    qpos_ids, _, actuator_ids, lows, highs = _robot_joint_data(environment)
    targets = np.asarray(joint_targets, dtype=float)
    if targets.shape != (7,) or not np.all(np.isfinite(targets)):
        raise ValueError("joint_targets must contain seven finite joint angles.")
    validate_joint_targets(targets, lows, highs)
    start = environment.data.qpos[qpos_ids].copy()
    distance = float(np.max(np.abs(targets - start)))
    frames = min(max_steps, max(100, int(distance / 0.003)))
    steps = 0
    for frame in range(1, frames + 1):
        if stop_event is not None and stop_event.is_set():
            return MotionResult(False, "Stop requested during joint trajectory.", steps)
        alpha = frame / frames
        command = start + alpha * (targets - start)
        for actuator_id, value in zip(actuator_ids, command):
            environment.data.ctrl[actuator_id] = float(value)
        # Advance in short chunks; viewer synchronization remains centralized
        # in the environment wrapper.
        environment.step(1)
        steps += 1
    # Hold the final target until the position actuators converge or the
    # bounded settling budget is exhausted.
    settle_budget = min(250, max(50, max_steps // 2))
    converged = False
    for _ in range(settle_budget):
        if stop_event is not None and stop_event.is_set():
            return MotionResult(
                False, "Stop requested while settling the joint trajectory.", steps
            )
        for actuator_id, value in zip(actuator_ids, targets):
            environment.data.ctrl[actuator_id] = float(value)
        environment.step(1)
        steps += 1
        if float(np.max(np.abs(environment.data.qpos[qpos_ids] - targets))) < 0.06:
            converged = True
            break
    hand_id = mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    final_position = environment.data.xpos[hand_id].tolist()
    if not converged:
        return MotionResult(
            False,
            "Joint trajectory did not converge within the settling budget.",
            steps,
            final_position,
        )
    return MotionResult(True, "Joint trajectory converged.", steps, final_position)


def move_end_effector(environment, target_xyz, *, stop_event=None) -> MotionResult:
    target_q = inverse_kinematics(environment, target_xyz)
    return execute_joint_trajectory(
        environment, target_q, max_steps=settings.motion_steps, stop_event=stop_event
    )
