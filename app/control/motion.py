"""Robot-profile-driven damped least-squares IK and bounded trajectories."""

from dataclasses import dataclass

import numpy as np

from app.control.safety import validate_joint_targets


@dataclass
class MotionResult:
    success: bool
    message: str = ""
    steps: int = 0
    final_position: list | None = None


def _joint_data(env):
    qids, dofs, aids, lows, highs = env._joint_info()
    return (
        qids,
        dofs,
        aids,
        np.asarray(lows, dtype=float),
        np.asarray(highs, dtype=float),
    )


def inverse_kinematics(
    env, target_xyz, max_iterations=120, tolerance=0.012, damping=0.05
):
    import mujoco

    target = np.asarray(target_xyz, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Target must be finite XYZ.")
    profile = env.profile
    ee = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, profile["end_effector_body"]
    )
    if ee < 0:
        raise RuntimeError(
            f"Configured end-effector body {profile['end_effector_body']!r} is missing."
        )
    qids, dofs, _, lows, highs = _joint_data(env)
    original = env.data.qpos[qids].copy()
    if len(qids) < 3:
        raise RuntimeError(
            "Cartesian position IK requires at least three configured arm joints."
        )
    for _ in range(max_iterations):
        mujoco.mj_forward(env.model, env.data)
        error = target - env.data.xpos[ee].copy()
        if np.linalg.norm(error) < tolerance:
            answer = env.data.qpos[qids].copy()
            env.data.qpos[qids] = original
            mujoco.mj_forward(env.model, env.data)
            return answer
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jacBody(env.model, env.data, jacp, jacr, ee)
        jac = jacp[:, dofs]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(3), error)
        env.data.qpos[qids] = np.clip(
            env.data.qpos[qids] + np.clip(dq, -0.08, 0.08), lows + 1e-4, highs - 1e-4
        )
    env.data.qpos[qids] = original
    mujoco.mj_forward(env.model, env.data)
    if np.linalg.norm(target - env.data.xpos[ee]) > 0.045:
        raise ValueError(f"IK failed to converge to {target.tolist()}.")
    return env.data.qpos[qids].copy()


def execute_joint_trajectory(env, targets, *, max_steps=120, stop_event=None):
    import mujoco

    qids, _, aids, lows, highs = _joint_data(env)
    start = env.data.qpos[qids].copy()
    targets = validate_joint_targets(targets, lows, highs)
    frames = max(20, min(int(max_steps), 300))
    steps = 0
    for frame in range(1, frames + 1):
        if stop_event is not None and stop_event.is_set():
            return MotionResult(False, "Stop requested during trajectory.", steps)
        command = start + (frame / frames) * (targets - start)
        for aid, val in zip(aids, command):
            env.data.ctrl[aid] = float(val)
        env.step()
        steps += 1
    converged = False
    for _ in range(120):
        if stop_event is not None and stop_event.is_set():
            return MotionResult(False, "Stop requested while settling.", steps)
        for aid, val in zip(aids, targets):
            env.data.ctrl[aid] = float(val)
        env.step()
        steps += 1
        if float(np.max(np.abs(env.data.qpos[qids] - targets))) < 0.06:
            converged = True
            break
    ee = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, env.profile["end_effector_body"]
    )
    return MotionResult(
        converged,
        (
            "Joint trajectory converged."
            if converged
            else "Joint trajectory did not converge."
        ),
        steps,
        env.data.xpos[ee].tolist(),
    )


def move_end_effector(env, target_xyz, *, stop_event=None):
    return execute_joint_trajectory(
        env, inverse_kinematics(env, target_xyz), stop_event=stop_event
    )
