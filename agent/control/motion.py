"""Robot-profile-driven damped least-squares IK and bounded trajectories."""

from dataclasses import dataclass, field

import numpy as np

from app.control.safety import validate_joint_targets


@dataclass
class MotionResult:
    success: bool
    message: str = ""
    steps: int = 0
    final_position: list | None = None
    diagnostics: dict = field(default_factory=dict)


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
    for iteration in range(max_iterations):
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
        dq = jac.T @ np.linalg.solve(
            jac @ jac.T + damping * damping * np.eye(3), error
        )
        env.data.qpos[qids] = np.clip(
            env.data.qpos[qids] + np.clip(dq, -0.08, 0.08),
            lows + 1e-4,
            highs - 1e-4,
        )
    mujoco.mj_forward(env.model, env.data)
    residual = float(np.linalg.norm(target - env.data.xpos[ee]))
    best = env.data.qpos[qids].copy()
    env.data.qpos[qids] = original
    mujoco.mj_forward(env.model, env.data)
    if residual > 0.045:
        raise ValueError(
            f"IK failed to converge to {target.tolist()}; "
            f"best Cartesian residual was {residual:.6f} m after {max_iterations} iterations."
        )
    return best


def _motion_diagnostics(env, qids, dofs, aids, targets, steps, target_xyz=None):
    import mujoco

    actual = env.data.qpos[qids].copy()
    errors = actual - targets
    forces = env.data.actuator_force[aids].copy()
    controls = env.data.ctrl[aids].copy()
    force_limited = env.model.actuator_forcelimited[aids].astype(bool)
    force_ranges = env.model.actuator_forcerange[aids].copy()
    saturated = []
    for index, limited in enumerate(force_limited):
        if not limited:
            saturated.append(False)
            continue
        lo, hi = force_ranges[index]
        force = forces[index]
        saturated.append(
            bool(force <= lo + 1e-3 or force >= hi - 1e-3)
        )

    ee_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY,
        env.profile["end_effector_body"]
    )
    final_position = env.data.xpos[ee_id].copy()
    result = {
        "joint_names": list(env.profile["arm_joint_names"]),
        "target_joint_positions": targets.tolist(),
        "actual_joint_positions": actual.tolist(),
        "joint_position_errors": errors.tolist(),
        "absolute_joint_position_errors": np.abs(errors).tolist(),
        "max_absolute_joint_error_rad": float(np.max(np.abs(errors))),
        "joint_velocities": env.data.qvel[dofs].tolist(),
        "actuator_ids": [int(x) for x in aids],
        "actuator_controls": controls.tolist(),
        "actuator_forces": forces.tolist(),
        "actuator_force_limited": force_limited.tolist(),
        "actuator_force_ranges": force_ranges.tolist(),
        "actuator_saturated": saturated,
        "simulation_steps": int(steps),
        "simulation_time": float(env.data.time),
        "end_effector_body": env.profile["end_effector_body"],
        "end_effector_final_position": final_position.tolist(),
    }
    if target_xyz is not None:
        target = np.asarray(target_xyz, dtype=float)
        delta = final_position - target
        result.update(
            end_effector_target_position=target.tolist(),
            end_effector_position_error=delta.tolist(),
            end_effector_error_norm_m=float(np.linalg.norm(delta)),
        )
    return result, final_position.tolist()


def execute_joint_trajectory(
    env, targets, *, max_steps=120, stop_event=None, target_xyz=None
):
    qids, dofs, aids, lows, highs = _joint_data(env)
    start = env.data.qpos[qids].copy()
    targets = validate_joint_targets(targets, lows, highs)
    frames = max(20, min(int(max_steps), 300))
    steps = 0

    def stopped(message):
        diagnostics, final_position = _motion_diagnostics(
            env, qids, dofs, aids, targets, steps, target_xyz
        )
        return MotionResult(False, message, steps, final_position, diagnostics)

    for frame in range(1, frames + 1):
        if stop_event is not None and stop_event.is_set():
            return stopped("Stop requested during trajectory.")
        command = start + (frame / frames) * (targets - start)
        for aid, val in zip(aids, command):
            env.data.ctrl[aid] = float(val)
        env.step()
        steps += 1

    converged = False
    # Preserve the existing settling budget; diagnostics will tell us whether
    # additional simulation time is actually needed.
    for _ in range(120):
        if stop_event is not None and stop_event.is_set():
            return stopped("Stop requested while settling.")
        for aid, val in zip(aids, targets):
            env.data.ctrl[aid] = float(val)
        env.step()
        steps += 1
        if float(np.max(np.abs(env.data.qpos[qids] - targets))) < 0.06:
            converged = True
            break

    diagnostics, final_position = _motion_diagnostics(
        env, qids, dofs, aids, targets, steps, target_xyz
    )
    return MotionResult(
        converged,
        "Joint trajectory converged." if converged
        else "Joint trajectory did not converge.",
        steps,
        final_position,
        diagnostics,
    )


def move_end_effector(env, target_xyz, *, stop_event=None):
    target = np.asarray(target_xyz, dtype=float)
    targets = inverse_kinematics(env, target)
    return execute_joint_trajectory(
        env, targets, stop_event=stop_event, target_xyz=target
    )
