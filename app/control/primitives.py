"""Executable bounded Panda pick-and-place primitive."""

from app.agent.verifier import verify_lift, verify_object_in_destination
from app.control.motion import move_end_effector


class PrimitiveError(RuntimeError):
    pass


def _check_stop(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Task stop requested.")


def _set_gripper(env, opened, stop_event=None):
    import mujoco

    _check_stop(stop_event)
    aid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    if aid < 0:
        raise PrimitiveError("Panda gripper actuator8 is missing.")
    env.data.ctrl[aid] = 255.0 if opened else 0.0
    for _ in range(80):
        _check_stop(stop_event)
        env.step()


def _move(env, xyz, stop_event=None):
    _check_stop(stop_event)
    result = move_end_effector(env, xyz, stop_event=stop_event)
    if not result.success:
        raise PrimitiveError(result.message)
    return result


def pick_and_place(
    env, *, object_name="red_block", destination_name="blue_tray", stop_event=None
):
    if object_name != "red_block" or destination_name != "blue_tray":
        raise ValueError("Only red_block → blue_tray is supported.")
    env.load()
    env.reset()
    start = env.body_position(object_name)
    tray = env.body_position(destination_name)
    trace = []
    _set_gripper(env, True, stop_event)
    _move(env, [start[0], start[1], start[2] + 0.12], stop_event)
    trace.append({"action": "approach_object", "success": True})
    _move(env, [start[0], start[1], start[2] + 0.035], stop_event)
    _set_gripper(env, False, stop_event)
    trace.append({"action": "grasp_object", "success": True})
    _move(env, [start[0], start[1], start[2] + 0.16], stop_event)
    lifted = env.body_position(object_name)
    lift_ok = verify_lift(start, lifted)
    trace.append(
        {
            "action": "lift_object",
            "success": lift_ok,
            "object_position": lifted.tolist(),
        }
    )
    if not lift_ok:
        raise PrimitiveError(
            "Object lift verification failed; object may not be grasped."
        )
    _move(env, [tray[0], tray[1], tray[2] + 0.18], stop_event)
    trace.append({"action": "move_to_destination", "success": True})
    _move(env, [tray[0], tray[1], tray[2] + 0.07], stop_event)
    _set_gripper(env, True, stop_event)
    trace.append({"action": "release_object", "success": True})
    for _ in range(100):
        _check_stop(stop_event)
        env.step()
    final = env.body_position(object_name)
    verdict = verify_object_in_destination(final, tray)
    trace.append({"action": "verify", "success": verdict["success"], **verdict})
    if not verdict["success"]:
        raise PrimitiveError(f"Placement verification failed: {verdict['reason']}.")
    return {
        "success": True,
        "object": object_name,
        "destination": destination_name,
        "object_start": start.tolist(),
        "object_final": final.tolist(),
        "destination_position": tray.tolist(),
        "trace": trace,
        "simulation_time": float(env.data.time),
    }
