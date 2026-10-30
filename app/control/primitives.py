"""Profile-driven pick-and-place primitive for configured MuJoCo robots."""

from app.agent.verifier import verify_lift, verify_object_in_destination
from app.control.motion import move_end_effector


class PrimitiveError(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _check_stop(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Task stop requested.")


def _set_gripper(env, opened, stop_event=None):
    _check_stop(stop_event)
    if not env.profile.get("gripper_actuators"):
        raise PrimitiveError(
            f"Robot profile {env.robot_name!r} has no configured gripper actuators."
        )
    env.set_gripper(opened)
    for _ in range(80):
        _check_stop(stop_event)
        env.step()


def _move(env, xyz, stop_event=None):
    _check_stop(stop_event)
    result = move_end_effector(env, xyz, stop_event=stop_event)
    if not result.success:
        raise PrimitiveError(result.message, result.diagnostics)
    return result


def pick_and_place(env, *, object_name=None, destination_name=None, stop_event=None):
    task = env.profile["task"]
    object_name = object_name or task.get("object_name", "red_block")
    destination_name = destination_name or task.get("destination_name", "blue_tray")
    if object_name != task.get(
        "object_name", "red_block"
    ) or destination_name != task.get("destination_name", "blue_tray"):
        raise ValueError(
            f"Requested object/destination are not configured for robot {env.robot_name!r}."
        )
    env.load()
    env.reset()
    start = env.body_position(object_name)
    tray = env.body_position(destination_name)
    trace = []
    lift_min = float(task.get("lift_min_height", 0.025))
    _set_gripper(env, True, stop_event)
    _move(env, [start[0], start[1], start[2] + 0.12], stop_event)
    trace.append({"action": "approach_object", "success": True})
    _move(env, [start[0], start[1], start[2] + 0.035], stop_event)
    _set_gripper(env, False, stop_event)
    trace.append({"action": "grasp_object", "success": True})
    _move(env, [start[0], start[1], start[2] + 0.16], stop_event)
    lifted = env.body_position(object_name)
    lift_ok = verify_lift(start, lifted, min_height=lift_min)
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
    tolerance = task.get("placement_tolerance", [0.105, 0.105, 0.085])
    verdict = verify_object_in_destination(final, tray, tolerance=tolerance)
    trace.append({"action": "verify", "success": verdict["success"], **verdict})
    if not verdict["success"]:
        raise PrimitiveError(f"Placement verification failed: {verdict['reason']}.")
    return {
        "success": True,
        "robot": env.robot_name,
        "object": object_name,
        "destination": destination_name,
        "object_start": start.tolist(),
        "object_final": final.tolist(),
        "destination_position": tray.tolist(),
        "trace": trace,
        "simulation_time": float(env.data.time),
    }
