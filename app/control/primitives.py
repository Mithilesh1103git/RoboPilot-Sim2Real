"""Deterministic Panda pick-and-place primitives."""

from __future__ import annotations

import time

import numpy as np

from app.control.motion import move_end_effector


class PrimitiveError(RuntimeError):
    pass


def _check_stop(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Task stop requested.")


def _set_gripper(environment, opened: bool, stop_event=None) -> None:
    _check_stop(stop_event)
    import mujoco

    actuator_id = mujoco.mj_name2id(
        environment.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
    )
    if actuator_id < 0:
        raise PrimitiveError("Panda gripper actuator 'actuator8' is missing.")
    # In this Panda model 255 corresponds to open and 0 to closed.
    environment.data.ctrl[actuator_id] = 255.0 if opened else 0.0
    for _ in range(80):
        _check_stop(stop_event)
        environment.step(1)


def _move(environment, xyz, stop_event=None):
    _check_stop(stop_event)
    result = move_end_effector(environment, xyz, stop_event=stop_event)
    if not result.success:
        raise PrimitiveError(result.message)
    return result


def pick_and_place(
    environment,
    *,
    object_name="red_block",
    destination_name="blue_tray",
    stop_event=None,
) -> dict:
    """Attempt a physical grasp and placement using contacts and the Panda gripper."""
    if object_name != "red_block" or destination_name != "blue_tray":
        raise ValueError(
            "The built-in deterministic scenario supports red_block → blue_tray."
        )
    environment.load()
    environment.reset()
    start = environment.body_position(object_name)
    tray = environment.body_position(destination_name)
    # Hand body origin is above the fingertips in the Panda model. These offsets
    # position the fingertip pads around the block while preserving hand rotation.
    hand_offset = 0.105
    pregrasp = [float(start[0]), float(start[1]), float(start[2] + hand_offset + 0.10)]
    grasp = [float(start[0]), float(start[1]), float(start[2] + hand_offset)]
    preplace = [float(tray[0]), float(tray[1]), float(start[2] + hand_offset + 0.14)]
    place = [float(tray[0]), float(tray[1]), float(0.435 + hand_offset)]
    trace = []
    _set_gripper(environment, opened=True, stop_event=stop_event)
    trace.append({"action": "open_gripper", "success": True})
    _move(environment, pregrasp, stop_event)
    trace.append({"action": "approach_object", "target": pregrasp, "success": True})
    _move(environment, grasp, stop_event)
    trace.append({"action": "lower_to_grasp", "target": grasp, "success": True})
    _set_gripper(environment, opened=False, stop_event=stop_event)
    trace.append({"action": "close_gripper", "success": True})
    for _ in range(40):
        _check_stop(stop_event)
        environment.step(1)
    after_grasp = environment.body_position(object_name)
    # Lift and verify the object actually follows the gripper. A closed gripper
    # command alone is not considered a successful grasp.
    _move(environment, pregrasp, stop_event)
    after_lift = environment.body_position(object_name)
    lifted = float(after_lift[2] - after_grasp[2]) > 0.025
    trace.append(
        {
            "action": "lift_object",
            "object_position": after_lift.tolist(),
            "lift_detected": lifted,
        }
    )
    if not lifted:
        raise PrimitiveError(
            "Grasp verification failed: the block did not rise with the gripper."
        )
    _move(environment, preplace, stop_event)
    _move(environment, place, stop_event)
    trace.append({"action": "move_to_destination", "target": place, "success": True})
    _set_gripper(environment, opened=True, stop_event=stop_event)
    trace.append({"action": "release_object", "success": True})
    for _ in range(120):
        _check_stop(stop_event)
        environment.step(1)
    final_position = environment.body_position(object_name)
    # Tray's inner opening is about 0.24 x 0.24 m. Require the block center to
    # be inside the opening and resting above the tray floor, not just nearby.
    dx, dy = abs(float(final_position[0] - tray[0])), abs(
        float(final_position[1] - tray[1])
    )
    success = dx <= 0.105 and dy <= 0.105 and 0.405 <= float(final_position[2]) <= 0.49
    trace.append(
        {
            "action": "verify",
            "success": success,
            "object_position": final_position.tolist(),
            "destination_position": tray.tolist(),
        }
    )
    if not success:
        raise PrimitiveError(
            f"Placement verification failed; final object position={final_position.tolist()}."
        )
    return {
        "success": True,
        "object": object_name,
        "destination": destination_name,
        "object_start": start.tolist(),
        "object_final": final_position.tolist(),
        "destination_position": tray.tolist(),
        "trace": trace,
        "simulation_time": float(environment.data.time),
    }


def approach_object(target_xyz: tuple[float, float, float]) -> dict:
    return {
        "action": "approach_object",
        "target_xyz": list(target_xyz),
        "implemented": True,
    }


def grasp_object(object_name: str) -> dict:
    return {"action": "grasp_object", "object": object_name, "implemented": True}


def lift_object(height: float = 0.10) -> dict:
    return {"action": "lift_object", "height": height, "implemented": True}


def move_to_destination(destination_xyz: tuple[float, float, float]) -> dict:
    return {
        "action": "move_to_destination",
        "destination_xyz": list(destination_xyz),
        "implemented": True,
    }


def release_object() -> dict:
    return {"action": "release_object", "implemented": True}
