"""Simulator-grounded verification helpers."""

from __future__ import annotations


def verify_object_in_destination(
    object_position,
    destination_position,
    *,
    half_width: float = 0.105,
    min_z: float = 0.405,
    max_z: float = 0.49,
) -> dict:
    x, y, z = (float(v) for v in object_position)
    tx, ty, _ = (float(v) for v in destination_position)
    inside = (
        abs(x - tx) <= half_width and abs(y - ty) <= half_width and min_z <= z <= max_z
    )
    return {
        "success": inside,
        "object_position": [x, y, z],
        "destination_position": [tx, ty],
        "reason": (
            "object is inside the tray bounds"
            if inside
            else "object is outside the tray bounds or not resting at tray height"
        ),
    }


def verify_lift(
    start_position, current_position, minimum_height: float = 0.025
) -> bool:
    return float(current_position[2]) - float(start_position[2]) >= minimum_height
