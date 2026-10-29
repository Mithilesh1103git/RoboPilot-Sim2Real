"""Simulator-grounded verification; LLM statements are never treated as evidence."""


def verify_object_in_destination(
    object_position, destination_position, *, half_width=0.105, min_z=0.405, max_z=0.49
):
    x, y, z = (float(v) for v in object_position)
    tx, ty, _ = (float(v) for v in destination_position)
    ok = abs(x - tx) <= half_width and abs(y - ty) <= half_width and min_z <= z <= max_z
    return {
        "success": ok,
        "object_position": [x, y, z],
        "destination_position": [tx, ty],
        "reason": (
            "object is inside tray bounds"
            if ok
            else "object is outside tray bounds or not resting at tray height"
        ),
    }


def verify_lift(start_position, current_position, minimum_height=0.025):
    return float(current_position[2]) - float(start_position[2]) >= minimum_height
