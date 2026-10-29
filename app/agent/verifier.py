"""Simulator-grounded verification; LLM statements are never treated as evidence."""


def verify_object_in_destination(
    object_position,
    destination_position,
    *,
    half_width=0.105,
    min_z=0.405,
    max_z=0.49,
    tolerance=None
):
    x, y, z = (float(v) for v in object_position)
    tx, ty, tz = (float(v) for v in destination_position)
    if tolerance is not None:
        tol = list(tolerance)
        half_x = float(tol[0])
        half_y = float(tol[1])
        min_z = tz - float(tol[2])
        max_z = tz + float(tol[2])
    else:
        half_x = half_y = float(half_width)
    ok = abs(x - tx) <= half_x and abs(y - ty) <= half_y and min_z <= z <= max_z
    return {
        "success": ok,
        "object_position": [x, y, z],
        "destination_position": [tx, ty, tz],
        "reason": (
            "object is inside destination bounds"
            if ok
            else "object is outside destination bounds or not at acceptable height"
        ),
    }


def verify_lift(
    start_position, current_position, minimum_height=0.025, *, min_height=None
):
    threshold = minimum_height if min_height is None else min_height
    return float(current_position[2]) - float(start_position[2]) >= float(threshold)
