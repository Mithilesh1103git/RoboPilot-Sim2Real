"""High-level manipulation primitives. Implement against your selected robot model."""


def approach_object(target_xyz: tuple[float, float, float]) -> dict:
    return {
        "action": "approach_object",
        "target_xyz": list(target_xyz),
        "implemented": False,
    }


def grasp_object(object_name: str) -> dict:
    return {"action": "grasp_object", "object": object_name, "implemented": False}


def lift_object(height: float = 0.10) -> dict:
    return {"action": "lift_object", "height": height, "implemented": False}


def move_to_destination(destination_xyz: tuple[float, float, float]) -> dict:
    return {
        "action": "move_to_destination",
        "destination_xyz": list(destination_xyz),
        "implemented": False,
    }


def release_object() -> dict:
    return {"action": "release_object", "implemented": False}
