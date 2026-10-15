"""Task outcome verification. Use simulator state for benchmark ground truth."""


def verify_object_in_destination(
    object_position, destination_bounds, tolerance: float = 0.02
) -> bool:
    """Simple axis-aligned destination check; adapt bounds to your scene convention."""
    x, y, z = object_position
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = destination_bounds
    return (
        xmin - tolerance <= x <= xmax + tolerance
        and ymin - tolerance <= y <= ymax + tolerance
        and zmin - tolerance <= z <= zmax + tolerance
    )
