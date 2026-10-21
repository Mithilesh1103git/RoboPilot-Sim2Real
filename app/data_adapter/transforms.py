"""Small, explicit coordinate-frame and embodiment mapping utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_transform(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError("coordinate_transform must be a 4x4 homogeneous matrix")
    if not np.isfinite(arr).all():
        raise ValueError("coordinate_transform must contain finite numbers")
    if not np.allclose(arr[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("coordinate_transform last row must be [0, 0, 0, 1]")
    return arr


def transform_xyz(xyz: Any, matrix: Any) -> list[float]:
    vec = np.asarray(xyz, dtype=float)
    if vec.shape != (3,) or not np.isfinite(vec).all():
        raise ValueError("position values must be three finite numbers")
    transform = validate_transform(matrix)
    out = transform @ np.array([vec[0], vec[1], vec[2], 1.0])
    return out[:3].tolist()


def transform_observation(value: Any, matrix: Any, key: str = "") -> Any:
    """Transform common position/pose fields; leave unknown vectors untouched."""
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            normalized = child_key.lower()
            if (
                normalized
                in {
                    "position",
                    "xyz",
                    "ee_position",
                    "hand_position",
                    "target_position",
                    "object_position",
                    "translation",
                }
                and isinstance(child_value, (list, tuple))
                and len(child_value) == 3
            ):
                result[child_key] = transform_xyz(child_value, matrix)
            elif (
                normalized
                in {"pose", "ee_pose", "hand_pose", "object_pose", "target_pose"}
                and isinstance(child_value, (list, tuple))
                and len(child_value) >= 3
            ):
                xyz = transform_xyz(child_value[:3], matrix)
                result[child_key] = xyz + list(child_value[3:])
            else:
                result[child_key] = transform_observation(
                    child_value, matrix, normalized
                )
        return result
    if isinstance(value, list):
        return [
            transform_observation(v, matrix, key) if isinstance(v, (dict, list)) else v
            for v in value
        ]
    return value


def remap_named_fields(value: Any, mapping: dict[str, str]) -> Any:
    """Rename named joints/actuators in dictionaries without guessing array order."""
    if isinstance(value, dict):
        return {
            mapping.get(str(k), str(k)): remap_named_fields(v, mapping)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [remap_named_fields(v, mapping) for v in value]
    return value
