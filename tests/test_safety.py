import numpy as np
import pytest

from app.control.safety import validate_joint_targets


def test_safe_joint_target():
    assert np.allclose(validate_joint_targets([0.1, 0.2], [-1, -1], [1, 1]), [0.1, 0.2])


def test_rejects_nan():
    with pytest.raises(ValueError):
        validate_joint_targets([float("nan")], [-1], [1])


def test_rejects_limit_violation():
    with pytest.raises(ValueError):
        validate_joint_targets([2], [-1], [1])
