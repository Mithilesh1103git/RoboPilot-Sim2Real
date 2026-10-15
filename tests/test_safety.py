from app.control.safety import validate_action_name, validate_joint_targets


def test_action_allowlist():
    assert validate_action_name("grasp_object")
    assert not validate_action_name("run_arbitrary_code")


def test_joint_limits_accept_valid_targets():
    validate_joint_targets([0.0, 0.5], [-1.0, -1.0], [1.0, 1.0])


def test_joint_limits_reject_invalid_targets():
    try:
        validate_joint_targets([2.0], [-1.0], [1.0])
    except ValueError:
        return
    raise AssertionError("Expected out-of-range joint target to fail")
