import json

import numpy as np
import pytest

from training.rl.replay_buffer import ReplayBuffer
from training.synthetic.domain_randomization import sample_scene


def test_replay_buffer_bounded_and_shapes():
    b = ReplayBuffer(capacity=2)
    for i in range(3):
        b.add([i, i + 1], [0.2], 1.0, [i + 1, i + 2], episode_id=i)
    assert len(b) == 2
    a = b.arrays()
    assert a.observations.shape == (2, 2)
    assert a.actions.shape == (2, 1)
    assert a.episode_ids.tolist() == [1, 2]


def test_scene_randomization_is_deterministic_and_bounded():
    a = sample_scene(42)
    b = sample_scene(42)
    assert a == b
    assert -3.1416 <= a["object_yaw"] <= 3.1416
    assert 0.7 <= a["friction_scale"] <= 1.3


def test_replay_rejects_mismatched_observation_shapes():
    b = ReplayBuffer()
    with pytest.raises(ValueError):
        b.add([1, 2], [0], 0, [1])
