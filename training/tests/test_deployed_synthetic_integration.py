"""Focused tests for Stage 4 -> Flow Controller deployment wiring."""

import inspect

from training.deploy.schemas import StartRunRequest
from training.synthetic.generate_episodes import generate_dataset


def test_training_request_exposes_synthetic_generation_controls():
    req = StartRunRequest(
        generate_synthetic=True,
        synthetic_episodes=3,
        synthetic_env_id="Pendulum-v1",
        upload_synthetic_to_buffer=True,
    )
    assert req.generate_synthetic is True
    assert req.synthetic_episodes == 3
    assert req.upload_synthetic_to_buffer is True
    assert req.synthetic_buffer_prefix == "unified-buffer/generated"


def test_synthetic_generator_is_callable_from_flow_control():
    # Deployment calls the Python function directly; CLI is only an optional wrapper.
    assert callable(generate_dataset)
    signature = inspect.signature(generate_dataset)
    assert {"env_id", "episodes", "seed", "policy", "output"}.issubset(
        signature.parameters
    )
