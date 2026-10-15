"""Observation extraction helpers."""
from typing import Any


def state_observation(environment: Any) -> dict:
    """Return low-level simulator state. Add camera rendering/object poses later."""
    return environment.observation()


def camera_observation(environment: Any, camera_name: str = "overview", width: int = 640, height: int = 480):
    """Render an RGB frame; requires a working MuJoCo rendering backend."""
    import mujoco
    renderer = mujoco.Renderer(environment.model, height=height, width=width)
    renderer.update_scene(environment.data, camera=camera_name)
    frame = renderer.render().copy()
    renderer.close()
    return frame
