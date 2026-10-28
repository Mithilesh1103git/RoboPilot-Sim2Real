"""Configuration for the RoboPilot simulation API."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "RoboPilot-Sim2Real")
    robot_name: str = os.getenv("ROBOT_NAME", "panda").strip().lower()
    assets_dir: Path = (
        Path(
            os.getenv(
                "ROBOPILOT_ASSETS_DIR",
                os.getenv("ROBOBOT_ASSETS_DIR", str(PROJECT_ROOT / "assets")),
            )
        )
        .expanduser()
        .resolve()
    )
    model_path: str = os.getenv(
        "ROBOPILOT_MODEL_PATH", os.getenv("ROBOBOT_MODEL_PATH", "")
    ).strip()
    simulation_headless: bool = _bool("SIMULATION_HEADLESS", True)
    simulation_realtime: bool = _bool("SIMULATION_REALTIME", False)
    max_retries: int = max(0, int(os.getenv("MAX_RECOVERY_RETRIES", "1")))
    motion_steps: int = max(
        20,
        int(
            os.getenv(
                "ROBOPILOT_MOTION_STEPS", os.getenv("ROBOBOT_MOTION_STEPS", "300")
            )
        ),
    )
    ik_iterations: int = max(
        20,
        int(
            os.getenv(
                "ROBOPILOT_IK_ITERATIONS", os.getenv("ROBOBOT_IK_ITERATIONS", "120")
            )
        ),
    )
    allow_origins: tuple[str, ...] = tuple(
        x.strip()
        for x in os.getenv(
            "CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
        ).split(",")
        if x.strip()
    )


settings = Settings()
