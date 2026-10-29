"""Environment-backed runtime settings; no credentials are stored in source."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _bool(name, default):
    value = os.getenv(name)
    return (
        default
        if value is None
        else value.strip().lower() in {"1", "true", "yes", "on"}
    )


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "RoboPilot-Sim2Real")
    robot_name: str = os.getenv("ROBOT_NAME", "panda").strip().lower()
    assets_dir: Path = (
        Path(os.getenv("ROBOPILOT_ASSETS_DIR", str(PROJECT_ROOT / "assets" / "robots")))
        .expanduser()
        .resolve()
    )
    data_dir: Path = (
        Path(os.getenv("ROBOPILOT_DATA_DIR", str(PROJECT_ROOT / "data" / "adapter")))
        .expanduser()
        .resolve()
    )
    nebius_api_key: str = os.getenv("NEBIUS_API_KEY", "").strip()
    nebius_base_url: str = os.getenv(
        "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
    ).strip()
    nebius_model: str = os.getenv("NEBIUS_MODEL", "").strip()
    nebius_timeout_seconds: float = max(
        5.0, float(os.getenv("NEBIUS_TIMEOUT_SECONDS", "45"))
    )
    nebius_temperature: float = min(
        1.0, max(0.0, float(os.getenv("NEBIUS_TEMPERATURE", "0.1")))
    )
    simulation_headless: bool = _bool("SIMULATION_HEADLESS", True)
    simulation_realtime: bool = _bool("SIMULATION_REALTIME", False)
    max_retries: int = max(0, min(3, int(os.getenv("MAX_RECOVERY_RETRIES", "1"))))
    task_workers: int = max(1, min(4, int(os.getenv("TASK_WORKERS", "1"))))
    allow_origins: tuple = tuple(
        x.strip()
        for x in os.getenv(
            "CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
        ).split(",")
        if x.strip()
    )


settings = Settings()
