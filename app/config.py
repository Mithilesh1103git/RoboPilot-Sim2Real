"""Environment-based configuration."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "RoboPilot")
    nebius_base_url: str = os.getenv(
        "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
    )
    nebius_api_key: str = os.getenv("NEBIUS_API_KEY", "")
    nebius_model: str = os.getenv("NEBIUS_MODEL", "")
    simulation_headless: bool = os.getenv("SIMULATION_HEADLESS", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    max_retries: int = int(os.getenv("MAX_RECOVERY_RETRIES", "2"))


settings = Settings()
