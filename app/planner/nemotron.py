"""Nebius Token Factory OpenAI-compatible Nemotron inference adapter."""

import logging
import time

from app.config import settings
from app.planner.prompts import build_planner_messages
from app.planner.validation import parse_plan

logger = logging.getLogger(__name__)

class PlannerConfigurationError(RuntimeError):
    pass


class NemotronPlanner:
    def __init__(self):
        if not settings.nebius_api_key:
            raise PlannerConfigurationError(
                "NEBIUS_API_KEY is not set; configure it in your local .env."
            )

        if not settings.nebius_model:
            raise PlannerConfigurationError(
                "NEBIUS_MODEL is not set; use a model ID enabled for your Nebius account."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PlannerConfigurationError(
                "The openai package is missing; install requirements.txt."
            ) from exc

        self.client = OpenAI(
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
            timeout=settings.nebius_timeout_seconds,
            max_retries=settings.max_retries,
        )

    def plan(self, instruction, observation, profile=None):
        started = time.perf_counter()

        logger.info(
            "Nemotron request started: model=%s, timeout=%ss",
            settings.nebius_model,
            settings.nebius_timeout_seconds,
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.nebius_model,
                messages=build_planner_messages(instruction, observation, profile),
                temperature=settings.nebius_temperature,
                response_format={"type": "json_object"},
            )

            logger.info(
                "Nemotron response received in %.2fs",
                time.perf_counter() - started,
            )

            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("Nemotron returned an empty response.")

            return parse_plan(content)

        except Exception:
            logger.exception(
                "Nemotron planning failed after %.2fs",
                time.perf_counter() - started,
            )
            raise