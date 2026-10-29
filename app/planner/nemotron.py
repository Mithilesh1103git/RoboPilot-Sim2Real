"""Nebius Token Factory OpenAI-compatible Nemotron inference adapter."""

from app.config import settings
from app.planner.prompts import build_planner_messages
from app.planner.validation import parse_plan


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
            max_retries=1,
        )

    def plan(self, instruction, observation):
        response = self.client.chat.completions.create(
            model=settings.nebius_model,
            messages=build_planner_messages(instruction, observation),
            temperature=settings.nebius_temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise ValueError("Nemotron returned an empty response.")
        return parse_plan(content)
