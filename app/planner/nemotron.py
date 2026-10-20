"""Nebius Token Factory OpenAI-compatible client scaffold."""

from openai import OpenAI

from app.config import settings
from app.planner.prompts import build_planner_messages
from app.planner.validation import parse_plan


class NemotronPlanner:
    def __init__(self):
        if not settings.nebius_api_key:
            raise RuntimeError("Set NEBIUS_API_KEY in your environment.")
        if not settings.nebius_model:
            raise RuntimeError(
                "Set NEBIUS_MODEL to a model ID available to your account."
            )
        self.client = OpenAI(
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
        )

    def plan(self, instruction: str, observation: dict):
        messages = build_planner_messages(instruction, observation)
        response = self.client.chat.completions.create(
            model=settings.nebius_model,
            messages=messages,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        return parse_plan(content)
