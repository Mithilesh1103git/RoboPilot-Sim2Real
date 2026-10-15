"""Prompt templates for constrained high-level task planning."""

import json

SYSTEM_PROMPT = """You are a high-level planner for a simulated robot arm.
Return only a JSON object matching this schema:
{
  "task": "pick_and_place",
  "target_object": "object name",
  "destination": "destination name",
  "steps": [
    {"action": "approach_object|grasp_object|lift_object|move_to_destination|release_object|verify",
     "target_object": "optional name", "destination": "optional name", "rationale": "short string"}
  ],
  "recovery": "reobserve_and_replan|stop_safely"
}
Do not output joint angles, motor commands, velocities, or arbitrary code.
Use only the listed action names. If the scene does not support the request, choose stop_safely.
"""


def build_planner_messages(instruction: str, observation: dict) -> list[dict[str, str]]:
    user_content = {
        "instruction": instruction,
        "observation": observation,
        "constraints": [
            "Plan high-level actions only.",
            "The backend validates all actions before execution.",
            "Never assume the task succeeded without verification.",
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content)},
    ]
