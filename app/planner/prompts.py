"""Constrained planner prompts. No low-level motor commands are allowed."""

import json

SYSTEM_PROMPT = """You are the high-level planner for a simulated Franka Panda robot.
Return only a JSON object with keys task, target_object, destination, steps, recovery.
task must be pick_and_place; target_object must be red_block; destination must be blue_tray.
steps must contain, in this exact order: approach_object, grasp_object, lift_object, move_to_destination, release_object, verify.
Each step has action, optional target_object, optional destination, and short rationale.
recovery must be stop_safely.
Do not output joint angles, motor commands, velocities, arbitrary coordinates, code, or unlisted actions.
The backend validates the plan and executes a bounded controller. The simulator, not you, decides whether the task succeeded."""


def build_planner_messages(instruction, observation):
    user = {
        "instruction": instruction,
        "scene_observation": observation,
        "constraints": [
            "Use only supported scene entities.",
            "Plan high-level actions only.",
            "Never claim success without simulator verification.",
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
    ]
