"""Robot-profile-aware high-level planner prompts; never request low-level motor commands."""

import json


def build_planner_messages(instruction, observation, profile=None):
    profile = profile or {}
    task = profile.get("task", {})
    object_name = task.get("object_name", "red_block")
    destination = task.get("destination_name", "blue_tray")
    robot = profile.get("display_name", profile.get("id", "configured robot"))
    system = f"""You are the high-level planner for a simulated {robot}.

    Return only a valid JSON object with exactly these top-level keys:
    task, target_object, destination, steps, recovery.

    Required top-level values:
    - task MUST be "pick_and_place".
    - target_object MUST be "{object_name}".
    - destination MUST be "{destination}".
    - recovery MUST be "stop_safely".

    The steps array MUST contain exactly six steps in this exact order:
    1. "approach_object"
    2. "grasp_object"
    3. "lift_object"
    4. "move_to_destination"
    5. "release_object"
    6. "verify"

    Every step MUST contain an "action" field whose value exactly matches
    the corresponding action string in the list above.
    Do not abbreviate, rename, or paraphrase action values.

    Required fields for each step:
    - approach_object:
      - target_object MUST be "{object_name}".
    - grasp_object:
      - target_object MUST be "{object_name}".
    - lift_object:
      - target_object MUST be "{object_name}".
    - move_to_destination:
      - target_object MUST be "{object_name}".
      - destination MUST be "{destination}".
    - release_object:
      - target_object MUST be "{object_name}".
      - destination MUST be "{destination}".
    - verify:
      - target_object MUST be "{object_name}".
      - destination MUST be "{destination}".

    Every step MUST include a short "rationale" string.
    Do not omit any required target_object or destination fields.

    Do not output joint angles, motor commands, velocities, arbitrary
    coordinates, code, or unlisted actions.
    Do not invent scene entities or unsupported actions.
    Do not claim the task succeeded before simulator verification.

    The backend validates this plan and executes a bounded,
    robot-specific controller. The simulator determines whether
    the task actually succeeded.

    Return JSON only. Do not include Markdown fences or explanatory
    text outside the JSON object."""
    user = {
        "instruction": instruction,
        "scene_observation": observation,
        "robot": robot,
        "supported_object": object_name,
        "supported_destination": destination,
        "constraints": [
            "Use only supported scene entities.",
            "Plan high-level actions only.",
            "Never claim success without simulator verification.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
    ]
