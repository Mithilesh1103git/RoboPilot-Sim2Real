import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

instruction = """Use the current state of robot and find the best trajectory for object to reach the destination."""
observation = observation = {
    "robot": {
        "joint_positions": [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8],
        "end_effector_position": [0.45, 0.0, 0.30],
        "end_effector_orientation": [1.0, 0.0, 0.0, 0.0],
        "gripper_open": True,
    },
    "objects": {"red_block": {"position": [0.48, -0.2, 0.425], "grasped": False}},
    "target": {"name": "blue_tray", "position": [0.57, 0.22, 0.405]},
    "task": {
        "instruction": "Pick up the red block and place it in the blue tray",
        "completed": False,
    },
}
robot = "franka_panda"
object_name = "red_block"
destination = "blue_tray"

SYSTEM_PROMPT = f"""You are the high-level planner for a simulated {robot}.
Return only a JSON object with keys task, target_object, destination, steps, recovery.
task must be pick_and_place; target_object must be {object_name}; destination must be {destination}.
steps must contain, in this exact order: approach_object, grasp_object, lift_object, move_to_destination, release_object, verify.
Each step has action, optional target_object, optional destination, and short rationale.
recovery must be stop_safely.
Do not output joint angles, motor commands, velocities, arbitrary coordinates, code, or unlisted actions.
The backend validates the plan and executes a bounded robot-specific controller.
The simulator, not you, decides whether the task succeeded."""

USER_MESSAGE = json.dumps(
    {
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
    },
    ensure_ascii=False,
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_MESSAGE},
]

client = OpenAI(
    base_url="https://api.tokenfactory.eu-north1.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    messages=messages,
    temperature=0.1,
    response_format={"type": "json_object"},
)

print(response.to_json())
