import json
import logging
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_TIMEOUT_SECONDS = float(os.getenv("TAVILY_TIMEOUT_SECONDS", "15"))

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not configured.")

client = TavilyClient(api_key=TAVILY_API_KEY)


# Robot profile: ideally load this from the configured profile JSON.
robot_profile = {
    "id": "franka_panda",
    "display_name": "Franka Panda",
    "robot_context": {
        "manufacturer": "Franka Robotics",
        "model": "Franka Emika Panda",
        "preferred_domains": ["franka.de"],
    },
    "arm_joint_names": [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ],
    "end_effector_body": "hand",
    "gripper_actuators": ["actuator8"],
}


def build_robot_search_query(profile: dict) -> str:
    context = profile.get("robot_context", {})

    manufacturer = context.get("manufacturer", "")
    model = context.get("model") or profile.get("display_name", "")

    return f"""
    Find official technical documentation for the
    {manufacturer} {model} robot.

    Research:
    - Robot specifications and degrees of freedom
    - Payload capacity and reach
    - Joint configuration and joint limits
    - End-effector and gripper specifications
    - Gripper operating characteristics
    - Manufacturer-recommended manipulation capabilities

    Prioritize manufacturer documentation and technical manuals.
    """.strip()


def retrieve_robot_knowledge(profile: dict) -> dict:
    context = profile.get("robot_context", {})
    query = build_robot_search_query(profile)

    response = client.search(
        query=query,
        search_depth="advanced",
        topic="general",
        max_results=5,
        include_domains=context.get("preferred_domains") or None,
        include_answer=False,
        include_raw_content=False,
        timeout=TAVILY_TIMEOUT_SECONDS,
    )

    return {
        "robot_id": profile["id"],
        "query": query,
        "results": [
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content"),
                "score": result.get("score"),
            }
            for result in response.get("results", [])
        ],
    }


if __name__ == "__main__":
    knowledge = retrieve_robot_knowledge(robot_profile)
    print(json.dumps(knowledge, indent=2))
