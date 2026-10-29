"""Runtime robot profile discovery."""

from fastapi import APIRouter

from app.robots.registry import available_robots, load_robot_profile

router = APIRouter(prefix="/robots", tags=["robots"])


@router.get("")
def list_robots():
    return {"items": available_robots()}


@router.get("/{robot_name}")
def get_robot(robot_name: str):
    profile = load_robot_profile(robot_name)
    return {
        "id": profile["id"],
        "display_name": profile.get("display_name", profile["id"]),
        "task": profile["task"],
        "arm_joint_names": profile["arm_joint_names"],
        "end_effector_body": profile["end_effector_body"],
    }
