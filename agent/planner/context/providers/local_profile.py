"""Trusted local context from robot profile and current simulator observation."""

from typing import Any


class LocalProfileProvider:
    name = "local_profile"

    @staticmethod
    def retrieve(
        profile: dict | None, instruction: str, observation: dict | None
    ) -> dict[str, Any]:
        profile = profile or {}
        task_profile = profile.get("task") or {}
        robot_context = profile.get("robot_context") or {}

        robot = {
            key: value
            for key, value in {
                "id": profile.get("id"),
                "display_name": profile.get("display_name"),
                "manufacturer": robot_context.get("manufacturer"),
                "model": robot_context.get("model"),
                "robot_category": robot_context.get("robot_category"),
                "control_mode": profile.get("control_mode"),
                "end_effector_body": profile.get("end_effector_body"),
                "arm_joint_names": profile.get("arm_joint_names"),
            }.items()
            if value is not None
        }
        task = {
            "instruction": instruction,
            "object_name": task_profile.get("object_name"),
            "destination_name": task_profile.get("destination_name"),
            "task_type": "pick_and_place",
        }
        return {
            "robot": robot,
            "task": {key: value for key, value in task.items() if value is not None},
            "scene": observation or {},
        }
