"""Test-only fake planner; never selected automatically in production."""

from app.schemas import TaskPlan


class FakePlanner:
    def __init__(self, plan=None):
        self._plan = plan or TaskPlan.model_validate(
            {
                "task": "pick_and_place",
                "target_object": "red_block",
                "destination": "blue_tray",
                "steps": [
                    {"action": "approach_object", "target_object": "red_block"},
                    {"action": "grasp_object", "target_object": "red_block"},
                    {"action": "lift_object", "target_object": "red_block"},
                    {
                        "action": "move_to_destination",
                        "target_object": "red_block",
                        "destination": "blue_tray",
                    },
                    {
                        "action": "release_object",
                        "target_object": "red_block",
                        "destination": "blue_tray",
                    },
                    {
                        "action": "verify",
                        "target_object": "red_block",
                        "destination": "blue_tray",
                    },
                ],
                "recovery": "stop_safely",
            }
        )
        self.calls = 0

    def plan(self, instruction, observation):
        self.calls += 1
        return self._plan
