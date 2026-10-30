from app.agent.executor import TaskExecutor
from app.planner.mock import FakePlanner


class FakeEnvironment:
    def __init__(self):
        self.closed = False

    def load(self):
        return self

    def observation(self):
        return {"connected": True, "red_block_position": [0.4, 0, 0.4]}

    def close(self):
        self.closed = True


def test_executor_calls_planner_before_controller(monkeypatch):
    import app.agent.executor as module

    called = {}

    def fake_pick_and_place(environment, **kwargs):
        called["goal"] = (kwargs["object_name"], kwargs["destination_name"])
        return {"success": True, "trace": []}

    monkeypatch.setattr(module, "pick_and_place", fake_pick_and_place)
    env = FakeEnvironment()
    planner = FakePlanner()
    result = TaskExecutor(env, planner, max_step_retries=0).run(
        "Pick the red block and place it in the blue tray"
    )
    assert planner.calls == 1 and called["goal"] == ("red_block", "blue_tray")
    assert result["success"] and env.closed
