"""Task state machine scaffold."""
from enum import StrEnum


class AgentState(StrEnum):
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskExecutor:
    def __init__(self, environment, planner, max_retries: int = 2):
        self.environment = environment
        self.planner = planner
        self.max_retries = max_retries
        self.state = AgentState.IDLE

    def run(self, instruction: str):
        """TODO: connect observations, planning, primitives, verification and recovery."""
        raise NotImplementedError("Implement the task loop after baseline manipulation works.")
