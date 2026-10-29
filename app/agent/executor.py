"""Planner-in-the-loop execution state machine."""

import threading
import time
from enum import StrEnum

from app.agent.recovery import choose_recovery
from app.control.primitives import pick_and_place
from app.planner.validation import validate_plan


class AgentState(StrEnum):
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    VALIDATING = "validating"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskExecutor:
    def __init__(self, environment, planner, max_retries=1):
        self.environment = environment
        self.planner = planner
        self.max_retries = max(0, min(int(max_retries), 3))
        self.state = AgentState.IDLE
        self._lock = threading.RLock()

    def run(self, instruction, *, stop_event=None, scenario="panda_pick_place"):
        started = time.perf_counter()
        result = {
            "success": False,
            "state": self.state.value,
            "instruction": instruction,
            "scenario": scenario,
            "attempts": 0,
            "attempt_history": [],
        }
        try:
            if scenario != "panda_pick_place":
                raise ValueError(f"Unsupported scenario: {scenario}")
            self.state = AgentState.OBSERVING
            self.environment.load()
            observation = self.environment.observation()
            result["initial_observation"] = observation
            self.state = AgentState.PLANNING
            plan = self.planner.plan(instruction, observation)
            self.state = AgentState.VALIDATING
            plan = validate_plan(plan, instruction)
            result["validated_plan"] = plan.model_dump()
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Task stop requested before execution.")
            for attempt in range(self.max_retries + 1):
                result["attempts"] = attempt + 1
                entry = {"attempt": attempt + 1}
                try:
                    self.state = AgentState.EXECUTING
                    motion = pick_and_place(
                        self.environment,
                        object_name=plan.target_object,
                        destination_name=plan.destination,
                        stop_event=stop_event,
                    )
                    self.state = AgentState.VERIFYING
                    if not motion.get("success"):
                        raise RuntimeError("Simulator did not verify placement.")
                    entry.update(success=True, result=motion)
                    result["attempt_history"].append(entry)
                    self.state = AgentState.SUCCEEDED
                    result.update(
                        success=True,
                        message="Task succeeded; simulator verification passed.",
                        task_result=motion,
                    )
                    break
                except InterruptedError as exc:
                    self.state = AgentState.STOPPED
                    entry.update(success=False, error=str(exc))
                    result["attempt_history"].append(entry)
                    result.update(message=str(exc), error=str(exc))
                    break
                except Exception as exc:
                    entry.update(success=False, error=f"{type(exc).__name__}: {exc}")
                    result["attempt_history"].append(entry)
                    if (
                        choose_recovery(attempt, self.max_retries, "controller_failure")
                        != "reset_and_retry"
                    ):
                        self.state = AgentState.FAILED
                        result.update(
                            message="Task failed; bounded recovery exhausted.",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        break
                    self.state = AgentState.RECOVERING
            result["final_observation"] = self.environment.observation()
        except InterruptedError as exc:
            self.state = AgentState.STOPPED
            result.update(message=str(exc), error=str(exc))
        except Exception as exc:
            self.state = AgentState.FAILED
            result.update(
                message=f"Task failed: {exc}", error=f"{type(exc).__name__}: {exc}"
            )
        finally:
            result["duration_seconds"] = round(time.perf_counter() - started, 4)
            result["state"] = self.state.value
            self.environment.close()
        return result
