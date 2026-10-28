"""Task execution state machine. Deterministic Panda control is the default; no LLM is required."""

from __future__ import annotations

import threading
import time
from enum import StrEnum
from typing import Any

from app.agent.recovery import choose_recovery
from app.control.primitives import pick_and_place
from app.simulation.scenarios import resolve_scenario


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
    def __init__(self, environment, planner=None, max_retries: int = 0):
        self.environment = environment
        # Planner remains injectable for future use but is deliberately not
        # instantiated or called by the current deterministic baseline.
        self.planner = planner
        self.max_retries = max(0, int(max_retries))
        self.state = AgentState.IDLE
        self.execution_lock = threading.RLock()

    def run(
        self,
        instruction: str,
        n: int | None = None,
        stop_event=None,
        scenario: str = "default",
        seed: int = 7,
    ) -> dict[str, Any]:
        """Run one deterministic red-block-to-blue-tray task and verify it.

        `n` is accepted for backward compatibility with the old smoke-test API;
        it no longer means "only step physics n times".
        """
        started = time.perf_counter()
        result: dict[str, Any] = {
            "success": False,
            "state": AgentState.IDLE.value,
            "attempts": 0,
            "instruction": instruction,
            "scenario": scenario,
            "seed": seed,
            "attempt_history": [],
            "message": "Task has not started.",
        }
        normalized = (instruction or "").lower()
        try:
            scenario_config = resolve_scenario(scenario, seed)
        except ValueError as exc:
            self.state = AgentState.FAILED
            result.update(
                state=self.state.value,
                message=str(exc),
                duration_seconds=time.perf_counter() - started,
            )
            return result
        result["scenario"] = scenario_config.name
        result["scenario_config"] = scenario_config.to_dict()
        if not any(
            word in normalized for word in ("pick", "place", "block", "cube", "object")
        ):
            self.state = AgentState.FAILED
            result.update(
                state=self.state.value,
                message="This deterministic controller supports pick-and-place instructions for the red block and blue tray.",
                duration_seconds=time.perf_counter() - started,
            )
            return result

        with self.execution_lock:
            max_attempts = self.max_retries + 1
            for attempt in range(1, max_attempts + 1):
                if stop_event is not None and stop_event.is_set():
                    self.state = AgentState.STOPPED
                    result.update(
                        state=self.state.value,
                        message="Task stopped before execution.",
                        attempts=attempt - 1,
                    )
                    break
                result["attempts"] = attempt
                attempt_result: dict[str, Any] = {"attempt": attempt, "success": False}
                try:
                    self.state = AgentState.OBSERVING
                    self.environment.load()
                    attempt_result["pre_observation"] = self.environment.observation()
                    self.state = AgentState.EXECUTING
                    outcome = pick_and_place(
                        self.environment,
                        scenario=scenario_config,
                        stop_event=stop_event,
                    )
                    self.state = AgentState.VERIFYING
                    if outcome.get("success") is True:
                        self.state = AgentState.SUCCEEDED
                        attempt_result.update(success=True, result=outcome)
                        result.update(
                            success=True,
                            state=self.state.value,
                            message="Pick-and-place completed and verified in MuJoCo.",
                            result=outcome,
                        )
                        result["attempt_history"].append(attempt_result)
                        break
                    raise RuntimeError("Task verifier did not confirm placement.")
                except InterruptedError as exc:
                    self.state = AgentState.STOPPED
                    attempt_result["message"] = str(exc)
                    result["attempt_history"].append(attempt_result)
                    result.update(state=self.state.value, message=str(exc))
                    break
                except Exception as exc:
                    attempt_result["message"] = f"{type(exc).__name__}: {exc}"
                    attempt_result["post_observation"] = self.environment.observation()
                    result["attempt_history"].append(attempt_result)
                    if attempt >= max_attempts:
                        self.state = AgentState.FAILED
                        result.update(
                            state=self.state.value,
                            message=f"Task failed: {exc}",
                            error=str(exc),
                        )
                        break
                    self.state = AgentState.RECOVERING
                    decision = choose_recovery(
                        attempt, self.max_retries, "missed_grasp"
                    )
                    if decision != "reset_and_retry":
                        self.state = AgentState.FAILED
                        result.update(
                            state=self.state.value,
                            message=f"Task failed; recovery stopped: {exc}",
                            error=str(exc),
                        )
                        break
            result["duration_seconds"] = round(time.perf_counter() - started, 4)
            result["state"] = self.state.value
            result["final_observation"] = self.environment.observation()
            return result
