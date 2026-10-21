"""Task and simulation endpoints. Tasks run in the background and are pollable."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.agent.executor import TaskExecutor
from app.config import PROJECT_ROOT, settings
from app.data_adapter.service import DatasetAdapter, DatasetValidationError
from app.schemas import TaskRequest, TaskStatus
from app.simulation.environment import SimulationEnvironment

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tasks"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_model_path():
    if settings.model_path:
        return settings.model_path
    return None


environment = SimulationEnvironment(
    model_path=_resolve_model_path(),
    asset_root=settings.assets_dir,
    robot_name=settings.robot_name,
    headless=settings.simulation_headless,
    enable_local_viewer=not settings.simulation_headless,
    realtime=settings.simulation_realtime,
)
task_executor = TaskExecutor(
    environment=environment, planner=None, max_retries=settings.max_retries
)
dataset_adapter = DatasetAdapter(PROJECT_ROOT / "data" / "adapter")
_tasks: dict[str, dict] = {}
_tasks_lock = threading.RLock()


def _public_task(task: dict) -> dict:
    return {key: value for key, value in task.items() if key != "stop_event"}


@router.post("/tasks", response_model=TaskStatus, status_code=202)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid4())
    task = {
        "task_id": task_id,
        "status": "queued",
        "instruction": request.instruction,
        "scenario": request.scenario,
        "message": "Task accepted and queued for deterministic Panda pick-and-place execution.",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
        "stop_event": threading.Event(),
    }
    with _tasks_lock:
        _tasks[task_id] = task
    background_tasks.add_task(execute_task, task_id)
    return TaskStatus(**_public_task(task))


class TaskFromDatasetRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    episode_id: str | None = None
    instruction: str | None = Field(default=None, min_length=3, max_length=500)
    scenario: str = Field(default="default", min_length=1, max_length=80)


@router.post("/tasks/from-dataset", response_model=TaskStatus, status_code=202)
def create_task_from_dataset(
    request: TaskFromDatasetRequest, background_tasks: BackgroundTasks
):
    """Create a simulator task from a stored demonstration episode.

    This uses the episode's instruction as a task prompt. It does not replay the
    recorded actions; trajectory replay is a separate capability.
    """
    try:
        manifest = dataset_adapter.load_manifest(request.dataset_id)
        episodes = dataset_adapter.load_episodes(request.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    selected = next(
        (
            episode
            for episode in episodes
            if request.episode_id is None or episode.episode_id == request.episode_id
        ),
        None,
    )
    if selected is None:
        raise HTTPException(status_code=404, detail="Episode not found in dataset")

    instruction = (request.instruction or selected.instruction or "").strip()
    if len(instruction) < 3:
        raise HTTPException(
            status_code=422,
            detail="Provide an instruction or choose an episode that contains one",
        )

    task_request = TaskRequest(instruction=instruction, scenario=request.scenario)
    task_id = str(uuid4())
    task = {
        "task_id": task_id,
        "status": "queued",
        "instruction": task_request.instruction,
        "scenario": task_request.scenario,
        "message": f"Task accepted from dataset {manifest.dataset_id}, episode {selected.episode_id}.",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
        "source_dataset_id": manifest.dataset_id,
        "source_episode_id": selected.episode_id,
        "stop_event": threading.Event(),
    }
    with _tasks_lock:
        _tasks[task_id] = task
    background_tasks.add_task(execute_task, task_id)
    return TaskStatus(**_public_task(task))


@router.get("/tasks", response_model=list[TaskStatus])
def list_tasks(limit: int = 50):
    limit = max(1, min(limit, 200))
    with _tasks_lock:
        rows = list(_tasks.values())[-limit:]
    return [TaskStatus(**_public_task(row)) for row in reversed(rows)]


@router.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskStatus(**_public_task(task))


@router.post("/tasks/{task_id}/stop", response_model=TaskStatus)
def stop_task(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if task["status"] in {"succeeded", "failed", "stopped"}:
            return TaskStatus(**_public_task(task))
        task["stop_event"].set()
        if task["status"] == "queued":
            task["status"] = "stopped"
            task["message"] = "Task stopped before execution started."
            task["finished_at"] = _now()
        else:
            task["message"] = (
                "Stop requested; executor will stop at the next safe checkpoint."
            )
        return TaskStatus(**_public_task(task))


async def execute_task(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None or task["status"] == "stopped":
            return
        task["status"] = "running"
        task["started_at"] = _now()
        task["message"] = (
            "Running deterministic Panda pick-and-place; Nemotron is disabled."
        )
        instruction = task["instruction"]
        scenario = task["scenario"]
        stop_event = task["stop_event"]

    try:
        result = await asyncio.to_thread(
            task_executor.run,
            instruction=instruction,
            stop_event=stop_event,
            scenario=scenario,
        )
        with _tasks_lock:
            task = _tasks.get(task_id)
            if task is None:
                return
            task["result"] = result
            task["finished_at"] = _now()
            if result.get("state") == "stopped" or stop_event.is_set():
                task["status"] = "stopped"
                task["message"] = result.get("message", "Task stopped.")
            elif result.get("success") is True and result.get("state") == "succeeded":
                task["status"] = "succeeded"
                task["message"] = result.get(
                    "message", "Task succeeded and was verified."
                )
            else:
                task["status"] = "failed"
                task["message"] = result.get("message", "Task failed.")
                task["error"] = result.get("error") or task["message"]
    except Exception as exc:
        logger.exception("Task %s crashed", task_id)
        with _tasks_lock:
            task = _tasks.get(task_id)
            if task is not None:
                task["status"] = "failed"
                task["message"] = f"Task execution crashed: {exc}"
                task["error"] = str(exc)
                task["finished_at"] = _now()


def _reset_environment():
    with task_executor.execution_lock:
        return environment.reset()


@router.post("/simulation/reset")
async def reset_simulation():
    try:
        observation = await asyncio.to_thread(_reset_environment)
        return {
            "status": "connected",
            "message": "Environment reset successfully.",
            "observation": observation,
        }
    except Exception as exc:
        logger.exception("Simulation reset failed")
        raise HTTPException(
            status_code=503, detail=f"Simulation reset failed: {exc}"
        ) from exc


@router.get("/simulation/observation")
def observe_simulation():
    return {
        "status": "connected" if environment.model is not None else "disconnected",
        "observation": environment.observation(),
    }
