"""In-memory task API with cooperative stop requests."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.agent.executor import TaskExecutor
from app.config import settings
from app.planner.nemotron import NemotronPlanner
from app.schemas import TaskRequest, TaskStatus

router = APIRouter(prefix="/tasks", tags=["tasks"])
_lock = threading.RLock()
_tasks = {}
_stop_events = {}
_pool = ThreadPoolExecutor(
    max_workers=settings.task_workers, thread_name_prefix="robopilot-task"
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _run_task(task_id, request):
    with _lock:
        if _tasks[task_id]["status"] == "stopped":
            _stop_events.pop(task_id, None)
            return
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["started_at"] = _now()
    try:
        planner = NemotronPlanner()
        from app.simulation.environment import SimulationEnvironment

        result = TaskExecutor(
            SimulationEnvironment(), planner, settings.max_retries
        ).run(
            request.instruction,
            stop_event=_stop_events[task_id],
            scenario=request.scenario,
        )
        status = {"succeeded": "succeeded", "stopped": "stopped"}.get(
            result.get("state"), "failed"
        )
        with _lock:
            _tasks[task_id].update(
                status=status,
                message=result.get("message", ""),
                result=result,
                error=result.get("error"),
                finished_at=_now(),
            )
    except Exception as exc:
        with _lock:
            _tasks[task_id].update(
                status="failed",
                message="Task failed before execution.",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_now(),
            )
    finally:
        with _lock:
            _stop_events.pop(task_id, None)


@router.post("", response_model=TaskStatus, status_code=202)
def create_task(request: TaskRequest):
    task_id = str(uuid.uuid4())
    record = {
        "task_id": task_id,
        "status": "queued",
        "instruction": request.instruction,
        "scenario": request.scenario,
        "message": "Queued for execution.",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    with _lock:
        _tasks[task_id] = record
        _stop_events[task_id] = threading.Event()
    _pool.submit(_run_task, task_id, request)
    return TaskStatus(**record)


@router.get("/{task_id}", response_model=TaskStatus)
def get_task(task_id):
    with _lock:
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(404, "Task not found.")
        return TaskStatus(**record)


@router.post("/{task_id}/stop", response_model=TaskStatus)
def stop_task(task_id):
    with _lock:
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(404, "Task not found.")
        if record["status"] in {"succeeded", "failed", "stopped"}:
            return TaskStatus(**record)
        event = _stop_events.get(task_id)
        if event:
            event.set()
        if record["status"] == "queued":
            record.update(
                status="stopped",
                message="Stopped before execution.",
                finished_at=_now(),
            )
        else:
            record["message"] = "Cooperative stop requested."
        return TaskStatus(**record)
