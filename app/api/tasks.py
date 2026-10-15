"""Task lifecycle API. Simulation execution is intentionally a stub initially."""

from uuid import uuid4
from fastapi import APIRouter
from app.schemas import TaskRequest, TaskStatus

router = APIRouter(tags=["tasks"])
_tasks: dict[str, dict] = {}


@router.post("/tasks", response_model=TaskStatus)
def create_task(request: TaskRequest):
    task_id = str(uuid4())
    _tasks[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "instruction": request.instruction,
        "scenario": request.scenario,
        "message": "Task accepted. Connect the simulator executor to run it.",
    }
    return TaskStatus(**_tasks[task_id])


@router.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task(task_id: str):
    task = _tasks.get(task_id)
    if task is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus(**task)


@router.post("/tasks/{task_id}/stop", response_model=TaskStatus)
def stop_task(task_id: str):
    from fastapi import HTTPException

    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task["status"] = "stopped"
    task["message"] = "Stop requested. Wire this to the simulation loop."
    return TaskStatus(**task)


@router.post("/simulation/reset")
def reset_simulation():
    # TODO: call the shared MuJoCo environment reset method.
    return {"status": "not_connected", "message": "Connect simulation/environment.py"}
