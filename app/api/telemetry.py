"""Live telemetry backed by the current MuJoCo environment and task store."""

from app.api.tasks import _tasks, _tasks_lock, environment
from app.config import settings
from fastapi import APIRouter

router = APIRouter(tags=["telemetry"])


@router.get("")
def get_telemetry():
    observation = environment.observation()
    with _tasks_lock:
        active = next(
            (
                t
                for t in reversed(list(_tasks.values()))
                if t["status"] in {"queued", "running"}
            ),
            None,
        )
        recent_events = [
            {
                "task_id": t["task_id"],
                "status": t["status"],
                "message": t["message"],
                "created_at": t["created_at"],
            }
            for t in list(_tasks.values())[-10:]
        ]
    return {
        "simulation": {
            "connected": observation.get("connected", False),
            "time": observation.get("time", 0.0),
        },
        "robot": {
            "name": settings.robot_name,
            "state": (
                "running"
                if active
                else ("ready" if observation.get("connected") else "not_connected")
            ),
        },
        "current_action": active["message"] if active else None,
        "observation": observation,
        "events": recent_events,
    }
