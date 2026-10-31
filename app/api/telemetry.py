"""Telemetry endpoint compatible with the current per-task simulation lifecycle."""

from fastapi import APIRouter

from app.api.tasks import _lock, _tasks
from app.settings import settings

router = APIRouter(tags=["telemetry"])


@router.get("/telemetry")
def get_telemetry():
    with _lock:
        rows = list(_tasks.values())
        active = next(
            (t for t in reversed(rows) if t["status"] in {"queued", "running"}), None
        )
        events = [
            {
                "task_id": t["task_id"],
                "status": t["status"],
                "message": t["message"],
                "created_at": t["created_at"],
            }
            for t in rows[-10:]
        ]
    return {
        "simulation": {"connected": bool(active), "time": 0.0},
        "robot": {
            "name": settings.robot_name,
            "state": "running" if active else "idle",
        },
        "current_action": active["message"] if active else None,
        "observation": None,
        "events": events,
        "note": "Each task owns its own simulation session; no persistent viewer is attached.",
    }
