"""Telemetry endpoints; replace demo payloads with live simulation state."""
from fastapi import APIRouter

router = APIRouter(tags=["telemetry"])


@router.get("/telemetry")
def get_telemetry():
    return {
        "simulation": {"connected": False, "time": 0.0},
        "robot": {"state": "not_connected"},
        "current_action": None,
        "events": [],
        "note": "Starter payload; connect to the simulator and event store.",
    }
