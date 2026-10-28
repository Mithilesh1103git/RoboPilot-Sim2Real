import threading

import pytest
from fastapi.testclient import TestClient

from app.api import tasks
from app.main import app


@pytest.fixture
def client():
    with tasks._tasks_lock:
        tasks._tasks.clear()
    with TestClient(app) as test_client:
        yield test_client
    with tasks._tasks_lock:
        tasks._tasks.clear()


def test_task_lifecycle_reaches_succeeded(client, monkeypatch):
    def run(**kwargs):
        return {
            "success": True,
            "state": "succeeded",
            "message": "verified",
            "scenario": kwargs["scenario"],
            "seed": kwargs["seed"],
        }

    monkeypatch.setattr(tasks.task_executor, "run", run)
    response = client.post(
        "/api/v1/tasks",
        json={
            "instruction": "Pick and place the block",
            "scenario": "randomized",
            "seed": 19,
        },
    )

    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "queued"
    assert created["scenario"] == "randomized"
    assert created["seed"] == 19

    task_id = created["task_id"]
    fetched = client.get(f"/api/v1/tasks/{task_id}")
    assert fetched.status_code == 200
    completed = fetched.json()
    assert completed["status"] == "succeeded"
    assert completed["started_at"] is not None
    assert completed["finished_at"] is not None
    assert completed["result"]["state"] == "succeeded"

    listed = client.get("/api/v1/tasks")
    assert listed.status_code == 200
    assert [item["task_id"] for item in listed.json()] == [task_id]


def test_task_lifecycle_surfaces_executor_failure(client, monkeypatch):
    monkeypatch.setattr(
        tasks.task_executor,
        "run",
        lambda **kwargs: {
            "success": False,
            "state": "failed",
            "message": "IK did not converge",
            "error": "unreachable target",
        },
    )

    response = client.post(
        "/api/v1/tasks",
        json={"instruction": "Pick up the block"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"

    fetched = client.get(f"/api/v1/tasks/{response.json()['task_id']}")
    body = fetched.json()
    assert body["status"] == "failed"
    assert body["error"] == "unreachable target"
    assert body["finished_at"] is not None


def test_queued_task_can_be_stopped(client):
    task_id = "queued-task"
    with tasks._tasks_lock:
        tasks._tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "instruction": "Pick up the block",
            "scenario": "default",
            "seed": 7,
            "message": "queued",
            "created_at": tasks._now(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "stop_event": threading.Event(),
        }

    response = client.post(f"/api/v1/tasks/{task_id}/stop")

    assert response.status_code == 200
    assert response.json()["status"] == "stopped"
    assert response.json()["finished_at"] is not None


def test_task_request_rejects_invalid_seed(client):
    response = client.post(
        "/api/v1/tasks",
        json={"instruction": "Pick up the block", "seed": -1},
    )

    assert response.status_code == 422
