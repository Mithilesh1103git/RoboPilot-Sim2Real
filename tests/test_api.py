from fastapi.testclient import TestClient

from app.main import app


def test_health_does_not_require_api_credentials():
    response = TestClient(app).get("/health")
    assert response.status_code == 200 and "planner_configured" in response.json()


def test_rejects_extra_task_fields():
    response = TestClient(app).post(
        "/tasks", json={"instruction": "pick block", "unexpected": True}
    )
    assert response.status_code == 422
