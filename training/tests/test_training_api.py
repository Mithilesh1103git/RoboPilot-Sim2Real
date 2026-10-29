from fastapi.testclient import TestClient

from training.main import app

client = TestClient(app)


def test_training_dashboard_and_health_are_available():
    response = client.get("/")
    assert response.status_code == 200
    assert "Training Lab" in response.text
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["application"] == "training"


def test_training_api_lists_jobs_and_artifacts():
    assert client.get("/api/jobs").status_code == 200
    assert client.get("/api/datasets").status_code == 200
    assert client.get("/api/artifacts").status_code == 200


def test_rl_job_request_validation():
    response = client.post(
        "/api/jobs/rl-train", json={"algorithm": "DQN", "timesteps": 10}
    )
    assert response.status_code == 422


def test_evaluation_requires_existing_checkpoint():
    response = client.post(
        "/api/jobs/evaluate", json={"checkpoint": "training/runs/not-there.zip"}
    )
    assert response.status_code == 404


def test_artifact_import_rejects_path_escape():
    response = client.post(
        "/api/artifacts/import",
        json={"source": "../../outside.npz", "destination_name": "safe.npz"},
    )
    assert response.status_code == 400


def test_rl_job_cannot_write_outside_training_runs():
    response = client.post(
        "/api/jobs/rl-train", json={"output": "/tmp/outside-checkpoint"}
    )
    assert response.status_code == 400


def test_synthetic_job_cannot_write_outside_training_datasets():
    response = client.post(
        "/api/jobs/synthetic-collect",
        json={
            "episodes": 1,
            "output": "data/adapter/standardized/should-not-write.npz",
        },
    )
    assert response.status_code == 400
