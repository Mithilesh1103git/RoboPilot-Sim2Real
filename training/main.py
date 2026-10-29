"""Standalone training API and dashboard. No imports from the runtime app."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from training.api.schemas import (
    ArtifactImportRequest,
    EvaluationRequest,
    RLTrainRequest,
    SyntheticRequest,
    WorldModelRequest,
)
from training.robots.registry import available_robots, load_robot_profile
from training.services.job_manager import PROJECT_ROOT, TRAINING_ROOT, manager

WEB_ROOT = TRAINING_ROOT / "web"
DATASETS = TRAINING_ROOT / "datasets"
RUNS = TRAINING_ROOT / "runs"
CHECKPOINTS = TRAINING_ROOT / "rl" / "checkpoints"
for directory in (DATASETS, RUNS, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="RoboPilot Sim2Real Training Lab",
    version="1.0.0",
    description="Independent training API and experiment dashboard.",
)
app.mount("/static", StaticFiles(directory=str(WEB_ROOT)), name="training-static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "application": "training",
        "boundary": "independent-from-runtime",
    }


@app.get("/api/robots")
def robots() -> dict[str, Any]:
    return {"items": available_robots()}


@app.get("/api/robots/{robot_name}")
def robot_detail(robot_name: str) -> dict[str, Any]:
    profile = load_robot_profile(robot_name)
    return {
        "id": profile["id"],
        "display_name": profile.get("display_name", profile["id"]),
        "arm_joint_names": profile["arm_joint_names"],
        "end_effector_body": profile["end_effector_body"],
        "task": profile["task"],
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    jobs = manager.list()
    return {
        "active_jobs": sum(
            j["status"] in {"queued", "starting", "running", "stopping"} for j in jobs
        ),
        "total_jobs": len(jobs),
        "datasets": _list_files(
            DATASETS, {".npz", ".npy", ".json", ".csv", ".parquet"}
        ),
        "checkpoints": _list_files(RUNS, {".zip", ".pt", ".pth", ".onnx"})
        + _list_files(CHECKPOINTS, {".zip", ".pt", ".pth", ".onnx"}),
    }


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"items": manager.list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    logs = manager.logs(job_id)
    return job | {"logs": logs["lines"] if logs else []}


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str, offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    result = manager.logs(job_id, offset)
    if result is None:
        raise HTTPException(404, "Training job not found")
    return result


@app.post("/api/jobs/rl-train", status_code=202)
def start_rl_train(request: RLTrainRequest) -> dict[str, Any]:
    payload = request.model_dump()
    try:
        load_robot_profile(payload["robot_name"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if payload["output"] == "training/runs/panda_ppo":
        payload["output"] = (
            f"training/runs/{payload['robot_name']}_{payload['algorithm'].lower()}"
        )
    payload["output"] = _validate_training_path(payload["output"], (RUNS,), "output")
    return manager.start("rl_train", payload)


@app.post("/api/jobs/synthetic-collect", status_code=202)
def start_synthetic(request: SyntheticRequest) -> dict[str, Any]:
    payload = request.model_dump()
    try:
        load_robot_profile(payload["robot_name"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if (
        payload["robot_name"] != "panda"
        and payload["output"] == "training/datasets/transitions.npz"
    ):
        payload["output"] = f"training/datasets/{payload['robot_name']}_transitions.npz"
    payload["output"] = _validate_training_path(
        payload["output"], (DATASETS,), "output"
    )
    return manager.start("synthetic_collect", payload)


@app.post("/api/jobs/evaluate", status_code=202)
def start_evaluation(request: EvaluationRequest) -> dict[str, Any]:
    try:
        load_robot_profile(request.robot_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _require_training_file(request.checkpoint, (RUNS, CHECKPOINTS), "checkpoint")
    return manager.start("evaluate", request.model_dump())


@app.post("/api/jobs/world-model-train", status_code=202)
def start_world_model(request: WorldModelRequest) -> dict[str, Any]:
    _require_training_file(request.dataset, (DATASETS,), "dataset")
    payload = request.model_dump()
    payload["output"] = _validate_training_path(payload["output"], (RUNS,), "output")
    return manager.start("world_model_train", payload)


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    job = manager.stop(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    return job


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    return {
        "items": _list_files(DATASETS, {".npz", ".npy", ".json", ".csv", ".parquet"})
    }


@app.get("/api/artifacts")
def artifacts() -> dict[str, Any]:
    extensions = {".zip", ".pt", ".pth", ".onnx", ".npz"}
    items = (
        _list_files(RUNS, extensions)
        + _list_files(CHECKPOINTS, extensions)
        + _list_files(DATASETS, extensions)
    )
    unique = {item["path"]: item for item in items}
    return {"items": sorted(unique.values(), key=lambda item: item["path"])}


@app.post("/api/artifacts/import", status_code=201)
def import_artifact(request: ArtifactImportRequest) -> dict[str, Any]:
    """Explicitly copy an artifact into training-owned storage; never auto-share with runtime."""
    source = _resolve_project_path(request.source)
    if not source.is_file():
        raise HTTPException(404, "Source artifact does not exist")
    dest_dir = (
        DATASETS
        if source.suffix.lower() in {".npz", ".npy", ".csv", ".parquet", ".json"}
        else RUNS
    )
    destination = dest_dir / request.destination_name
    if destination.resolve() == source.resolve():
        raise HTTPException(400, "Source and destination are the same file")
    shutil.copy2(source, destination)
    return {
        "status": "imported",
        "source": str(source),
        "destination": destination.relative_to(PROJECT_ROOT).as_posix(),
        "note": "Copied explicitly into training-owned storage; no runtime API was called.",
    }


@app.get("/api/artifacts/download")
def download_artifact(path: str = Query(min_length=1, max_length=1000)) -> FileResponse:
    target = _resolve_project_path(path)
    allowed_roots = (DATASETS.resolve(), RUNS.resolve(), CHECKPOINTS.resolve())
    if not target.is_file() or not any(
        _is_relative_to(target.resolve(), root) for root in allowed_roots
    ):
        raise HTTPException(
            404, "Artifact not found in training-owned artifact directories"
        )
    return FileResponse(target, filename=target.name)


def _list_files(root: Path, extensions: set[str]) -> list[dict[str, Any]]:
    results = []
    if not root.exists():
        return results
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in extensions
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        ):
            results.append(
                {
                    "name": path.name,
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "modified_at": path.stat().st_mtime,
                }
            )
    return sorted(results, key=lambda item: item["path"])


def _resolve_project_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if not _is_relative_to(resolved, PROJECT_ROOT.resolve()):
        raise HTTPException(400, "Paths must stay inside the project workspace")
    return resolved


def _require_project_file(value: str, label: str) -> Path:
    path = _resolve_project_path(value)
    if not path.is_file():
        raise HTTPException(404, f"{label.capitalize()} file not found: {value}")
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_training_path(
    value: str, allowed_roots: tuple[Path, ...], label: str
) -> str:
    target = _resolve_project_path(value)
    if not any(_is_relative_to(target, root.resolve()) for root in allowed_roots):
        choices = ", ".join(
            root.relative_to(PROJECT_ROOT).as_posix() for root in allowed_roots
        )
        raise HTTPException(400, f"{label.capitalize()} path must be inside: {choices}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.relative_to(PROJECT_ROOT).as_posix()


def _require_training_file(
    value: str, allowed_roots: tuple[Path, ...], label: str
) -> Path:
    target = _require_project_file(value, label)
    if not any(
        _is_relative_to(target.resolve(), root.resolve()) for root in allowed_roots
    ):
        choices = ", ".join(
            root.relative_to(PROJECT_ROOT).as_posix() for root in allowed_roots
        )
        raise HTTPException(
            400,
            f"{label.capitalize()} must be inside training-owned storage: {choices}",
        )
    return target
