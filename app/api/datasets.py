"""HTTP routes for dataset ingestion and task handoff.

Keep HTTP concerns here; reusable normalization/storage logic lives in
``app.data_adapter`` so scripts and training code can call it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import PROJECT_ROOT
from app.data_adapter.models import IngestRequest, IngestResponse
from app.data_adapter.service import DatasetAdapter, DatasetValidationError

router = APIRouter(prefix="/datasets", tags=["data-adapter"])
adapter = DatasetAdapter(PROJECT_ROOT / "data" / "adapter")


@router.get("/schema")
def get_schema():
    return {
        "schema_name": "robopilot.demonstrations",
        "schema_version": "1.0.0",
        "episode": {
            "episode_id": "string",
            "instruction": "string",
            "robot": "string",
            "embodiment": "string",
            "coordinate_frame": "string",
            "transitions": "non-empty list",
            "metadata": "object",
        },
        "transition": {
            "timestamp": "number",
            "observation": "object",
            "action": "JSON value",
            "instruction": "string",
            "episode_id": "string",
            "robot": "string",
            "embodiment": "string",
            "coordinate_frame": "string",
            "reward": "number|null",
            "terminated": "boolean",
            "truncated": "boolean",
            "success": "boolean|null",
        },
        "aliases": "obs/observations/state, actions/a/command, language/task_description, episode/trajectory_id/demo_id",
        "coordinate_transform": "optional 4x4 homogeneous transform applied to recognized position/pose fields",
        "embodiment_mapping": "optional source-name to target-name dictionary; named fields only, array order is never guessed",
    }


@router.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest_dataset(request: IngestRequest):
    try:
        manifest, episodes = adapter.normalize(
            request.records,
            source_format=request.source_format,
            source_name=request.source_name,
            metadata=request.metadata,
            coordinate_frame=request.coordinate_frame,
            robot=request.robot,
            embodiment=request.embodiment,
            instruction=request.instruction,
        )
        path = adapter.save(manifest, episodes, raw_payload=request.records)
        return IngestResponse(
            dataset_id=manifest.dataset_id,
            schema_name=manifest.schema_name,
            schema_version=manifest.schema_version,
            episode_count=manifest.episode_count,
            transition_count=manifest.transition_count,
            validation={
                "valid": True,
                "errors": [],
                "warnings": [
                    "Missing success labels are retained as null; they are not inferred."
                ],
            },
            manifest_path=str(path / "manifest.json"),
        )
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=422, detail={"message": str(exc), "errors": exc.errors}
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not ingest dataset: {exc}"
        ) from exc


@router.get("")
def list_datasets():
    return {"datasets": [m.model_dump() for m in adapter.list_datasets()]}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        manifest = adapter.load_manifest(dataset_id)
        episodes = adapter.load_episodes(dataset_id)
        return {
            "manifest": manifest.model_dump(),
            "episodes": [e.model_dump() for e in episodes],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc


@router.get("/{dataset_id}/export")
def export_dataset(
    dataset_id: str, format: str = Query(default="jsonl", pattern="^(json|jsonl)$")
):
    try:
        path = adapter.export(dataset_id, format)
        return FileResponse(path, filename=path.name, media_type="application/json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
