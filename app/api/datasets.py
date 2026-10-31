"""Runtime dataset adapter API: ingest, list, inspect, and export normalized demonstrations."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.settings import settings
from app.data_adapter import DatasetAdapter, DatasetValidationError
from app.data_adapter.models import IngestRequest, IngestResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _adapter():
    return DatasetAdapter(settings.data_dir)


@router.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest_dataset(request: IngestRequest):
    adapter = _adapter()
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
            coordinate_transform=request.coordinate_transform,
            embodiment_mapping=request.embodiment_mapping,
        )
        path = adapter.save(manifest, episodes, raw_payload=request.records)
        return IngestResponse(
            dataset_id=manifest.dataset_id,
            schema_name=manifest.schema_name,
            schema_version=manifest.schema_version,
            episode_count=manifest.episode_count,
            transition_count=manifest.transition_count,
            validation={"valid": True, "errors": []},
            manifest_path=str(path / "manifest.json"),
        )
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=422, detail={"message": str(exc), "errors": exc.errors}
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
def list_datasets():
    return [item.model_dump() for item in _adapter().list_datasets()]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str):
    adapter = _adapter()
    try:
        manifest = adapter.load_manifest(dataset_id)
        episodes = adapter.load_episodes(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found.") from exc
    return {
        "manifest": manifest.model_dump(),
        "episodes": [episode.model_dump() for episode in episodes],
    }


@router.get("/{dataset_id}/export")
def export_dataset(
    dataset_id: str, fmt: str = Query(default="jsonl", pattern="^(json|jsonl)$")
):
    adapter = _adapter()
    try:
        path = adapter.export(dataset_id, fmt)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found.") from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    media_type = "application/json" if fmt == "json" else "application/x-ndjson"
    return FileResponse(path, media_type=media_type, filename=path.name)
