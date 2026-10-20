"""Container-deployable training orchestration service.

Polls S3-compatible object storage for unified-buffer NPZ transition datasets,
runs the existing WM/RL pipeline, and writes checkpoints/reports back to storage.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException

from .schemas import StartRunRequest
from .storage import ObjectStorage

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("robopilot.training")
app = FastAPI(
    title="RoboPilot-Sim2Real Training Service",
    version="1.0.0",
    description="GPU-capable container service for unified-buffer, World Model and RL training.",
)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_storage: ObjectStorage | None = None
_worker_started = False
_seen_keys: set[str] = set()


def storage():
    global _storage
    if _storage is None:
        _storage = ObjectStorage()
    return _storage


def _set_job(run_id, **updates):
    with _lock:
        _jobs.setdefault(run_id, {"run_id": run_id, "status": "queued"})
        _jobs[run_id].update(updates)


def _execute_run(run_id: str, req: StartRunRequest):
    from training.rl.flow_control import run_pipeline

    run_dir = Path(tempfile.mkdtemp(prefix=f"robopilot-{run_id}-"))
    try:
        _set_job(run_id, status="running", started_at=time.time(), error=None)
        obj = storage()
        obj.ensure_bucket_access()
        keys = sorted(
            k for k in obj.list_keys(req.dataset_prefix) if k.endswith(".npz")
        )
        if not keys:
            raise RuntimeError(
                f"No .npz transition datasets found under s3://{obj.bucket}/{req.dataset_prefix}"
            )
        local_datasets = []
        for i, key in enumerate(keys):
            local_datasets.append(
                str(obj.download(key, run_dir / "datasets" / f"{i:05d}.npz"))
            )
        output_dir = run_dir / "output"
        args = SimpleNamespace(
            datasets=local_datasets,
            output_dir=str(output_dir),
            replay_capacity=req.replay_capacity,
            world_model_every=req.world_model_every,
            rl_every=req.rl_every,
            max_queue_size=req.max_queue_size,
            world_model_epochs=req.world_model_epochs,
            device=req.device,
            env_id=req.env_id or "",
            algorithm=req.algorithm,
            timesteps=req.timesteps,
        )
        run_pipeline(args)
        # Upload all generated outputs, including model checkpoints and reports.
        uploaded = []
        for path in output_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(output_dir).as_posix()
                key = f"{req.artifact_prefix.rstrip('/')}/{run_id}/{rel}"
                obj.upload(
                    path,
                    key,
                    (
                        "application/json"
                        if path.suffix == ".json"
                        else "application/octet-stream"
                    ),
                )
                uploaded.append(key)
        manifest = {
            "run_id": run_id,
            "status": "succeeded",
            "input_keys": keys,
            "artifacts": uploaded,
            "completed_at": time.time(),
            "device_requested": req.device,
        }
        obj.upload_json(
            f"{req.artifact_prefix.rstrip('/')}/{run_id}/run-manifest.json", manifest
        )
        _set_job(
            run_id,
            status="succeeded",
            input_keys=keys,
            artifacts=uploaded,
            completed_at=manifest["completed_at"],
            manifest_key=f"{req.artifact_prefix.rstrip('/')}/{run_id}/run-manifest.json",
        )
    except Exception as exc:
        log.exception("Training run %s failed", run_id)
        _set_job(run_id, status="failed", error=str(exc), completed_at=time.time())
        try:
            storage().upload_json(
                f"{req.artifact_prefix.rstrip('/')}/{run_id}/run-manifest.json",
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": time.time(),
                },
            )
        except Exception:
            log.exception("Could not upload failure manifest for %s", run_id)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _launch(req: StartRunRequest):
    run_id = req.run_id or str(uuid.uuid4())
    with _lock:
        if run_id in _jobs and _jobs[run_id].get("status") in {"queued", "running"}:
            raise HTTPException(409, f"Run {run_id} is already active")
        _jobs[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "submitted_at": time.time(),
        }
    threading.Thread(target=_execute_run, args=(run_id, req), daemon=True).start()
    return {"run_id": run_id, "status": "queued"}


@app.get("/health")
def health():
    import torch

    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/runs", status_code=202)
def start_run(req: StartRunRequest):
    return _launch(req)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    with _lock:
        job = _jobs.get(run_id)
    if not job:
        raise HTTPException(404, "Training run not found on this service instance")
    return job


@app.get("/storage/buffer")
def inspect_buffer(prefix: str = "unified-buffer/"):
    try:
        keys = list(storage().list_keys(prefix))
        return {
            "bucket": storage().bucket,
            "prefix": prefix,
            "datasets": [k for k in keys if k.endswith(".npz")],
            "object_count": len(keys),
        }
    except Exception as exc:
        raise HTTPException(503, f"Object storage unavailable: {exc}") from exc


def _buffer_poll_worker():
    """Optional ingestion watcher: starts runs when unseen NPZ datasets arrive."""
    prefix = os.getenv("BUFFER_WATCH_PREFIX", "")
    if not prefix:
        return
    poll_seconds = max(10, int(os.getenv("BUFFER_POLL_SECONDS", "60")))
    while True:
        try:
            keys = [k for k in storage().list_keys(prefix) if k.endswith(".npz")]
            new_keys = [k for k in keys if k not in _seen_keys]
            # Ingestion is intentionally observable; run submission remains explicit
            # to avoid accidentally retraining on every historical object.
            if new_keys:
                log.info(
                    "Unified buffer has %d new dataset objects: %s",
                    len(new_keys),
                    new_keys,
                )
                _seen_keys.update(new_keys)
        except Exception:
            log.exception("Unified buffer polling failed")
        time.sleep(poll_seconds)


@app.on_event("startup")
def startup():
    global _worker_started
    if os.getenv("BUFFER_WATCH_PREFIX") and not _worker_started:
        threading.Thread(target=_buffer_poll_worker, daemon=True).start()
        _worker_started = True
