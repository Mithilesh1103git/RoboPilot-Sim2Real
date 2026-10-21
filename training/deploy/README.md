# RoboPilot-Sim2Real Training Deployment

This is a deployable VM/container service for Stage 5. It is not CLI-first: the container exposes a REST API, uses an S3-compatible object store as the durable unified-buffer/artifact store, runs the existing World Model and optional RL training pipeline, and uploads checkpoints, metrics, reports, and a run manifest back to object storage. CLI modules remain available for local development.

## Architecture

1. Transition datasets (`.npz`) are placed in the configured object-storage bucket under `unified-buffer/` (or the requested `dataset_prefix`).
2. `POST /runs` submits a training run. The service downloads `.npz` objects under the selected prefix into temporary scratch space. If `generate_synthetic=true`, the Flow Controller calls `training.synthetic.generate_episodes.generate_dataset()` inside the same container and adds the generated transitions to the same replay buffer.
3. The Flow Controller builds a bounded replay buffer and orchestrates World Model training and optional PPO/SAC policy training according to configured frequencies.
4. Generated transitions, checkpoints, metrics, replay buffer and reports are uploaded under `training-artifacts/<run_id>/`. Set `upload_synthetic_to_buffer=true` to also write generated transitions to `synthetic_buffer_prefix` in the unified object-storage buffer.
5. `GET /runs/{run_id}` reports status for the current service instance; `GET /health` reports service and CUDA status; `GET /storage/buffer` inspects available dataset objects.

Object storage is the durable shared source of truth. Local disk is temporary scratch space only. Do not place credentials in source control.

## Build and run

```bash
cd training/deploy
cp .env.example .env
# Edit .env with your bucket, S3 endpoint, and credentials.
docker compose up --build -d
```

The host needs Docker Compose and NVIDIA Container Toolkit for GPU access. CPU mode can be requested per run for development; the image itself is CUDA-capable. Check `http://localhost:8080/health` and `http://localhost:8080/docs`.

## Submit a run

```bash
curl -X POST http://localhost:8080/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_prefix": "unified-buffer/",
    "artifact_prefix": "training-artifacts/",
    "world_model_epochs": 20,
    "world_model_every": 1,
    "rl_every": 1,
    "device": "auto",
    "generate_synthetic": true,
    "synthetic_env_id": "Pendulum-v1",
    "synthetic_episodes": 10,
    "synthetic_seed": 7,
    "synthetic_policy": "random",
    "upload_synthetic_to_buffer": true,
    "synthetic_buffer_prefix": "unified-buffer/generated"
  }'
```

If `env_id` is supplied, policy training is also invoked; it must be a Gymnasium environment registered in the container and wired to the robot simulator for robot-specific training. Without `env_id`, the pipeline trains the World Model and skips policy training. `generate_synthetic` invokes Stage 4 from inside the same service/Flow Controller; `Pendulum-v1` is only a smoke-test environment, not robot manipulation data. The current training implementation uses sequential WM/RL stages; a learned world model is not a substitute for MuJoCo physics.

## Object layout

```text
s3://<bucket>/
  unified-buffer/
    transitions-000001.npz
    transitions-000002.npz
  training-artifacts/
    <run-id>/
      replay_buffer.npz
      world_model.pt
      world_model_metrics.json
      flow_report.json
      run-manifest.json
      policy/...
```

Use `.npz` transition datasets with keys `observations`, `actions`, `rewards`, `next_observations`, `terminated`, `truncated`, and `episode_ids`. Dataset writers should upload complete objects atomically (write temporary key then copy/rename if supported) so training never reads partial uploads.

## API

- `GET /health`
- `GET /storage/buffer?prefix=unified-buffer/`
- `POST /runs` — queue a run and return HTTP 202; supports in-pipeline synthetic generation options
- `GET /runs/{run_id}` — current run state

## Production hardening notes

This implementation provides a container-deployable baseline, not a multi-node scheduler. Run state is in process memory; use a durable job database/queue for multi-replica deployment. The optional buffer watcher logs newly arrived objects but does not automatically retrain; this prevents duplicate training on historical data. Configure authentication/network policy before exposing the API publicly. Nebius endpoint URL and region must match the bucket's actual configuration.
