# Training deployment options

There are two distinct deployment entry points in this folder:

## Standalone training dashboard + API

This is the web application implemented by `training.main:app`:

```bash
uvicorn training.main:app --host 127.0.0.1 --port 8010
```

For a container deployment from the repository root:

```bash
docker compose -f training/deploy/docker-compose.dashboard.yml up --build
```

The dashboard is bound to localhost by default. `TRAINING_PORT` can change the host port. The compose file persists training-owned datasets, run outputs, and RL checkpoints in the repository. This prototype has no user authentication; do not expose it publicly without adding authentication and hardening.

## Object-storage / replay-buffer worker

The existing `Dockerfile` and `docker-compose.yml` run `training.deploy.service:app`, a separate object-storage/replay-buffer ingestion service with its own environment variables. It is not the browser dashboard and does not replace `training.main:app`. Configure its `.env` and storage provider independently if that optional worker is needed.

Both services are training-side components. Neither imports the runtime application or calls its API.
