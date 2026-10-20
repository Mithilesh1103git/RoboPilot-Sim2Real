# RoboPilot-Sim2Real `/app`

This package provides the FastAPI service, reusable MuJoCo environment, a deterministic Franka Panda pick-and-place controller, task lifecycle tracking, and trial evaluation. Nemotron is optional and is not initialized or called by the default execution path.

## Requirements

- Python 3.11+ (Python 3.12 recommended)
- MuJoCo 3.x
- Dependencies from the repository `requirements.txt`
- The repository `assets/franka_emika_panda/panda.xml` and its `assets/` mesh directory

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Start the API

Run from the repository root so `assets/` resolves correctly:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## Run a task

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"Pick up the red block and place it inside the blue tray.","scenario":"default"}'
```

The endpoint returns `202 Accepted` with a task ID. Poll `GET /api/v1/tasks/{task_id}` until the state is `succeeded`, `failed`, or `stopped`. The task's `result` includes the action trace and simulated object positions.

Other endpoints:

- `GET /health`
- `GET /api/v1/tasks`
- `POST /api/v1/tasks/{task_id}/stop`
- `POST /api/v1/simulation/reset`
- `GET /api/v1/simulation/observation`
- `GET /api/v1/telemetry`

## Evaluate

From the repository root:

```bash
python -m app.evaluation.run_trials --trials 10 --output runs/trials.jsonl
```

## Environment variables

- `ROBOT_NAME=panda` (the deterministic controller currently supports Panda only)
- `ROBOPILOT_ASSETS_DIR=/absolute/path/to/assets`
- `ROBOPILOT_MODEL_PATH=/absolute/path/to/custom.xml` (optional; custom scenes must provide the expected Panda body/joint names and `red_block`/`blue_tray` objects)
- `SIMULATION_HEADLESS=true` (default; recommended for servers)
- `SIMULATION_REALTIME=false` (default)
- `MAX_RECOVERY_RETRIES=1`
- `ROBOPILOT_MOTION_STEPS=100`
- `ROBOPILOT_IK_ITERATIONS=120`

## Scope and verification

The deterministic controller attempts a physical gripper grasp and verifies the block's final simulated position in the tray. A successful HTTP request means the task was accepted, not that manipulation succeeded. A task is only reported as `succeeded` if simulator-grounded verification passes. Physics and contact behavior depend on the installed MuJoCo version and model assets; validate the first run in your local environment before relying on success metrics. UR5e control, learned policies, and VLA/QVLA inference are intentionally not part of this baseline.
