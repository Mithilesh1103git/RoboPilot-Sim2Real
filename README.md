# RoboPilot

Simulation-first scaffold for a language-guided robot manipulation agent. The intended path is:

1. Make a deterministic MuJoCo pick-and-place baseline.
2. Add NVIDIA Nemotron inference through Nebius Token Factory for high-level planning.
3. Add observation-based verification and bounded recovery.
4. Compare the baseline and closed-loop agent with repeatable trials.

> **Important:** This is a starter scaffold, not a finished robot controller. Several modules intentionally contain stubs or TODOs. No robot actions are executed by the API yet. The MuJoCo scene XML and robot model are not bundled; choose a model and verify its license before adding it.

## Requirements

- Python 3.11+ recommended.
- MuJoCo-compatible rendering support for GUI/camera rendering.
- Nebius Token Factory API access for live model inference (not required to start the API).
- A compatible MuJoCo robot MJCF model, added locally under `assets/`.

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to set `NEBIUS_API_KEY` and `NEBIUS_MODEL` when you are ready to test inference. Keep secrets out of source control. Set the model ID to one actually available to your Nebius account.

## Run the API

```bash
uvicorn app.main:app --reload
```

- Health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Starter UI: http://127.0.0.1:8000/web/index.html

The starter UI is static and served by your browser. To make the `/web` path work as shown, add this to `app/main.py` before starting the app:

```python
from fastapi.staticfiles import StaticFiles
app.mount("/web", StaticFiles(directory="web"), name="web")
```

## Run tests

```bash
pytest -q
```

## Check the planner client

The planner module uses the OpenAI Python SDK against the Nebius Token Factory OpenAI-compatible endpoint. It is not called by the task API until you wire it into `TaskExecutor`.

## Suggested implementation order

1. Add and load a licensed robot MJCF plus a tabletop scene.
2. Implement joint limits, IK and trajectory execution for the selected robot.
3. Implement and test the pick/grasp/lift/place primitives without an LLM.
4. Connect the task executor to the simulator and API.
5. Add planner inference, strict validation and safe primitive dispatch.
6. Add simulator-grounded verification and bounded recovery.
7. Run paired baseline/agent trials and export measured metrics.
8. Connect live camera frames or render snapshots to the dashboard.

## Architecture

```text
Natural-language instruction
          |
          v
FastAPI task API ---> Nemotron via Nebius Token Factory
          |                       |
          v                       v
Task executor <------------ validated high-level plan
          |
          v
Safety checks -> motion primitives -> MuJoCo
          ^                         |
          |                         v
          +---- observations / verifier / recovery
```

## Evaluation principles

- Use identical initial conditions for baseline and agent comparisons.
- Record scene seed, instruction, model ID, action trace, terminal state and timing.
- Do not report placeholder trials as real results.
- Treat simulator state as benchmark ground truth; do not rely on the language model to certify its own success.
- Keep the planner out of direct low-level motor/joint command generation.

## Security and licensing

- Never commit `.env` or API keys.
- Validate model output and enforce action/joint limits before execution.
- Review the license and attribution requirements for every robot asset, model checkpoint and dataset you include.

## Updated simulation application and web UI

The `app/` package and `web/` UI are integrated for the deterministic Panda pick-and-place baseline. Planner credentials are not required for the default task path. Start from the repository root:

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the UI or `http://127.0.0.1:8000/docs` for API documentation. The UI polls task status and telemetry. The simulation requires MuJoCo and the included Panda assets; successful physical manipulation must be confirmed by running the task in your local MuJoCo environment.

The bundle intentionally excludes `.env`, `.git`, IDE workspace files, Python bytecode, and cache directories. Copy any required local environment values from `.env.example` into your own untracked `.env` file.

## Data Adapter & Task Interface (Stage 2)

The `app/data_adapter/` package provides reusable ingestion, normalization, validation, storage, and export logic for JSON/JSONL-compatible VR4IL, LIBERO-derived, or other demonstration records. HTTP endpoints are centralized under `app/api/`: dataset ingestion/listing/export live in `app/api/datasets.py`, while task creation (including tasks sourced from a dataset episode) lives in `app/api/tasks.py`. Records are normalized into the versioned `robopilot.demonstrations` schema, grouped into episodes, validated, and exported as JSON/JSONL.

- `GET /api/v1/datasets/schema`
- `POST /api/v1/datasets/ingest`
- `GET /api/v1/datasets`
- `GET /api/v1/datasets/{dataset_id}`
- `GET /api/v1/datasets/{dataset_id}/export?format=jsonl`
- `POST /api/v1/tasks/from-dataset` (body includes `dataset_id` and optional `episode_id`, `instruction`, and `scenario`)

See `app/data_adapter/README.md` for the schema, payload example, limitations, and API behavior. Binary image/video payloads are not embedded; store those in object storage and pass references in metadata. The adapter does not blindly replay actions from another robot embodiment: target-robot mapping and validation are prerequisites for safe replay.
