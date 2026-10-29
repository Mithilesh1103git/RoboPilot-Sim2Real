# RoboPilot-Sim2Real v2.4 audit report

## Source archive and preservation

The v2.4 package is based on the user-provided v2.3 archive. The earlier file-level comparison against the original uploaded project is retained: all original non-secret project files identified in that audit are present. `.git/`, IDE metadata, caches/bytecode, and `.env` secrets are intentionally excluded from distributable archives.

## Added in v2.4

- Standalone training FastAPI application at `training/main.py`.
- Vanilla HTML/CSS/JavaScript dashboard at `training/web/`.
- Process-backed job manager with status, bounded logs, exit codes, stop requests, and output artifact discovery.
- API operations for PPO/SAC training, simulator transition collection, policy evaluation, world-model training, datasets, artifacts, and explicit project-local artifact import/download.
- Separate training API dependencies and startup documentation.
- Dedicated dashboard container recipe and Compose file, while retaining the existing object-storage/replay-buffer worker as a separate deployment option.
- API validation tests, including checks that training outputs remain within training-owned directories.

## Validation performed

- `python -m compileall -q app training main.py`: passed.
- `node --check training/web/app.js`: passed when Node.js is available.
- `pytest -q`: **26 passed**.
- Training API smoke checks: dashboard, health, overview, jobs, datasets, artifacts, and API docs returned HTTP 200.
- Invalid RL algorithm request returned HTTP 422.
- Training output path escape checks returned HTTP 400.
- Source scan found no training-to-runtime Python imports or runtime API URL references; the `app` mention in the training environment is explanatory documentation only.

## Important limitations

- The current validation environment does not have `mujoco`, `gymnasium`, or `stable-baselines3` installed. Therefore, this audit does **not** claim that an actual MuJoCo episode, RL training run, evaluation run, or world-model training job was executed end-to-end. Install `training/requirements-training.txt` in the training environment before those operations.
- The job manager stores job metadata and logs in memory. Restarting the API loses job history; use one API worker only. Multi-worker persistence/queueing is not implemented.
- Job cancellation terminates a subprocess on a best-effort basis and is not checkpoint-aware.
- PPO/SAC checkpoints are not assumed to solve pick-and-place; run evaluation and report measured results.
- Artifact import is a copy into training-owned storage, not policy promotion or runtime integration. Compatibility must be validated separately before a selected policy is used by runtime.
- The dashboard API has no authentication. It is intended for localhost/trusted development use only until authentication and deployment hardening are added.
- Passing unit/API tests does not prove the full runtime planner → controller → MuJoCo → verifier loop is successful. The Nebius API credential issue previously observed must be fixed in the local environment and the full runtime task tested there.

## Architectural boundaries retained

- `training/` does not import `app/` or call runtime APIs.
- Training has its own MuJoCo assets/environment and job processes.
- No LLM is used to generate training demonstrations, synthetic labels, rewards, or policy updates.
- The runtime planner produces high-level intent only; validated controller code owns bounded robot execution.
- Cross-application exchange is explicit artifact import/export only.
