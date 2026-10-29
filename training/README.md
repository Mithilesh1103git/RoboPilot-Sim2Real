# RoboPilot Sim2Real — Training Lab

`training/` is an independent training application with **both a FastAPI API and a browser dashboard**. It can be launched without starting the runtime application. It does not import `app/`, call runtime APIs, or read runtime process state.

## Start the training application

Install dependencies into a separate virtual environment from the runtime environment:

```bash
python -m venv .venv-training
# Windows PowerShell: .\.venv-training\Scripts\Activate.ps1
# Linux/macOS: source .venv-training/bin/activate
python -m pip install --upgrade pip
pip install -r training/requirements-training.txt
uvicorn training.main:app --host 127.0.0.1 --port 8010 --reload
```

Open `http://127.0.0.1:8010/` for the dashboard and `http://127.0.0.1:8010/docs` for the API. Health endpoint: `GET /health`.

## Dashboard/API capabilities

- Launch PPO/SAC policy-training jobs using the training-owned MuJoCo/Gymnasium environment selected by robot profile.
- Collect simulator-generated transitions into an NPZ dataset for the selected robot profile.
- Evaluate a saved PPO/SAC checkpoint.
- Train a world model from an explicit dataset path.
- Inspect process status, logs, exit codes, and generated artifacts; request job cancellation.
- List datasets/artifacts, download training-owned artifacts, and explicitly import a project-local file into training-owned storage.

Example:

```bash
curl -X POST http://127.0.0.1:8010/api/jobs/synthetic-collect \
  -H 'Content-Type: application/json' \
  -d '{"episodes":2,"seed":7,"output":"training/datasets/transitions.npz"}'
```

Then poll `GET /api/jobs` and `GET /api/jobs/{job_id}`. `202 Accepted` means the job was submitted; it does not mean training completed or succeeded.

## Use training capabilities from Python (without CLI parsing)

Each operation exposes a reusable Python function. `argparse` is only used by the
optional command-line wrapper, so importing a module does not inspect notebook or
host-process arguments.

```python
from training.programmatic import (
    train_policy,
    evaluate_policy,
    train_world_model,
    collect_synthetic_episodes,
    augment_dataset,
    export_dataset,
    summarize_dataset,
    generate_scenes,
)

# Simulator data collection (uses the selected training-owned robot profile)
data = collect_synthetic_episodes(
    episodes=10,
    seed=7,
    output="training/datasets/transitions.npz",
)

# Train a policy; returns checkpoint metadata.
trained = train_policy(
    algorithm="PPO",
    timesteps=50_000,
    seed=7,
    output="training/runs/panda_ppo",
)

# Evaluate a saved checkpoint.
metrics = evaluate_policy(
    checkpoint=trained["checkpoint"],
    algorithm="PPO",
    episodes=10,
)

# Train the world model from a compatible transitions dataset.
world_model = train_world_model(
    dataset="training/datasets/transitions.npz",
    output="training/runs/world_model.pt",
    epochs=30,
)
```

Equivalent CLI entry points remain available, for example:

```bash
python -m training.rl.train --algorithm PPO --timesteps 50000
python -m training.synthetic.collect_episodes --episodes 10
python -m training.rl.evaluate --checkpoint training/runs/panda_ppo.zip --algorithm PPO
python -m training.rl.train_world_model --dataset training/datasets/transitions.npz --epochs 30
```

For the flow controller, use `training.programmatic.run_flow_pipeline(...)` with keyword
arguments from Python, or `python -m training.rl.flow_control ...` from a shell.
The dashboard's job manager continues to launch long-running jobs out of process so
API requests remain responsive; this is separate from whether the underlying training
capabilities are importable and callable directly.

## Architectural boundaries

- `training/` owns its environment, assets, job processes, datasets, checkpoints, and world-model/RL code.
- No imports from the runtime `app/` package and no runtime API calls.
- Artifacts are transferred only through an explicit import/export action. Import copies a project-local file to training-owned storage; it does not promote a policy to runtime.
- No LLM is used to create training demonstrations, labels, rewards, or policy updates.
- The planner is not part of the training API's RL job execution path.
- CLI scripts remain optional development/automation interfaces; they are not the only training interface.

## Limitations to understand

- Job metadata is in memory and is lost when the training API process restarts. Use one API worker for this prototype; multiple workers would each have separate in-memory job state.
- A job is a subprocess, so stop requests are best-effort process termination, not checkpoint-aware graceful interruption.
- Logs are retained in memory up to a bounded number of lines.
- PPO/SAC training is not guaranteed to solve pick-and-place. Use the evaluation job and report measured success rate; do not assume a saved checkpoint is competent.
- The world-model job requires a compatible existing dataset and model implementation.
- Artifact import is constrained to paths inside the project workspace. It is a file copy only; format validation and runtime compatibility must be checked separately before promotion.
- This prototype has no user authentication. Bind it to localhost or a trusted network; do not expose it publicly without authentication and deployment hardening.


## Selecting or adding a robot

Both the runtime and training dashboard load robot options from `configs/robots/*.json`. The default is `panda`; API requests can pass `robot_name` explicitly. For example:

```python
from training.programmatic import train_policy, collect_synthetic_episodes

collect_synthetic_episodes(robot_name="panda", episodes=10,
                           output="training/datasets/panda_transitions.npz")
result = train_policy(robot_name="panda", algorithm="PPO", timesteps=50_000,
                      output="training/runs/panda_ppo")
```

To add another robot, copy `configs/robots/custom_robot.template.json` to a new profile file, fill in the actual MJCF joint/body/actuator names and control values, then place the model and its assets in both `assets/robots/<asset_directory>/` and `training/assets/robots/<asset_directory>/`. Run `python -m training.robots.validate_profile --robot-name <robot_id>`.

The generic adapter assumes scalar position-controlled arm joints and a configured gripper. Torque/velocity actuators, mobile bases, bimanual robots, non-MuJoCo simulators, and specialized end effectors need a robot-specific controller/environment adapter. A profile makes the interface configurable; it cannot infer a robot's physical semantics from its XML alone.
