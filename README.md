# RoboPilot-Sim2Real

**Simulation-first robot manipulation with a planner-in-the-loop runtime
agent and a fully independent reinforcement-learning project.**

RoboPilot-Sim2Real combines a high-level LLM planner, a validated
robot-task plan, bounded motion primitives, a MuJoCo simulation, and
simulator-grounded verification. A separate `training/` project provides
RL training and evaluation without importing or calling the runtime
application.

> **Current development status:** The runtime can load the Franka Panda
> simulation, obtain a scene observation, call Nebius Nemotron, and
> validate the returned pick-and-place plan. End-to-end pick-and-place
> is not yet verified: the current Panda controller has reported
> `PrimitiveError: Joint trajectory did not converge`. Treat the
> controller and task execution as active development, not as a
> demonstrated successful manipulation pipeline.

## Contents

-   [Architecture and boundaries](#architecture-and-boundaries)
-   [Runtime behavior](#runtime-behavior)
-   [Requirements](#requirements)
-   [Installation](#installation)
-   [Configuration](#configuration)
-   [Run the runtime application](#run-the-runtime-application)
-   [Submit and monitor a task](#submit-and-monitor-a-task)
-   [Planner contract and validation](#planner-contract-and-validation)
-   [Independent training
    application](#independent-training-application)
-   [Custom robot profiles](#custom-robot-profiles)
-   [Troubleshooting](#troubleshooting)
-   [Testing and safety limitations](#testing-and-safety-limitations)

## Architecture and boundaries

### Runtime: `app/`

The runtime task flow is:

1.  Receive a manipulation instruction through the API.
2.  Load the selected robot profile and current MuJoCo observation.
3.  Ask Nebius Nemotron for a **high-level task plan**.
4.  Parse and validate the JSON plan, including supported action names,
    task entities, action order, and recovery policy.
5.  Execute bounded, robot-specific motion primitives.
6.  Check task outcome against simulator state.
7.  Apply bounded recovery when configured; stop safely when recovery is
    exhausted.

The planner proposes intent only. It must not supply joint angles,
velocities, motor commands, executable code, or arbitrary control
coordinates. The backend/controller remains responsible for translating
validated actions into robot motion.

### Training: `training/`

`training/` is a separate application with its own MuJoCo assets,
Gymnasium environment, data pipeline, RL training jobs, evaluation, and
checkpoints.

The separation is intentional:

-   Training does **not** import `app/`.
-   Training does **not** call the runtime API or depend on runtime
    processes.
-   Runtime and training maintain separate robot asset copies.
-   Artifact exchange is explicit; policies and checkpoints are not
    silently shared.
-   The project architecture does not use an LLM to generate training
    labels, demonstrations, rewards, or policy-learning data.

## Runtime behavior

The default/reference robot is the **Franka Emika Panda**. Robot
selection is profile-driven. The built-in manipulation scenario is
pick-and-place, with `red_block` as the default object and `blue_tray`
as the destination.

The generic runtime controller currently targets compatible MuJoCo
manipulators with position-controlled scalar arm joints and configured
gripper actuators. A robot with torque/velocity control, a mobile base,
bimanual control, or unusual gripper semantics may require a
robot-specific controller adapter.

A valid LLM plan is not proof that a trajectory is reachable or that a
manipulation succeeded. Only simulator-grounded verification can
establish task success.

## Requirements

-   Python 3.12 is the documented development target.
-   Dependencies from `requirements.txt`.
-   A Nebius Token Factory API key and an enabled model ID.
-   Compatible MuJoCo MJCF and mesh/texture assets for the selected
    robot.
-   For the training application, install
    `training/requirements-training.txt` in a separate environment.

## Installation

### Windows PowerShell

From the repository root:

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` as described in [Configuration](#configuration), then start
the runtime:

``` powershell
uvicorn app.main:app --reload
```

### Linux/macOS

Create and activate a Python 3.12 virtual environment, install
`requirements.txt`, copy `.env.example` to `.env`, configure the
environment variables, and run:

``` bash
uvicorn app.main:app --reload
```

The repository-root `main.py` is also a convenience launcher.

## Configuration

Configuration is loaded by `app/config.py` from environment variables
and the project `.env` file.

  -----------------------------------------------------------------------
  Variable                            Purpose
  ----------------------------------- -----------------------------------
  `NEBIUS_API_KEY`                    Nebius Token Factory API key. Keep
                                      it private; do not commit it.

  `NEBIUS_BASE_URL`                   OpenAI-compatible Nebius API base
                                      URL.

  `NEBIUS_MODEL`                      Enabled model ID to request.

  `NEBIUS_TIMEOUT_SECONDS`            API request timeout; the current
                                      settings code enforces a minimum of
                                      5 seconds and defaults to 45
                                      seconds.

  `NEBIUS_TEMPERATURE`                Planner sampling temperature;
                                      clamped to the range 0--1.

  `ROBOT_NAME`                        Default runtime robot profile ID.

  `ROBOPILOT_ASSETS_DIR`              Runtime robot asset root; defaults
                                      to `assets/robots/`.

  `ROBOPILOT_DATA_DIR`                Runtime data-adapter directory;
                                      defaults to `data/adapter/`.

  `SIMULATION_HEADLESS`               Headless simulation setting.

  `SIMULATION_REALTIME`               Whether simulation stepping
                                      attempts to run in real time.

  `MAX_RECOVERY_RETRIES`              Bounded recovery retry setting,
                                      clamped by the current settings
                                      code to 0--3.

  `TASK_WORKERS`                      Runtime task worker count, clamped
                                      by the current settings code to
                                      1--4.

  `CORS_ALLOW_ORIGINS`                Comma-separated allowed browser
                                      origins.
  -----------------------------------------------------------------------

Example Nebius configuration:

``` dotenv
NEBIUS_API_KEY=replace_with_your_key
NEBIUS_BASE_URL=https://api.tokenfactory.eu-north1.nebius.com/v1/
NEBIUS_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
NEBIUS_TIMEOUT_SECONDS=120
NEBIUS_TEMPERATURE=0.1
ROBOT_NAME=franka_panda
```

Use a base URL and model ID enabled for your Nebius account. The
regional URL and model above match the standalone request that was
successfully tested during development; confirm they remain valid for
your account. Do not put a real API key in source control, issue
reports, screenshots, or README commits.

Environment-backed settings are initialized when the Python module is
imported. **Restart the application after changing `.env`.**

## Run the runtime application

Start the server:

``` bash
uvicorn app.main:app --reload
```

Open:

-   Web UI: `http://127.0.0.1:8000/`
-   API documentation: `http://127.0.0.1:8000/docs`

## Submit and monitor a task

Submit a pick-and-place task:

``` bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d "{\"instruction\":\"Pick up the red block and place it in the blue tray\"}"
```

The API returns a task ID. Poll its status:

``` text
GET /tasks/{task_id}
```

Request cooperative cancellation:

``` text
POST /tasks/{task_id}/stop
```

Use the API documentation at `/docs` to confirm request and response
schemas for the checked-out version. Task state is currently in memory
and is lost when the runtime process restarts.

## Planner contract and validation

The planner returns a JSON object with these top-level fields:

-   `task`: `pick_and_place`
-   `target_object`: the configured object name
-   `destination`: the configured destination name
-   `steps`: the six supported actions in order
-   `recovery`: `stop_safely`

The required action order is:

1.  `approach_object`
2.  `grasp_object`
3.  `lift_object`
4.  `move_to_destination`
5.  `release_object`
6.  `verify`

Relevant steps must reference the configured object and destination. In
particular, `move_to_destination`, `release_object`, and `verify` must
reference the destination. The parser can normalize known short aliases
such as `approach`, `grasp`, `lift`, `move`, and `release` to the
canonical action names, but this does not replace schema and sequence
validation.

The LLM plan is untrusted input. Keep backend validation strict. Do not
treat a well-formed plan as evidence that the robot can reach a target
or complete the task.

## Independent training application

Install training dependencies in a separate environment:

``` bash
pip install -r training/requirements-training.txt
```

Start the training application from the repository root:

``` bash
uvicorn training.main:app --host 127.0.0.1 --port 8010 --reload
```

Open:

-   Training dashboard: `http://127.0.0.1:8010/`
-   Training API documentation: `http://127.0.0.1:8010/docs`

The dashboard supports the training workflows implemented in the current
checkout, including PPO/SAC training, simulator-generated transition
collection, checkpoint evaluation, world-model training, log inspection,
and explicit artifact import/download. CLI scripts remain available for
automation. See [`training/README.md`](training/README.md) for the
training API, commands, and operational details.

The training environment is a profile-selected, joint-target MuJoCo
manipulation environment; Panda is the reference profile. Evaluate
checkpoints using measured results. A trained policy is robot-specific
and must not be assumed to work on a different profile.

## Custom robot profiles

1.  Copy `configs/robots/custom_robot.template.json` to
    `configs/robots/<robot_id>.json`; the filename stem is the robot ID.

2.  Configure the robot's MJCF filename, ordered scalar arm joints, safe
    home positions, end-effector body, gripper actuator names/control
    values, gripper reset joint positions, and task geometry/names.

3.  Place the MJCF and all referenced meshes, textures, and included
    files under `assets/robots/<asset_directory>/` for runtime.

4.  Make a **separate** copy under
    `training/assets/robots/<asset_directory>/` for training. Do not
    make training import runtime code or depend on runtime processes.

5.  Validate the training asset path with:

    ``` bash
    python -m training.robots.validate_profile --robot-name <robot_id>
    ```

6.  Restart the applications. Confirm that the robot appears in the
    runtime UI and training dashboard dropdowns and in `GET /robots` /
    `GET /api/robots`.

### Robot compatibility checklist

-   Confirm all joint, body, actuator, mesh, texture, and include names
    against the actual MJCF.
-   Preserve asset directory structure when staging MJCF files. Relative
    paths such as `meshdir="assets"` require the referenced subdirectory
    to exist at the expected location.
-   Verify actuator control ranges and gripper semantics from the model;
    do not assume a gripper uses a particular numeric open value.
-   Check joint limits, home positions, workspace reachability,
    collision geometry, object/tray placement, and placement tolerances.
-   Test reset, motion, grasp, lift, release, and simulator verification
    separately.
-   Use the same robot profile and compatible assets for training and
    evaluation. Do not automatically deploy a checkpoint to a different
    robot.

Profile-driven dimensions make the environment extensible to compatible
manipulators, but do not guarantee every robot works without
controller-specific adaptation.

## Troubleshooting

### Nebius API request times out

1.  Verify `NEBIUS_API_KEY`, `NEBIUS_BASE_URL`, and `NEBIUS_MODEL`.
2.  Run a minimal standalone OpenAI-compatible request with the same
    endpoint, model, and credentials.
3.  Compare the standalone request with the application's actual
    `messages`, model ID, `response_format`, timeout, and retry
    settings.
4.  Inspect the logged exception and request duration. Do not log the
    API key.
5.  Restart the runtime after changing `.env`.

A successful standalone request confirms that the endpoint can answer
that request; it does not guarantee that the application's request
payload or timeout policy is identical.

### Plan fails Pydantic validation

Check that the response is valid JSON and uses the exact supported
action strings and order. Ensure the required `target_object` and
`destination` values are present on the relevant steps. Keep both schema
validation and sequence validation enabled.

### `PrimitiveError: Joint trajectory did not converge`

This indicates that planning/validation progressed far enough to enter
motion execution, but a controller convergence condition failed. It is
not an API timeout.

Investigate the specific primitive's target, inverse-kinematics result,
joint limits, actuator behavior, convergence tolerance, and
iteration/time budget. Log the primitive name, target pose, starting and
target joint values, residual/error metric, and iteration count. Do not
simply remove the convergence check or report success after a failed
motion.

For the current Franka Panda development state, this error has been
observed during pick-and-place. Successful end-to-end object placement
has **not** yet been established by the available task logs.

### MuJoCo cannot find a mesh

Check the model's `<compiler meshdir="...">`, mesh filenames,
`<include>` paths, and the exact staged directory structure. A file
existing in the source tree is not enough if a generated temporary scene
resolves a different relative path.

### Robot does not move or gripper behaves unexpectedly

Verify that the profile's joint and actuator names match the loaded
MJCF, that every arm joint has a compatible actuator, and that control
values and gripper joint positions match the model. Confirm that the
requested end-effector target is reachable before adjusting tolerances.

## Testing and safety limitations

Run the available test suite from the repository root:

``` bash
pytest -q
```

Unit tests may use fakes for planner/executor behavior. Real MuJoCo
integration tests require the relevant dependencies and model assets;
passing unit tests does not prove physical plausibility or successful
manipulation.

Other limitations:

-   Runtime task state is in memory and resets when the process
    restarts.
-   A planner response is not a motion plan guarantee.
-   A validated action sequence is not proof of successful grasp or
    placement.
-   End-to-end Panda pick-and-place remains unverified while trajectory
    convergence errors persist.
-   Do not connect this prototype to physical hardware without a
    separate safety review, a hardware-specific controller, collision
    and workspace validation, and an emergency-stop path.
