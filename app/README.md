# RoboPilot-Sim2Real

![project_logo_artwork]assets/images/RoboPilot-Sim2Real-logo-artwork.png

**Simulation-first robot manipulation with a planner-in-the-loop agent,
MuJoCo-grounded execution, and an independent reinforcement-learning
research pipeline.**

RoboPilot-Sim2Real explores how a language-conditioned agent can
translate a natural-language manipulation request into a constrained
sequence of robot actions, execute those actions in a physics simulator,
and verify the resulting state. A separate training application develops
and evaluates policies using simulator-generated experience without
depending on the language-model runtime.

> **Project status:** Active research prototype. The runtime, browser
> UI, training application, and robot-profile system are under
> development. Check the running API's `/docs` page for the routes
> actually registered by your checkout. Do not assume that older route
> examples or a particular API prefix match the current backend.

------------------------------------------------------------------------

## Contents

-   [Project goals](#project-goals)
-   [System overview](#system-overview)
-   [Runtime architecture](#runtime-architecture)
-   [Simulation and browser viewport](#simulation-and-browser-viewport)
-   [Independent training
    architecture](#independent-training-architecture)
-   [Repository layout](#repository-layout)
-   [Requirements](#requirements)
-   [Runtime setup](#runtime-setup)
-   [Training setup](#training-setup)
-   [Configuration](#configuration)
-   [Robot profiles](#robot-profiles)
-   [API discovery and
    troubleshooting](#api-discovery-and-troubleshooting)
-   [Testing](#testing)
-   [Current scope and limitations](#current-scope-and-limitations)
-   [Safety](#safety)
-   [Research direction](#research-direction)

------------------------------------------------------------------------

## Project goals

RoboPilot-Sim2Real is organized around four goals:

1.  **Ground language-conditioned planning in a simulator.** A
    Nebius-hosted Nemotron model proposes a high-level task plan from
    the user's instruction and the current task context.
2.  **Keep execution deterministic and bounded.** The planner does not
    issue low-level motor commands. Application code validates the plan
    and maps supported actions to bounded control primitives.
3.  **Verify outcomes against simulation state.** Execution should be
    assessed using simulator observations and task conditions rather
    than treating a generated plan as proof of success.
4.  **Develop policies independently of LLMs.** The training project
    uses simulator-driven data, scripted/teacher behavior, reinforcement
    learning, and evaluation. It does not use an LLM to generate
    training instructions, labels, demonstrations, rewards, or policy
    updates.

The current reference manipulation task is pick-and-place: move a
configured object (by default `red_block`) into a configured destination
(by default `blue_tray`).

## System overview

``` text
                         RUNTIME APPLICATION
                         ====================

 User instruction
       |
       v
 Browser UI -----> FastAPI runtime API
                         |
                         v
                 Task / observation
                         |
                         v
                 Nebius Nemotron
                 high-level planner
                         |
                         v
                 Strict plan parsing
                 and validation
                         |
                         v
                 Bounded motion primitives
                         |
                         v
                   MuJoCo physics
                         |
                         v
               State-based verification
                         |
                  +------+------+
                  |             |
                Success     Bounded recovery
                                |
                            Safe stop


                    INDEPENDENT TRAINING
                    ====================

       MuJoCo training assets + robot profile
                         |
                         v
             Profile-driven Gymnasium env
                         |
             +-----------+------------+
             |                        |
       Scripted / teacher       Simulator-generated
          behavior                 transitions
             |                        |
             +-----------+------------+
                         |
                         v
                 Replay / datasets
                         |
                         v
                RL policy training
                    (e.g. PPO/SAC)
                         |
                         v
                 Policy evaluation

        World-model training and evaluation are
        a separate training workstream. The intended
        information flow is world model -> RL, not
        an LLM-driven training loop.
```

The runtime and training application are deliberately separate. They may
use corresponding robot profiles and compatible MuJoCo models, but the
training project must not import the runtime package, call runtime
endpoints, or depend on runtime processes or runtime-generated
artifacts.

## Runtime architecture

The runtime follows this control flow:

1.  **API layer** accepts a task request and manages task execution.
2.  **Observation layer** obtains the relevant state from the
    simulation.
3.  **Planner** calls Nemotron through the Nebius OpenAI-compatible API.
4.  **Plan validation** parses the model response into the application's
    typed plan schema and rejects unsupported tasks, actions, targets,
    destinations, or recovery behavior.
5.  **Motion primitives** execute the supported action sequence through
    application-controlled robot commands.
6.  **MuJoCo simulation** advances the configured robot and task scene.
7.  **Verification** checks the simulator-grounded result.
8.  **Recovery** is bounded. If the supported operation cannot safely
    continue, the runtime stops rather than improvising arbitrary
    low-level commands.

### Planner boundary

The language model is a high-level planner---not a controller.

It must not provide or directly execute:

-   Joint positions or joint trajectories.
-   Joint velocities or torques.
-   Arbitrary Python code or shell commands.
-   Direct actuator or motor commands.
-   Unvalidated action names, targets, or destinations.

The application owns validation, control limits, simulator interaction,
verification, and recovery.

### Current task scope

The documented built-in task is profile-configured pick-and-place. The
validated action sequence currently represented in the runtime includes:

1.  `approach_object`
2.  `grasp_object`
3.  `lift_object`
4.  `move_to_destination`
5.  `release_object`
6.  `verify`

This is not a claim of general-purpose manipulation. Additional tasks
require corresponding scene definitions, schemas, validation rules,
control primitives, and success criteria.

## Simulation and browser viewport

MuJoCo is the physics and simulation foundation. The runtime environment
loads a robot model from its profile, constructs the task scene, and
advances the simulation.

The browser UI is a vanilla HTML/CSS/JavaScript application served by
the runtime API. The current `RoboPilot.html` contains a **MuJoCo
viewport placeholder**. A live browser-rendered viewport requires an
additional rendering/streaming integration; it should not be considered
operational merely because the placeholder is visible.

### Planned browser rendering options

Two suitable implementation paths are:

-   **Backend-rendered frames:** render the MuJoCo scene in Python and
    stream image frames to the browser, for example over a WebSocket.
    This most directly displays the simulator's rendered camera view.
-   **Browser-side 3D rendering:** send simulation state (such as joint
    positions) to a Three.js scene. This supports interactive orbit,
    pan, and zoom, but requires a synchronized visual model and scene
    representation.

Whichever renderer is selected, MuJoCo remains authoritative for physics
and task state. A browser visualization must not become a second physics
engine or be treated as proof that the simulated task succeeded.

## Independent training architecture

`training/` is a standalone application for simulator-driven policy
development and evaluation.

### Independence contract

The training project:

-   Has its own MuJoCo assets and robot asset copies.
-   Has its own profile-driven Gymnasium environment and training code.
-   Runs as a separate FastAPI application and dashboard.
-   Does not import `app/`.
-   Does not call the runtime API.
-   Does not depend on runtime processes, runtime task state, or runtime
    artifacts.
-   Does not use an LLM for training data generation, labels,
    demonstrations, rewards, or policy learning.

### Training workflow

The training application is intended to support:

-   Simulator-driven transition collection.
-   Scripted or teacher-policy behavior for data generation.
-   RL training, including PPO/SAC workflows exposed by the training
    application.
-   Checkpoint evaluation and measured performance reporting.
-   World-model training and inspection as a distinct workstream.
-   Job logs and explicit artifact import/export operations where
    supported by the dashboard and CLI.

The intended research direction keeps the world-model-to-RL information
flow unidirectional. Training results must be evaluated empirically; the
existence of a checkpoint does not imply that a policy is successful,
safe, or transferable.

For training-specific commands, API details, and limitations, see
[`training/README.md`](training/README.md).

## Repository layout

The following is a **conceptual map of the documented components**, not
a guarantee that every optional file is present in every checkout. Use
your working tree as the source of truth.

``` text
RoboPilot-Sim2Real/
├── app/                         # Runtime FastAPI application
│   ├── api/                     # Runtime API routers
│   ├── control/                 # Bounded motion primitives
│   ├── planner/                 # Nebius / Nemotron planner integration
│   ├── robots/                  # Runtime robot-profile loading
│   ├── simulation/              # MuJoCo runtime environment
│   ├── config.py                # Environment-backed runtime settings
│   ├── schemas.py               # Request and plan schemas
│   └── main.py                  # Runtime FastAPI app
├── assets/
│   └── robots/                  # Runtime MJCF and referenced assets
├── configs/
│   └── robots/                  # Robot profile JSON and template
├── web/
│   ├── RoboPilot.html           # Runtime browser UI
│   ├── styles.css
│   └── app.js
├── training/                    # Independent training application
│   ├── assets/                  # Separate training robot assets
│   ├── robots/                  # Training-side profile utilities
│   ├── ...                      # Environment, data, RL, evaluation, API, UI
│   ├── main.py                  # Training FastAPI app
│   └── README.md
├── tests/                       # Runtime and project tests
├── .env.example                 # Runtime configuration template
├── requirements.txt             # Runtime dependencies
└── README.md
```

Do not collapse the runtime and training directories into one import
graph. In particular, do not make `training/` import modules from `app/`
for convenience.

## Requirements

-   Python 3.12 is the documented development baseline.
-   MuJoCo Python package (`mujoco`) and compatible robot MJCF assets.
-   FastAPI and Uvicorn for the applications.
-   A Nebius API key and a supported Nemotron model identifier for
    runtime planning.
-   Training dependencies from `training/requirements-training.txt` for
    the independent training application.
-   A compatible OS, Python environment, and graphics/rendering setup
    for the selected MuJoCo rendering mode.

The exact dependency set is defined by the repository's requirements
files. Install training dependencies separately from runtime
dependencies.

## Runtime setup

### 1. Create a virtual environment

**Windows PowerShell**

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**

``` bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the template:

**PowerShell**

``` powershell
Copy-Item .env.example .env
```

**Linux / macOS**

``` bash
cp .env.example .env
```

Set at least the runtime planner credentials and model:

``` dotenv
NEBIUS_API_KEY=your_nebius_api_key
NEBIUS_MODEL=your_supported_nemotron_model
```

Do not commit `.env` or expose API credentials in browser JavaScript,
source control, screenshots, or logs.

### 3. Start the runtime API

From the repository root, with the virtual environment activated:

``` bash
uvicorn app.main:app --reload
```

The default local address is:

-   Runtime base URL: `http://127.0.0.1:8000`
-   Interactive API documentation, if enabled:
    `http://127.0.0.1:8000/docs`
-   OpenAPI schema, if enabled: `http://127.0.0.1:8000/openapi.json`

The UI is expected to be served by the runtime application when its
static-file and root-page routes are configured. If `/` returns 404, the
root page is not registered in the application currently running; verify
the entry point and route table rather than assuming the HTML file is
automatically served.

## Training setup

The training service runs separately from the runtime service.

Install its dependencies in a separate environment (recommended) or
otherwise keep its dependency installation isolated:

``` bash
python -m pip install --upgrade pip
pip install -r training/requirements-training.txt
```

Start the training application from the repository root:

``` bash
uvicorn training.main:app --host 127.0.0.1 --port 8010 --reload
```

Expected local addresses, when the relevant routes are enabled:

-   Training dashboard: `http://127.0.0.1:8010/`
-   Training API documentation: `http://127.0.0.1:8010/docs`

The training dashboard and runtime UI are separate applications.
Starting one does not start the other.

## Configuration

Runtime settings are environment-backed. The current configuration
includes:

  -----------------------------------------------------------------------
  Variable                            Purpose
  ----------------------------------- -----------------------------------
  `APP_NAME`                          Runtime application name

  `ROBOT_NAME`                        Default runtime robot profile;
                                      defaults to `panda`

  `ROBOPILOT_ASSETS_DIR`              Runtime robot asset root

  `ROBOPILOT_DATA_DIR`                Runtime adapter/data directory

  `NEBIUS_API_KEY`                    Nebius API credential

  `NEBIUS_BASE_URL`                   Nebius OpenAI-compatible API base
                                      URL

  `NEBIUS_MODEL`                      Model identifier used by the
                                      planner

  `NEBIUS_TIMEOUT_SECONDS`            Planner request timeout

  `NEBIUS_TEMPERATURE`                Planner sampling temperature

  `SIMULATION_HEADLESS`               Runtime headless-simulation setting

  `SIMULATION_REALTIME`               Whether runtime simulation is paced
                                      in real time

  `MAX_STEP_RETRIES`                  Per-step retry limit where used by
                                      the runtime

  `MAX_RECOVERY_RETRIES`              Bounded recovery retry limit

  `TASK_WORKERS`                      Runtime task worker limit

  `CORS_ALLOW_ORIGINS`                Comma-separated allowed browser
                                      origins
  -----------------------------------------------------------------------

Use `.env.example` as the authoritative list of variables for the exact
checkout. Do not assume a setting is active merely because it appears in
this table; configuration behavior is determined by the checked-out
code.

## Robot profiles

Robot selection is profile-driven. The runtime profile directory is:

``` text
configs/robots/
```

The runtime asset root defaults to:

``` text
assets/robots/
```

The independent training application uses its own robot asset root:

``` text
training/assets/robots/
```

### Adding a compatible robot

1. Copy `configs` to
    `configs`.
2.  Configure the robot's MJCF filename and asset directory.
3.  Define ordered scalar arm joints and safe home positions.
4.  Define the end-effector body.
5.  Configure gripper actuator names and control values, plus gripper
    reset joint positions where required.
6.  Configure task object/destination names and geometry/placement
    parameters.
7.  Put the MJCF and all referenced meshes, textures, includes, and
    other model files under `assets/robots/<asset_directory>/`.
8.  Make a separate asset copy under
    `training/assets/robots/<asset_directory>/` for training. Do not
    point training at runtime modules or runtime process state.
9.  Validate the training-side profile and asset path:

``` bash
python -m training.robots.validate_profile --robot-name <robot_id>
```

10. Restart the relevant applications and confirm the profile is
    discoverable in each application.

Use the existing Panda profile as the reference implementation for the
schema and expected fields.

### Compatibility boundaries

The generic manipulation environment derives action and observation
dimensions from the profile; it is not inherently limited to seven arm
joints. However, profile-driven loading does not mean every robot is
automatically supported.

Each robot must be checked for:

-   MJCF naming and include/mesh path correctness.
-   Joint and actuator mapping.
-   Kinematics and end-effector definition.
-   Gripper semantics and reset behavior.
-   Collision geometry and self-collision behavior.
-   Workspace and task-object placement.
-   Controller compatibility and task tolerances.

The documented built-in controller targets position-controlled scalar
arm joints and configured gripper actuators. Torque-controlled or
velocity-controlled robots, mobile bases, bimanual systems, and unusual
grippers may require a robot-specific controller adapter.

A trained checkpoint is robot- and environment-specific unless transfer
has been explicitly implemented and evaluated. Train and evaluate
against the same robot profile; do not automatically deploy a checkpoint
to a different robot.

## API discovery and troubleshooting

API paths are determined by the actual FastAPI router registrations and
route decorators in the running checkout. A router prefix and a route
decorator combine to form the final path. Do not add `/api/v1` in the
frontend unless the backend registers that prefix.

### Inspect registered routes

Temporarily add this after all `include_router(...)` calls in the
runtime application entry point:

``` python
for route in app.routes:
    methods = getattr(route, "methods", None)
    print(f"{','.join(sorted(methods or [])):20} {route.path}")
```

Restart the same Uvicorn process that serves the browser UI and inspect
the terminal output.

If OpenAPI is enabled, also inspect:

``` text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/openapi.json
```

Use the routes shown by the running application as the source of truth
for frontend URLs and API examples.

### Common symptoms

  -----------------------------------------------------------------------
  Symptom                             What to check
  ----------------------------------- -----------------------------------
  `404` for `/api/v1/...`             The backend may register a
                                      different prefix or no matching
                                      router. Inspect `app.routes` and
                                      `/docs`.

  `404` for `/`                       The root page route may be missing,
                                      or Uvicorn may be serving a
                                      different FastAPI app.

  `/docs` also returns `404`          Confirm the correct module/app is
                                      running and that docs have not been
                                      disabled.

  Robot profile list is empty         Check profile JSON files, profile
                                      directory, and referenced robot
                                      assets.

  MuJoCo model fails to load          Check MJCF path, includes,
                                      mesh/texture references, and
                                      profile names.

  Planner request fails               Check `NEBIUS_API_KEY`,
                                      `NEBIUS_MODEL`, base URL, network
                                      access, and server-side logs.

  Browser UI cannot reach the API     Check the browser's Network panel,
                                      the actual route table, CORS
                                      settings, and frontend request
                                      paths.

  Viewport remains a placeholder      The browser rendering/streaming
                                      path has not been connected or is
                                      not receiving frames.
  -----------------------------------------------------------------------

A `404` is a route-registration or path mismatch, not evidence that the
MuJoCo physics engine itself has failed.

## Testing

Run the project test suite from the repository root:

``` bash
pytest -q
```

Unit tests may use fakes for planner and executor behavior. Real MuJoCo
integration tests require the MuJoCo package and the relevant robot
assets. Read individual test requirements before interpreting a passing
unit-test suite as proof of end-to-end simulation behavior.

For meaningful evaluation, record the robot profile, environment
configuration, task setup, policy/checkpoint identifier, number of
evaluation episodes, success criteria, and measured results. Do not
report unmeasured performance claims.

## Current scope and limitations

-   The reference runtime task is profile-configured pick-and-place.
-   The planner is constrained by application-side schemas and
    validation; it is not a general-purpose robot controller.
-   The runtime's supported controller and gripper semantics are limited
    to compatible configured actuators.
-   The current browser HTML includes a viewport placeholder; a live
    rendered browser viewport needs a separate rendering/streaming
    integration.
-   Runtime API paths must be verified from the actual running app.
    Older examples may not match later router changes.
-   Runtime task state and training job state may be in-memory in the
    current prototype; state can be lost when the corresponding process
    restarts.
-   Training and runtime use separate assets and code paths. Any
    intentional artifact exchange must be explicit.
-   RL and world-model checkpoints require measured evaluation. Training
    completion is not evidence of task success or safe behavior.
-   Physical-robot deployment, sim-to-real transfer, and cross-robot
    policy transfer are not implied by a successful MuJoCo
    demonstration.

## Safety

This repository is a simulation-first research prototype. **Do not
connect it to physical hardware or use it to command a real robot
without a separate engineering and safety review.**

A real deployment would require, at minimum:

-   A hardware-specific controller and validated command interface.
-   Independent safety-rated limits and workspace constraints.
-   A physical emergency-stop path.
-   Collision, payload, speed, and force limits appropriate to the
    hardware.
-   Fault handling, operator supervision, and controlled recovery.
-   Validation of the complete system under the applicable safety
    standards and operating procedures.

Simulation checks and application-level validation are not substitutes
for hardware safety systems.

## Research direction

The project is intended as a foundation for investigating:

-   Language-conditioned task planning with constrained execution.
-   Simulator-grounded action verification and bounded recovery.
-   Robot-profile-driven manipulation environments.
-   Synthetic experience collection using scripted/teacher policies.
-   RL policy training and reproducible evaluation.
-   World-model learning and its downstream use by RL.
-   Future sim-to-real evaluation, only after the simulation and
    hardware interfaces have been independently validated.

The central architectural principle is to keep **language planning,
low-level control, simulation verification, and policy training as
distinct responsibilities**.
