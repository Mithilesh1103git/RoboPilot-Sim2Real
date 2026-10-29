# RoboPilot-Sim2Real

Simulation-first robot manipulation with a **planner-in-the-loop runtime agent** and a **fully independent training project**.

## Architecture contract
- `app/`: API → observation → Nebius Nemotron planner → strict plan validation → bounded motion primitives → simulator-grounded verification → bounded recovery.
- `training/`: separate project with its own MuJoCo assets, Gymnasium environment, data pipeline, RL training and checkpoints. It does not import `app`, call its API, or depend on runtime processes/artifacts.
- No LLM is used for training data generation, labels, demonstrations, rewards, or policy learning.
- The planner proposes high-level intent only. It never supplies joint angles, velocities, code, or motor commands.

## Runtime scope
The initial supported scenario is Panda pick-and-place: move `red_block` into `blue_tray`. The runtime calls Nemotron through the Nebius OpenAI-compatible endpoint. Missing credentials/model configuration fail the task clearly; there is no silent deterministic-planner fallback.

## Install (PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set NEBIUS_API_KEY and NEBIUS_MODEL.
uvicorn app.main:app --reload
```
Linux/macOS: create and activate a Python 3.12 venv, `pip install -r requirements.txt`, copy `.env.example` to `.env`, configure credentials, then run `uvicorn app.main:app --reload`.

Open `http://127.0.0.1:8000/` for the web UI or `http://127.0.0.1:8000/docs` for the API. The root `main.py` file is a convenience launcher.

## Task example
```bash
curl -X POST http://127.0.0.1:8000/tasks -H "Content-Type: application/json" -d '{"instruction":"Pick up the red block and place it in the blue tray"}'
```
Poll `GET /tasks/{task_id}`; request cooperative cancellation with `POST /tasks/{task_id}/stop`.

## Independent training
Install `training/requirements-training.txt` in a separate environment, then:
```bash
python -m training.synthetic.collect_episodes --episodes 20
python -m training.rl.train --timesteps 50000 --algorithm PPO
python -m training.rl.evaluate --checkpoint training/runs/panda_ppo.zip --episodes 10
```
The training environment is a joint-target Panda manipulation baseline. A trained policy is not assumed to succeed; evaluate and report measured metrics.

## Tests and limitations
Run `pytest -q`. Unit tests use fakes for planner/executor behavior. Real MuJoCo integration requires `mujoco` and bundled assets. In-memory task state is lost on process restart. Do not connect this prototype to physical hardware without a separate safety review, hardware-specific controller, and emergency-stop path.
