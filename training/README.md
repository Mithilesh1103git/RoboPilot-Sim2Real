# RoboPilot-Sim2Real Training Pipeline

The training subsystem is a **VM/container-deployable service** as well as a set of optional local CLI tools. The deployment lives in `training/deploy/`.

- Stage 4: synthetic transition generation, domain-randomization manifests and augmentation. The deployed Flow Controller can invoke the transition generator in-process as part of a training run.
- Stage 5: object-storage-backed unified buffer, Flow Controller, World Model training, optional RL policy training, checkpoints and artifact upload.
- Durable datasets, checkpoints, metrics and run manifests reside in S3-compatible object storage; container disk is temporary scratch space.
- PyTorch uses CUDA when available and configured. MuJoCo remains physics ground truth and is not automatically GPU accelerated.

## Deploy as a container

See [`deploy/README.md`](deploy/README.md). In brief:

```bash
cd training/deploy
cp .env.example .env
# Set object storage endpoint, bucket and credentials in .env
docker compose up --build -d
```

The service exposes `/health`, `/storage/buffer`, `POST /runs`, `GET /runs/{run_id}`, and interactive API docs at `/docs`. To generate synthetic transitions inside the deployed pipeline, submit `generate_synthetic: true` in the `POST /runs` body. Optionally set `upload_synthetic_to_buffer: true` to persist the generated dataset into the configured unified-buffer prefix.

## Local development (optional CLI)

```bash
python -m pip install -r training/requirements-training.txt
python -m training.synthetic.generate_episodes --env-id Pendulum-v1 --episodes 10 --output training/datasets/transitions.npz
python -m training.rl.world_model --dataset training/datasets/transitions.npz --epochs 20 --output training/rl/checkpoints/world_model.pt
```

`Pendulum-v1` is only a smoke-test environment. Robot-specific RL training requires a registered Gymnasium environment connected to the RoboPilot simulation/control stack.
