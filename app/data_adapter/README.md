# Data Adapter & Task Interface

This package implements the reusable processing layer for Stage 2 of RoboPilot-Sim2Real: ingestion, normalization, episode grouping, validation, standardized storage, and export. HTTP routes live in `app/api/datasets.py for dataset operations; app/api/tasks.py for task creation and execution`; this package does not own an API router.

## Canonical format

Schema: `robopilot.demonstrations`, version `1.0.0`.

- Dataset manifest: dataset ID, source format/name, timestamp, episode/transition counts, robot labels, metadata.
- Episode: episode ID, instruction, robot, embodiment, coordinate frame, ordered transitions, metadata.
- Transition: timestamp, observation object, action, instruction, episode ID, robot/embodiment/frame, reward/terminal flags, success label, source metadata.

The adapter accepts a JSON object or list. An optional 4×4 homogeneous `coordinate_transform` can transform recognized 3D position and pose fields. An optional `embodiment_mapping` renames named fields such as joint names; it deliberately does not guess mappings for unnamed action arrays. It recognizes common aliases (`obs`, `observations`, `state`; `actions`, `a`, `command`; `language`, `task_description`; `trajectory_id`, `demo_id`). It accepts `{episodes:[{episode_id, instruction, transitions:[...]}]}` and generic transition rows. This is a normalization baseline, not a guarantee of complete compatibility with every LIBERO release; convert source-specific binary/image formats to JSON records or add a dedicated reader.

## HTTP API

Routes are implemented in `app/api/datasets.py for dataset operations; app/api/tasks.py for task creation and execution` and registered by `app/main.py`. The `app/data_adapter/` package contains processing and storage logic only.

- `GET /api/v1/datasets/schema` — canonical schema description.
- `POST /api/v1/datasets/ingest` — normalize and persist JSON records.
- `GET /api/v1/datasets` — list standardized datasets.
- `GET /api/v1/datasets/{dataset_id}` — fetch manifest and episodes.
- `GET /api/v1/datasets/{dataset_id}/export?format=jsonl|json` — download standardized output.
- `POST /api/v1/tasks/from-dataset` — create a normal simulation task using an episode's instruction. This deliberately does not execute recorded actions directly; embodiment mapping and action replay must be validated for the target robot first.

Example ingest body:

```json
{
  "source_format": "vr4il",
  "source_name": "quest-demo-001",
  "robot": "panda",
  "embodiment": "franka_panda",
  "coordinate_frame": "world",
  "coordinate_transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "embodiment_mapping": {"source_joint_1":"panda_joint1"},
  "instruction": "Pick up the red block and place it in the tray",
  "records": [
    {"episode_id":"demo-1","timestamp":0.0,"obs":{"ee_pose":[0.4,0,0.3]},"action":[0.1,0.0,0.0]},
    {"episode_id":"demo-1","timestamp":0.1,"obs":{"ee_pose":[0.41,0,0.3]},"action":[0.1,0.0,0.0],"terminated":true,"success":true}
  ]
}
```

## Storage

Raw input: `data/adapter/raw/<dataset_id>.json`.
Standardized: `data/adapter/standardized/<dataset_id>/manifest.json` and `episodes.jsonl`.

Data may contain images or arrays too large for JSON; this baseline expects JSON-serializable records. Keep image/video assets in object storage and refer to them by URI in observation metadata. Do not treat missing success labels as success.
