"""Normalize common demonstration formats into RoboPilot's canonical schema.

Supported inputs are intentionally pragmatic: JSON/JSONL records, episode-oriented
JSON, generic transition dictionaries, and LIBERO/VR4IL-style aliases. This is not
a claim of full compatibility with every upstream dataset release; source-specific
readers can be added behind the same normalization boundary.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import DatasetEpisode, DatasetManifest, Transition
from .transforms import remap_named_fields, transform_observation, validate_transform

ALIASES = {
    "observation": ("observation", "observations", "obs", "state", "robot_state"),
    "action": ("action", "actions", "a", "command", "robot_action"),
    "instruction": (
        "instruction",
        "language_instruction",
        "task",
        "task_description",
        "language",
    ),
    "episode_id": ("episode_id", "episode", "trajectory_id", "demo_id", "sequence_id"),
    "timestamp": ("timestamp", "time", "t", "sim_time"),
    "reward": ("reward", "rewards"),
    "success": ("success", "is_success", "task_success"),
    "terminated": ("terminated", "done", "terminal"),
    "truncated": ("truncated", "timeout"),
}


class DatasetValidationError(ValueError):
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or [message]


def _first(mapping: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise DatasetValidationError(
                "Dataset contains NaN or infinite numeric values."
            )
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return str(value)


def _observation(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return _json_safe(value)
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        return {"vector": _json_safe(value)}
    return {"value": _json_safe(value)}


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        # Episode-oriented formats often put transition rows under one of these keys.
        rows = _first(payload, ("transitions", "records", "steps", "data"))
        if rows is None:
            episodes = payload.get("episodes")
            if isinstance(episodes, list):
                flattened = []
                for idx, ep in enumerate(episodes):
                    if not isinstance(ep, dict):
                        raise DatasetValidationError(
                            f"episodes[{idx}] must be an object"
                        )
                    eid = str(_first(ep, ("episode_id", "id"), f"episode-{idx+1:04d}"))
                    instruction = str(_first(ep, ALIASES["instruction"], ""))
                    steps = _first(ep, ("transitions", "steps", "records", "data"), [])
                    if not isinstance(steps, list):
                        raise DatasetValidationError(
                            f"Episode {eid} transitions must be a list"
                        )
                    for step in steps:
                        if not isinstance(step, dict):
                            raise DatasetValidationError(
                                f"Episode {eid} contains a non-object transition"
                            )
                        row = dict(step)
                        row.setdefault("episode_id", eid)
                        if instruction:
                            row.setdefault("instruction", instruction)
                        flattened.append(row)
                rows = flattened
            else:
                # One record object is accepted as a single-transition dataset.
                rows = [payload]
    else:
        raise DatasetValidationError("Input must be a JSON object or list of objects")
    if not isinstance(rows, list) or not rows:
        raise DatasetValidationError("No transitions were found in the input")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DatasetValidationError(f"Transition at index {idx} must be an object")
    return rows


class DatasetAdapter:
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root).expanduser().resolve()
        self.raw_root = self.data_root / "raw"
        self.standardized_root = self.data_root / "standardized"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.standardized_root.mkdir(parents=True, exist_ok=True)

    def normalize(
        self,
        payload: Any,
        *,
        source_format: str = "auto",
        source_name: str = "uploaded",
        metadata: dict | None = None,
        coordinate_frame: str = "world",
        robot: str = "unknown",
        embodiment: str = "unknown",
        instruction: str = "",
        coordinate_transform: Any = None,
        embodiment_mapping: dict[str, str] | None = None,
    ) -> tuple[DatasetManifest, list[DatasetEpisode]]:
        rows = _records_from_payload(payload)
        transform = (
            validate_transform(coordinate_transform)
            if coordinate_transform is not None
            else None
        )
        mapping = embodiment_mapping or {}
        dataset_id = "ds_" + uuid4().hex[:16]
        grouped: dict[str, list[Transition]] = {}
        episode_meta: dict[str, dict] = {}
        for idx, row in enumerate(rows):
            eid = str(_first(row, ALIASES["episode_id"], "episode-0001"))
            obs = _first(row, ALIASES["observation"], {})
            action = _first(row, ALIASES["action"], None)
            if transform is not None:
                obs = transform_observation(obs, transform)
            if mapping:
                obs = remap_named_fields(obs, mapping)
                action = remap_named_fields(action, mapping)
            inst = str(_first(row, ALIASES["instruction"], instruction) or "")
            timestamp = _first(row, ALIASES["timestamp"], None)
            try:
                timestamp = float(timestamp) if timestamp is not None else float(idx)
            except (TypeError, ValueError):
                raise DatasetValidationError(
                    f"Transition {idx}: timestamp must be numeric"
                )
            tr = Transition(
                timestamp=timestamp,
                observation=_observation(obs),
                action=_json_safe(action),
                instruction=inst,
                episode_id=eid,
                robot=str(_first(row, ("robot", "robot_name", "robot_type"), robot)),
                embodiment=str(
                    _first(row, ("embodiment", "robot_embodiment"), embodiment)
                ),
                coordinate_frame=str(
                    _first(
                        row,
                        ("coordinate_frame", "frame", "reference_frame"),
                        coordinate_frame,
                    )
                ),
                task_metadata=_json_safe(
                    _first(row, ("task_metadata", "metadata", "task_info"), {}) or {}
                ),
                source_metadata={
                    "source_format": source_format,
                    "source_name": source_name,
                    "source_index": idx,
                },
                reward=(
                    float(_first(row, ALIASES["reward"]))
                    if _first(row, ALIASES["reward"]) is not None
                    else None
                ),
                terminated=bool(_first(row, ALIASES["terminated"], False)),
                truncated=bool(_first(row, ALIASES["truncated"], False)),
                success=(
                    bool(_first(row, ALIASES["success"]))
                    if _first(row, ALIASES["success"]) is not None
                    else None
                ),
            )
            # Preserve unrecognized fields in task metadata so source information is not silently lost.
            known = {alias for aliases in ALIASES.values() for alias in aliases}
            extras = {
                k: v
                for k, v in row.items()
                if k not in known
                and k
                not in {
                    "robot",
                    "robot_name",
                    "robot_type",
                    "embodiment",
                    "robot_embodiment",
                    "coordinate_frame",
                    "frame",
                    "reference_frame",
                    "task_metadata",
                    "metadata",
                    "task_info",
                }
            }
            if extras:
                tr.task_metadata = {
                    **tr.task_metadata,
                    "source_fields": _json_safe(extras),
                }
            grouped.setdefault(eid, []).append(tr)
            episode_meta.setdefault(
                eid,
                {
                    "instruction": inst,
                    "robot": tr.robot,
                    "embodiment": tr.embodiment,
                    "coordinate_frame": tr.coordinate_frame,
                },
            )
        episodes = []
        errors = []
        for eid, transitions in grouped.items():
            if not any(t.observation for t in transitions):
                errors.append(f"Episode {eid} has no observations")
            if not any(t.action is not None for t in transitions):
                errors.append(f"Episode {eid} has no actions")
            if any(t.timestamp is None for t in transitions):
                errors.append(f"Episode {eid} has a transition without timestamp")
            transitions.sort(
                key=lambda t: t.timestamp if t.timestamp is not None else 0
            )
            m = episode_meta[eid]
            episodes.append(
                DatasetEpisode(
                    episode_id=eid,
                    instruction=m["instruction"],
                    robot=m["robot"],
                    embodiment=m["embodiment"],
                    coordinate_frame=m["coordinate_frame"],
                    transitions=transitions,
                    metadata={"transition_count": len(transitions)},
                )
            )
        if errors:
            raise DatasetValidationError("Dataset validation failed", errors)
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            source_format=source_format,
            source_name=source_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            episode_count=len(episodes),
            transition_count=len(rows),
            robots=sorted({ep.robot for ep in episodes}),
            metadata=_json_safe(
                {
                    **(metadata or {}),
                    "coordinate_transform_applied": transform is not None,
                    "embodiment_mapping": mapping,
                }
            ),
        )
        return manifest, episodes

    def save(
        self,
        manifest: DatasetManifest,
        episodes: list[DatasetEpisode],
        raw_payload: Any | None = None,
    ) -> Path:
        target = self.standardized_root / manifest.dataset_id
        target.mkdir(parents=True, exist_ok=False)
        if raw_payload is not None:
            (self.raw_root / f"{manifest.dataset_id}.json").write_text(
                json.dumps(_json_safe(raw_payload), indent=2), encoding="utf-8"
            )
        (target / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        with (target / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(episode.model_dump_json() + "\n")
        return target

    def load_manifest(self, dataset_id: str) -> DatasetManifest:
        path = self._dataset_path(dataset_id) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(dataset_id)
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def load_episodes(self, dataset_id: str) -> list[DatasetEpisode]:
        path = self._dataset_path(dataset_id) / "episodes.jsonl"
        if not path.is_file():
            raise FileNotFoundError(dataset_id)
        episodes = []
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.strip():
                try:
                    episodes.append(DatasetEpisode.model_validate_json(line))
                except Exception as exc:
                    raise DatasetValidationError(
                        f"Invalid standardized episode on line {line_no}: {exc}"
                    ) from exc
        return episodes

    def list_datasets(self) -> list[DatasetManifest]:
        manifests = []
        for path in sorted(
            self.standardized_root.glob("ds_*/manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                manifests.append(
                    DatasetManifest.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except Exception:
                continue
        return manifests

    def export(self, dataset_id: str, fmt: str = "jsonl") -> Path:
        episodes = self.load_episodes(dataset_id)
        manifest = self.load_manifest(dataset_id)
        if fmt not in {"json", "jsonl"}:
            raise DatasetValidationError("Export format must be json or jsonl")
        out = self._dataset_path(dataset_id) / f"export.{fmt}"
        if fmt == "json":
            out.write_text(
                json.dumps(
                    {
                        "manifest": manifest.model_dump(),
                        "episodes": [e.model_dump() for e in episodes],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            out.write_text(
                "".join(e.model_dump_json() + "\n" for e in episodes), encoding="utf-8"
            )
        return out

    def _dataset_path(self, dataset_id: str) -> Path:
        if not re.fullmatch(r"ds_[a-f0-9]{16}", dataset_id):
            raise FileNotFoundError(dataset_id)
        path = (self.standardized_root / dataset_id).resolve()
        if path.parent != self.standardized_root.resolve():
            raise FileNotFoundError(dataset_id)
        return path
