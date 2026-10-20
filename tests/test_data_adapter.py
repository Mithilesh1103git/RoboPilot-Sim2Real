import json
import pytest
from app.data_adapter.service import DatasetAdapter, DatasetValidationError
from pathlib import Path


def test_normalize_generic_records(tmp_path):
    adapter = DatasetAdapter(tmp_path)
    payload = [
        {
            "episode_id": "e1",
            "timestamp": 0,
            "obs": {"qpos": [0, 1]},
            "action": [0.1],
            "language": "pick block",
        },
        {
            "episode_id": "e1",
            "timestamp": 1,
            "observations": {"qpos": [1, 2]},
            "actions": [0.2],
            "terminated": True,
        },
    ]
    manifest, episodes = adapter.normalize(payload, source_format="test", robot="panda")
    assert manifest.episode_count == 1
    assert manifest.transition_count == 2
    assert episodes[0].transitions[0].observation["qpos"] == [0, 1]
    saved = adapter.save(manifest, episodes, raw_payload=payload)
    assert (saved / "manifest.json").exists()
    assert len(adapter.load_episodes(manifest.dataset_id)) == 1


def test_normalize_episode_oriented_payload(tmp_path):
    adapter = DatasetAdapter(tmp_path)
    payload = {
        "episodes": [
            {
                "episode_id": "e1",
                "instruction": "move object",
                "transitions": [
                    {"obs": {"x": 1}, "action": [0]},
                    {"obs": {"x": 2}, "action": [1]},
                ],
            }
        ]
    }
    manifest, episodes = adapter.normalize(payload)
    assert manifest.episode_count == 1
    assert episodes[0].instruction == "move object"


def test_missing_action_rejected(tmp_path):
    adapter = DatasetAdapter(tmp_path)
    with pytest.raises(DatasetValidationError):
        adapter.normalize([{"episode_id": "e1", "obs": {"x": 1}}])


def test_coordinate_transform_and_embodiment_mapping(tmp_path):
    adapter = DatasetAdapter(tmp_path)
    transform = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    payload = [
        {
            "episode_id": "e1",
            "obs": {"ee_position": [0, 0, 0], "joint_positions": {"source_joint": 0.5}},
            "action": {"source_joint": 0.2},
        }
    ]
    _, episodes = adapter.normalize(
        payload,
        coordinate_transform=transform,
        embodiment_mapping={"source_joint": "panda_joint1"},
    )
    tr = episodes[0].transitions[0]
    assert tr.observation["ee_position"] == [1.0, 2.0, 3.0]
    assert tr.observation["joint_positions"]["panda_joint1"] == 0.5
    assert tr.action["panda_joint1"] == 0.2
