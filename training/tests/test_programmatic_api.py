import json

import numpy as np

from training.programmatic import (
    augment_dataset,
    export_dataset,
    generate_randomized_scene_manifest,
    generate_scenes,
    label_episodes,
    summarize_dataset,
)


def _dataset(path):
    np.savez_compressed(
        path,
        observations=np.zeros((4, 3), dtype=np.float32),
        actions=np.ones((4, 2), dtype=np.float32),
        rewards=np.array([1, 0, 2, 3], dtype=np.float32),
        next_observations=np.ones((4, 3), dtype=np.float32),
        terminated=np.array([False, True, False, True]),
        truncated=np.array([False, False, False, False]),
        episode_ids=np.array([0, 0, 1, 1], dtype=np.int32),
    )


def test_scene_generation_is_callable_without_cli(tmp_path):
    output = tmp_path / "scenes.jsonl"
    result = generate_scenes(count=3, seed=10, output=str(output))
    assert result["count"] == 3
    assert len(output.read_text().splitlines()) == 3
    manifest = tmp_path / "randomized.jsonl"
    assert (
        generate_randomized_scene_manifest(count=2, output=str(manifest))["count"] == 2
    )
    assert len(manifest.read_text().splitlines()) == 2


def test_dataset_utilities_are_callable_without_cli(tmp_path):
    source = tmp_path / "transitions.npz"
    _dataset(source)
    summary = summarize_dataset(
        input=str(source), output=str(tmp_path / "summary.json")
    )
    assert summary["summary"]["reward_mean"] == 1.5
    exported = export_dataset(input=str(source), output=str(tmp_path / "export.npz"))
    assert exported["transitions"] == 4
    augmented = augment_dataset(
        input=str(source), output=str(tmp_path / "augmented.npz"), noise_std=0, seed=1
    )
    assert augmented["transitions"] == 4
    labels = label_episodes(input=str(source), output=str(tmp_path / "labels.json"))
    assert labels["episodes"] == 2
    assert json.loads((tmp_path / "labels.json").read_text())[0]["success"] is None
