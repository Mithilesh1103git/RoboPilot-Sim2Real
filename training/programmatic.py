"""Stable Python entry points for the independent training application.

These functions are callable from Python scripts, notebooks, tests, and API
workers without parsing CLI arguments. CLI scripts are thin wrappers around the
same functions.
"""

from __future__ import annotations


def train_policy(**kwargs):
    from training.rl.train import run_training

    return run_training(**kwargs)


def evaluate_policy(**kwargs):
    from training.rl.evaluate import evaluate_policy as evaluate

    return evaluate(**kwargs)


def train_world_model(**kwargs):
    from training.rl.train_world_model import run_training

    return run_training(**kwargs)


def collect_synthetic_episodes(**kwargs):
    from training.synthetic.collect_episodes import collect

    return collect(**kwargs)


def generate_transition_dataset(**kwargs):
    from training.synthetic.generate_episodes import generate_dataset

    return generate_dataset(**kwargs)


def collect_policy_rollouts(**kwargs):
    from training.rl.collect_rollouts import collect_policy_rollouts as collect

    return collect(**kwargs)


def augment_dataset(**kwargs):
    from training.synthetic.augment_data import augment_dataset as augment

    return augment(**kwargs)


def export_dataset(**kwargs):
    from training.synthetic.export_dataset import export_dataset as export

    return export(**kwargs)


def summarize_dataset(**kwargs):
    from training.synthetic.collect_observations import summarize_dataset as summarize

    return summarize(**kwargs)


def label_episodes(**kwargs):
    from training.synthetic.label_episodes import label_episodes as label

    return label(**kwargs)


def generate_scenes(**kwargs):
    from training.synthetic.generate_scenes import generate_scenes as generate

    return generate(**kwargs)


def generate_randomized_scene_manifest(**kwargs):
    from training.synthetic.domain_randomization import generate_scene_manifest

    return generate_scene_manifest(**kwargs)


def run_flow_pipeline(**kwargs):
    from training.rl.flow_control import run_flow_pipeline as run

    return run(**kwargs)
