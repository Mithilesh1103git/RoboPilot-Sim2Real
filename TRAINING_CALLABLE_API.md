# Training callable API update

The training package now exposes reusable Python functions alongside optional CLI wrappers.

## Stable Python entry points

Import from `training.programmatic`:
- `train_policy`
- `evaluate_policy`
- `train_world_model`
- `collect_synthetic_episodes`
- `generate_transition_dataset`
- `collect_policy_rollouts`
- `augment_dataset`
- `export_dataset`
- `summarize_dataset`
- `label_episodes`
- `generate_scenes`
- `generate_randomized_scene_manifest`
- `run_flow_pipeline`

The individual modules also expose their operation-specific functions. Their `main(argv=None)` functions are CLI adapters and parse arguments only when invoked.

## Validation performed
- Python compilation passed for the training package.
- CLI `--help` checks passed for all 13 argparse-based training/data scripts.
- Full available test suite: 28 passed.

## Important environment limitation
The current validation environment does not have `stable-baselines3` or `gymnasium` installed. Therefore, actual PPO/SAC training, MuJoCo rollouts, policy evaluation, and world-model learning were not run end-to-end here. Install `training/requirements-training.txt` in the training environment before running those workloads.

The dashboard job manager still runs long jobs in isolated subprocesses. The underlying capabilities are now also directly importable and callable from Python without parsing CLI arguments.
