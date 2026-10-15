# RoboPilot Training

Two workflows:
- `synthetic`: generate scene parameter samples, collect transitions, inspect/label/augment/export datasets.
- `rl`: Gymnasium integration helpers, reward-design example, PPO/SAC training, rollout collection and evaluation.

This is an environment-agnostic starter. `Pendulum-v1` is used only for smoke tests; it is not a robot manipulation task. Register your RoboPilot MuJoCo environment with Gymnasium and pass its environment ID to the scripts.

Install from the repository root:
```bash
pip install -r training/requirements-training.txt
```

Smoke-test synthetic collection:
```bash
python -m training.synthetic.generate_episodes --env-id Pendulum-v1 --episodes 5 --output training/datasets/demo.npz
python -m training.synthetic.collect_observations --input training/datasets/demo.npz
```

Smoke-test RL:
```bash
python -m training.rl.train --env-id Pendulum-v1 --timesteps 10000 --output training/rl/checkpoints/ppo
python -m training.rl.evaluate --env-id Pendulum-v1 --model training/rl/checkpoints/ppo.zip --episodes 5
```

A RoboPilot environment must implement Gymnasium `reset()` and `step()` and define stable action/observation spaces. Add task-specific success labels, camera frames, language instructions, object poses and domain randomization when connecting the manipulation simulator. Synthetic generation is a data-production pipeline; RL is a policy-learning pipeline. Demonstrations can later seed imitation learning, and RL rollouts can expand datasets.
