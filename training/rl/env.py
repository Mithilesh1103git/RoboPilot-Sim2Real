import gymnasium as gym


def make_env(env_id, seed=None, render_mode=None):
    """Register your RoboPilot task with Gymnasium, then pass its env ID here."""
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env
