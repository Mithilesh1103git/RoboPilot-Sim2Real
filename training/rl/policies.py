from stable_baselines3 import PPO, SAC


def build_model(algorithm, env, seed=7, device="auto"):
    algorithm = algorithm.upper()
    if algorithm == "PPO":
        return PPO("MlpPolicy", env, seed=seed, device=device, verbose=1)
    if algorithm == "SAC":
        return SAC("MlpPolicy", env, seed=seed, device=device, verbose=1)
    raise ValueError("Supported algorithms: PPO, SAC")
