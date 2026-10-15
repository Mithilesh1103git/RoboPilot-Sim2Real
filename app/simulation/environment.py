"""MuJoCo lifecycle wrapper (starter implementation)."""
from pathlib import Path


class SimulationEnvironment:
    def __init__(self, model_path: str | Path | None = None, headless: bool = True):
        self.model_path = Path(model_path) if model_path else None
        self.headless = headless
        self.model = None
        self.data = None

    def load(self):
        """Load an MJCF/XML model. Requires a model_path and mujoco installed."""
        if self.model_path is None:
            raise ValueError("Set model_path to a MuJoCo MJCF/XML scene.")
        import mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        return self

    def reset(self):
        if self.model is None or self.data is None:
            raise RuntimeError("Call load() before reset().")
        import mujoco
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1):
        if self.model is None or self.data is None:
            raise RuntimeError("Call load() before step().")
        import mujoco
        for _ in range(max(1, n)):
            mujoco.mj_step(self.model, self.data)

    def observation(self) -> dict:
        if self.model is None or self.data is None:
            return {"connected": False}
        return {
            "connected": True,
            "time": float(self.data.time),
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
        }
