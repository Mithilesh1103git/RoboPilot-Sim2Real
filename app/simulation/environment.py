"""MuJoCo Panda scene with explicit asset-root resolution."""

import re
import tempfile
import threading
import time
from pathlib import Path

from app.config import settings


class SimulationEnvironment:
    def __init__(self, asset_root=None, realtime=None):
        self.asset_root = Path(asset_root or settings.assets_dir).expanduser().resolve()
        self.realtime = (
            settings.simulation_realtime if realtime is None else bool(realtime)
        )
        self.model = self.data = self.viewer = self._generated_dir = None
        self._lock = threading.RLock()

    def _make_scene(self):
        robot_dir = self.asset_root / "franka_emika_panda"
        panda_xml = robot_dir / "panda.xml"
        mesh_dir = robot_dir / "assets"
        if not panda_xml.is_file() or not mesh_dir.is_dir():
            raise FileNotFoundError(
                f"Panda MJCF/assets not found. Expected {panda_xml} and {mesh_dir}. Set ROBOPILOT_ASSETS_DIR to the directory containing franka_emika_panda/."
            )
        self._generated_dir = tempfile.TemporaryDirectory(prefix="robopilot_runtime_")
        tmp = Path(self._generated_dir.name)
        text = panda_xml.read_text(encoding="utf-8").replace(
            'meshdir="assets"', f'meshdir="{mesh_dir.as_posix()}"'
        )
        text = text.replace('<option integrator="implicitfast"/>', "")
        text = re.sub(r"\s*<keyframe>.*?</keyframe>", "", text, flags=re.DOTALL)
        (tmp / "panda.xml").write_text(text, encoding="utf-8")
        scene = """<mujoco model="RoboPilot Panda Pick Place">
<include file="panda.xml"/>
<option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
<statistic center="0.42 0 0.42" extent="1.3"/>
<worldbody>
<light name="key_light" pos="0.2 -0.3 2.0" dir="0 0 -1" directional="true"/>
<geom name="floor" type="plane" pos="0 0 0" size="2 2 0.1"/>
<body name="table" pos="0.48 0 0.38"><geom name="table_top" type="box" size="0.33 0.58 0.02"/></body>
<body name="red_block" pos="0.48 -0.20 0.425"><freejoint name="red_block_free"/><geom name="red_block_geom" type="box" size="0.022 0.022 0.025" mass="0.08" rgba="0.88 0.08 0.06 1" friction="1.2 0.02 0.002" condim="4"/></body>
<body name="blue_tray" pos="0.57 0.22 0.405"><geom name="tray_floor" type="box" size="0.13 0.13 0.005"/><geom name="tray_wall_xp" type="box" pos="0.13 0 0.025" size="0.005 0.13 0.025"/><geom name="tray_wall_xn" type="box" pos="-0.13 0 0.025" size="0.005 0.13 0.025"/><geom name="tray_wall_yp" type="box" pos="0 0.13 0.025" size="0.13 0.005 0.025"/><geom name="tray_wall_yn" type="box" pos="0 -0.13 0.025" size="0.13 0.005 0.025"/></body>
</worldbody></mujoco>"""
        path = tmp / "panda_pick_place.xml"
        path.write_text(scene, encoding="utf-8")
        return path

    def load(self):
        if self.model is not None:
            return self
        import mujoco

        path = self._make_scene()
        try:
            self.model = mujoco.MjModel.from_xml_path(str(path))
            self.data = mujoco.MjData(self.model)
            self.reset()
            return self
        except Exception:
            self.close()
            raise

    def _id(self, kind, name):
        import mujoco

        idx = mujoco.mj_name2id(self.model, kind, name)
        if idx < 0:
            raise KeyError(f"MuJoCo element not found: {name}")
        return int(idx)

    def _actuator_for_joint(self, jid):
        if jid < 0:
            return None
        for aid in range(self.model.nu):
            if int(self.model.actuator_trnid[aid, 0]) == jid:
                return aid
        return None

    def reset(self):
        import mujoco

        if self.model is None or self.data is None:
            raise RuntimeError("Load simulation before reset.")
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            home = (0.0, -0.45, 0.0, -2.2, 0.0, 1.8, 0.8)
            for i, val in enumerate(home, 1):
                jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                self.data.qpos[self.model.jnt_qposadr[jid]] = val
                aid = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
                self.data.ctrl[aid] = val
            for i in (1, 2):
                jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")
                self.data.qpos[self.model.jnt_qposadr[jid]] = 0.04
            self.data.ctrl[self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")] = 255.0
            mujoco.mj_forward(self.model, self.data)
            return self.observation()

    def step(self, n=1):
        import mujoco

        if self.model is None or self.data is None:
            raise RuntimeError("Simulation is not loaded.")
        with self._lock:
            for _ in range(max(1, min(int(n), 10000))):
                started = time.perf_counter()
                mujoco.mj_step(self.model, self.data)
                if self.realtime:
                    delay = float(self.model.opt.timestep) - (
                        time.perf_counter() - started
                    )
                    if delay > 0:
                        time.sleep(delay)
        return self.observation()

    def observation(self):
        if self.model is None or self.data is None:
            return {"connected": False}
        import mujoco

        with self._lock:
            out = {
                "connected": True,
                "time": float(self.data.time),
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
            }
            for name in ("hand", "red_block", "blue_tray"):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    out[f"{name}_position"] = self.data.xpos[bid].tolist()
            return out

    def body_position(self, name):
        import mujoco

        if self.model is None or self.data is None:
            raise RuntimeError("Simulation is not loaded.")
        with self._lock:
            return self.data.xpos[self._id(mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    def close(self):
        with self._lock:
            if self.viewer is not None:
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = None
            self.data = self.model = None
            if self._generated_dir:
                self._generated_dir.cleanup()
                self._generated_dir = None
