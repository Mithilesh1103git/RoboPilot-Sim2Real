"""MuJoCo environment with deterministic scene creation and reusable lifecycle.

The default Panda scene is generated in a temporary directory and references the
robot meshes in the repository's assets folder. It contains a table, a free red
block, and a fixed blue tray. No project files are modified at runtime.
"""

from __future__ import annotations

import numpy as np
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class SimulationEnvironment:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        asset_root: str | Path | None = None,
        robot_name: str = "panda",
        headless: bool = True,
        enable_local_viewer: bool = False,
        realtime: bool = False,
    ):
        self.model_path = (
            Path(model_path).expanduser().resolve() if model_path else None
        )
        self.asset_root = (
            Path(asset_root).expanduser().resolve() if asset_root else None
        )
        self.robot_name = robot_name
        self.headless = bool(headless)
        self.enable_local_viewer = bool(enable_local_viewer)
        self.realtime = bool(realtime)
        if self.headless and self.enable_local_viewer:
            raise ValueError("Cannot enable the local viewer in headless mode.")
        self.model = None
        self.data = None
        self.viewer = None
        self._viewer_model = None
        self._generated_dir: tempfile.TemporaryDirectory | None = None
        self._lock = threading.RLock()
        self._cached_hand_rotation: np.ndarray | None = None

    def _make_default_scene(self) -> Path:
        if self.robot_name not in {"panda", "franka", "franka_emika_panda"}:
            raise ValueError(
                "The deterministic pick-and-place controller currently supports the Panda only. "
                "Set ROBOT_NAME=panda; UR5e support can be added after a separate controller is validated."
            )
        if self.asset_root is None:
            raise ValueError("asset_root is required when model_path is not provided.")
        robot_dir = self.asset_root / "franka_emika_panda"
        panda_xml = robot_dir / "panda.xml"
        mesh_dir = robot_dir / "assets"
        if not panda_xml.is_file() or not mesh_dir.is_dir():
            raise FileNotFoundError(
                f"Panda model or meshes not found under {robot_dir}"
            )

        self._generated_dir = tempfile.TemporaryDirectory(prefix="robopilot_mujoco_")
        temp_dir = Path(self._generated_dir.name)
        # Panda's original compiler contains meshdir="assets". Replace it with
        # an absolute path so this generated XML can live outside the repo.
        panda_text = panda_xml.read_text(encoding="utf-8")
        panda_text = panda_text.replace(
            'meshdir="assets"', f'meshdir="{mesh_dir.as_posix()}"'
        )
        panda_text = panda_text.replace('<option integrator="implicitfast"/>', "")
        # The source Panda keyframe only contains the robot's 9 qpos values;
        # this scene adds a free object, so remove the partial keyframe and set
        # all relevant joint states explicitly in _set_home_state().
        panda_text = re.sub(
            r"\s*<keyframe>.*?</keyframe>", "", panda_text, flags=re.DOTALL
        )
        (temp_dir / "panda.xml").write_text(panda_text, encoding="utf-8")
        scene = temp_dir / "panda_pick_place.xml"
        scene.write_text(
            f"""<mujoco model="RoboPilot Panda Pick Place">
  <include file="panda.xml"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <statistic center="0.42 0 0.42" extent="1.3"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.25 0.25 0.25" specular="0.15 0.15 0.15"/>
    <global azimuth="135" elevation="-25" offwidth="960" offheight="720"/>
  </visual>
  <worldbody>
    <light name="key_light" pos="0.2 -0.3 2.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" pos="0 0 0" size="2 2 0.1" rgba="0.82 0.84 0.87 1"/>
    <body name="table" pos="0.48 0 0.38">
      <geom name="table_top" type="box" pos="0 0 0" size="0.33 0.58 0.02" rgba="0.35 0.22 0.13 1"/>
    </body>
    <body name="red_block" pos="0.48 -0.20 0.425">
      <freejoint name="red_block_free"/>
      <geom name="red_block_geom" type="box" size="0.022 0.022 0.025" mass="0.08" rgba="0.88 0.08 0.06 1" friction="1.2 0.02 0.002" condim="4" solref="0.01 1"/>
    </body>
    <body name="blue_tray" pos="0.57 0.22 0.405">
      <geom name="tray_floor" type="box" size="0.13 0.13 0.005" rgba="0.08 0.28 0.88 1"/>
      <geom name="tray_wall_xp" type="box" pos="0.13 0 0.025" size="0.005 0.13 0.025" rgba="0.05 0.2 0.7 1"/>
      <geom name="tray_wall_xn" type="box" pos="-0.13 0 0.025" size="0.005 0.13 0.025" rgba="0.05 0.2 0.7 1"/>
      <geom name="tray_wall_yp" type="box" pos="0 0.13 0.025" size="0.13 0.005 0.025" rgba="0.05 0.2 0.7 1"/>
      <geom name="tray_wall_yn" type="box" pos="0 -0.13 0.025" size="0.13 0.005 0.025" rgba="0.05 0.2 0.7 1"/>
    </body>
  </worldbody>
</mujoco>""",
            encoding="utf-8",
        )
        return scene

    def load(self):
        with self._lock:
            if self.model is not None and self.data is not None:
                return self
            path = self.model_path or self._make_default_scene()
            if not path.is_file():
                raise FileNotFoundError(f"MuJoCo model file not found: {path}")
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            self.model, self.data = model, data
            try:
                self._set_home_state()
                mujoco.mj_forward(model, data)
                hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
                self._cached_hand_rotation = (
                    data.xmat[hand_id].reshape(3, 3).copy() if hand_id >= 0 else None
                )
            except Exception:
                self.model = self.data = None
                raise
            return self

    def _name_id(self, obj_type, name: str) -> int:
        import mujoco

        if self.model is None:
            raise RuntimeError("Call load() before looking up model objects.")
        result = mujoco.mj_name2id(self.model, obj_type, name)
        if result < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return int(result)

    def _set_home_state(self) -> None:
        """Set a known Panda home pose and initial free-object pose."""
        import mujoco

        assert self.model is not None and self.data is not None
        home = [0.0, -0.35, 0.0, -2.15, 0.0, 1.85, 0.75]
        for i, value in enumerate(home, 1):
            jid = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            adr = int(self.model.jnt_qposadr[jid])
            self.data.qpos[adr] = value
            aid = self._actuator_for_joint(jid)
            if aid is not None:
                self.data.ctrl[aid] = value
        for finger in ("finger_joint1", "finger_joint2"):
            jid = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, finger)
            self.data.qpos[int(self.model.jnt_qposadr[jid])] = 0.04
        gripper_id = self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.data.ctrl[gripper_id] = 255.0
        # Reset the free block to the deterministic start pose on every reset.
        try:
            jid = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, "red_block_free")
            adr = int(self.model.jnt_qposadr[jid])
            self.data.qpos[adr : adr + 3] = [0.48, -0.20, 0.425]
            self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        except KeyError:
            pass
        self.data.qvel[:] = 0
        self.data.ctrl[:] = np.clip(
            self.data.ctrl,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

    def _actuator_for_joint(self, joint_id: int) -> int | None:
        assert self.model is not None
        for actuator_id in range(self.model.nu):
            if int(self.model.actuator_trnid[actuator_id, 0]) == joint_id:
                return actuator_id
        return None

    def reset(self):
        with self._lock:
            if self.model is None or self.data is None:
                self.load()
            import mujoco

            mujoco.mj_resetData(self.model, self.data)
            self._set_home_state()
            mujoco.mj_forward(self.model, self.data)
            self._sync_viewer()
            return self.observation()

    def _ensure_viewer(self):
        if not self.enable_local_viewer:
            return
        if self.headless:
            raise RuntimeError("Cannot open the local viewer in headless mode.")
        if self.model is None or self.data is None:
            raise RuntimeError("Call load() before opening the viewer.")
        if self.viewer is not None:
            try:
                if self.viewer.is_running() and self._viewer_model is self.model:
                    return
            except Exception:
                pass
            self._close_viewer()
        import mujoco
        import mujoco.viewer as mujoco_viewer

        self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
        self._viewer_model = self.model
        try:
            self.viewer.cam.lookat[:] = [0.42, 0.0, 0.42]
            self.viewer.cam.distance = 2.0
            self.viewer.cam.elevation = -25
            self.viewer.cam.azimuth = 135
        except Exception:
            self._close_viewer()
            raise

    def _sync_viewer(self):
        if self.enable_local_viewer:
            self._ensure_viewer()
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def step(self, n: int = 1):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer.")
        with self._lock:
            if self.model is None or self.data is None:
                raise RuntimeError("Call load() before step().")
            import mujoco

            if self.enable_local_viewer:
                self._ensure_viewer()
            for _ in range(n):
                start = time.perf_counter()
                mujoco.mj_step(self.model, self.data)
                if self.enable_local_viewer and self.viewer is not None:
                    if not self.viewer.is_running():
                        self._close_viewer()
                        raise RuntimeError(
                            "The MuJoCo viewer was closed during simulation."
                        )
                    self.viewer.sync()
                if (
                    self.realtime
                    and self.model.opt.timestep > time.perf_counter() - start
                ):
                    time.sleep(self.model.opt.timestep - (time.perf_counter() - start))
            return self.observation()

    def observation(self) -> dict[str, Any]:
        with self._lock:
            if self.model is None or self.data is None:
                return {"connected": False}
            import mujoco

            result: dict[str, Any] = {
                "connected": True,
                "time": float(self.data.time),
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
            }
            for body_name in ("hand", "red_block", "blue_tray"):
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                if body_id >= 0:
                    result[f"{body_name}_position"] = self.data.xpos[body_id].tolist()
            return result

    def body_position(self, body_name: str) -> np.ndarray:
        with self._lock:
            if self.model is None or self.data is None:
                raise RuntimeError("Simulation is not loaded.")
            import mujoco

            body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
            return self.data.xpos[body_id].copy()

    def close(self):
        with self._lock:
            self._close_viewer()
            self.data = None
            self.model = None
            self._cached_hand_rotation = None
            if self._generated_dir is not None:
                self._generated_dir.cleanup()
                self._generated_dir = None

    def _close_viewer(self):
        viewer, self.viewer = self.viewer, None
        self._viewer_model = None
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
