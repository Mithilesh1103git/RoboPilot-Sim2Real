"""Configurable MuJoCo runtime environment driven by robot profiles."""

import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from app.config import settings
from app.robots.registry import load_robot_profile


class SimulationEnvironment:
    def __init__(self, asset_root=None, realtime=None, robot_name=None):
        self.asset_root = Path(asset_root or settings.assets_dir).expanduser().resolve()
        self.robot_name = (robot_name or settings.robot_name).lower()
        self.profile = load_robot_profile(self.robot_name)
        self.realtime = (
            settings.simulation_realtime if realtime is None else bool(realtime)
        )
        self.model = self.data = self.viewer = self._generated_dir = None
        self._lock = threading.RLock()

    def _make_scene(self):
        robot_dir = self.asset_root / self.profile["asset_directory"]
        model_file = robot_dir / self.profile["mjcf_file"]
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Robot MJCF missing for profile {self.robot_name!r}: {model_file}. Install the model under {robot_dir}."
            )
        self._generated_dir = tempfile.TemporaryDirectory(
            prefix=f"robopilot_{self.robot_name}_"
        )
        tmp = Path(self._generated_dir.name)
        copied = tmp / "robot"
        shutil.copytree(robot_dir, copied)

        # Some MJCFs are loaded through <include>, where meshdir resolution can
        # differ from loading the robot XML as the top-level model. Keep the
        # original assets/ tree and add root-level aliases for files in it so
        # references such as `file="link2.stl"` resolve in either case.
        # This is intentionally limited to direct files in the asset directory.
        mesh_assets = copied / "assets"
        if mesh_assets.is_dir():
            for asset_file in mesh_assets.iterdir():
                if asset_file.is_file():
                    alias = copied / asset_file.name
                    if not alias.exists():
                        shutil.copy2(asset_file, alias)

        local_model = copied / self.profile["mjcf_file"]
        text = local_model.read_text(encoding="utf-8")
        text = re.sub(r"<option\b[^>]*/>", "", text)
        text = re.sub(r"\s*<keyframe>.*?</keyframe>", "", text, flags=re.DOTALL)
        local_model.write_text(text, encoding="utf-8")
        task = self.profile["task"]
        ox, oy, oz = task.get("object_start", [0.48, -0.20, 0.425])
        tx, ty, tz = task.get("destination_start", [0.57, 0.22, 0.405])
        table_x, table_y, table_z = task.get("table_position", [0.48, 0, 0.38])
        sx, sy, sz = task.get("object_half_size", [0.022, 0.022, 0.025])
        tray_x, tray_y, tray_z = task.get("tray_half_size", [0.13, 0.13, 0.005])
        obj = task.get("object_name", "red_block")
        dest = task.get("destination_name", "blue_tray")
        scene = f"""<mujoco model="RoboPilot {self.robot_name} pick place">
<include file="{local_model.as_posix()}"/>
<option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
<statistic center="{table_x} {table_y} {table_z}" extent="1.3"/>
<worldbody>
<light name="key_light" pos="0.2 -0.3 2.0" dir="0 0 -1" directional="true"/>
<geom name="floor" type="plane" pos="0 0 0" size="2 2 0.1"/>
<body name="table" pos="{table_x} {table_y} {table_z}"><geom name="table_top" type="box" size="0.33 0.58 0.02"/></body>
<body name="{obj}" pos="{ox} {oy} {oz}"><freejoint name="{obj}_free"/><geom name="{obj}_geom" type="box" size="{sx} {sy} {sz}" mass="0.08" rgba="0.88 0.08 0.06 1" friction="1.2 0.02 0.002" condim="4"/></body>
<body name="{dest}" pos="{tx} {ty} {tz}"><geom name="{dest}_floor" type="box" size="{tray_x} {tray_y} {tray_z}"/><geom name="{dest}_wall_xp" type="box" pos="{tray_x} 0 0.025" size="0.005 {tray_y} 0.025"/><geom name="{dest}_wall_xn" type="box" pos="-{tray_x} 0 0.025" size="0.005 {tray_y} 0.025"/><geom name="{dest}_wall_yp" type="box" pos="0 {tray_y} 0.025" size="{tray_x} 0.005 0.025"/><geom name="{dest}_wall_yn" type="box" pos="0 -{tray_y} 0.025" size="{tray_x} 0.005 0.025"/></body>
</worldbody></mujoco>"""
        path = tmp / "task_scene.xml"
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

    def _joint_info(self):
        import mujoco

        qids = []
        dofs = []
        aids = []
        lows = []
        highs = []
        for name in self.profile["arm_joint_names"]:
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = self._actuator_for_joint(jid)
            if aid is None:
                raise RuntimeError(
                    f"Robot profile {self.robot_name}: no actuator controls arm joint {name!r}."
                )
            qids.append(int(self.model.jnt_qposadr[jid]))
            dofs.append(int(self.model.jnt_dofadr[jid]))
            aids.append(aid)
            if self.model.jnt_limited[jid]:
                lows.append(float(self.model.jnt_range[jid, 0]))
                highs.append(float(self.model.jnt_range[jid, 1]))
            else:
                lows.append(-3.14159)
                highs.append(3.14159)
        return qids, dofs, aids, lows, highs

    def set_gripper(self, opened: bool):
        import mujoco

        values = (
            self.profile["gripper_open_control"]
            if opened
            else self.profile["gripper_closed_control"]
        )
        for name, value in zip(self.profile["gripper_actuators"], values):
            self.data.ctrl[self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = float(value)

    def reset(self):
        import mujoco

        if self.model is None or self.data is None:
            raise RuntimeError("Load simulation before reset.")
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            qids, dofs, aids, lows, highs = self._joint_info()
            for qid, val, aid in zip(qids, self.profile["home_positions"], aids):
                self.data.qpos[qid] = float(val)
                self.data.ctrl[aid] = float(val)
            for joint_name, value in self.profile.get(
                "gripper_joint_positions", {}
            ).items():
                jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.data.qpos[int(self.model.jnt_qposadr[jid])] = float(value)
            self.set_gripper(True)
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
            return {"connected": False, "robot": self.robot_name}
        import mujoco

        with self._lock:
            out = {
                "connected": True,
                "robot": self.robot_name,
                "time": float(self.data.time),
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
            }
            for name in (
                self.profile["end_effector_body"],
                self.profile["task"].get("object_name", "red_block"),
                self.profile["task"].get("destination_name", "blue_tray"),
            ):
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
