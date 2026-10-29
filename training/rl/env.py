# Robot-profile-driven Gymnasium manipulation environment; independent of app.
import re
import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from training.robots.registry import load_robot_profile

TRAINING_ROOT = Path(__file__).resolve().parents[1]


class RobotPickPlaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, robot_name=None, render_mode=None, episode_steps=250, seed=None):
        super().__init__()
        self.robot_name = (robot_name or "panda").lower()
        self.profile = load_robot_profile(self.robot_name)
        self.task = self.profile["task"]
        self.episode_steps = int(episode_steps)
        self._tmp = None
        self.model = self.data = None
        self.steps = 0
        n = len(self.profile["arm_joint_names"])
        g = max(1, len(self.profile.get("gripper_actuators", [])))
        self.arm_dof = n
        self.gripper_dof = g
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n + g,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(2 * n + 12,), dtype=np.float32
        )
        self._build_model()
        if seed is not None:
            self.reset(seed=seed)

    def _build_model(self):
        import mujoco

        robot_dir = (
            TRAINING_ROOT / "assets" / "robots" / self.profile["asset_directory"]
        )
        src = robot_dir / self.profile["mjcf_file"]
        if not src.is_file():
            raise FileNotFoundError(
                f"Training-owned MJCF missing for {self.robot_name!r}: {src}"
            )
        self._tmp = tempfile.TemporaryDirectory(
            prefix=f"robopilot_training_{self.robot_name}_"
        )
        tmp = Path(self._tmp.name)
        copied = tmp / "robot"
        shutil.copytree(robot_dir, copied)
        local_model = copied / self.profile["mjcf_file"]
        text = local_model.read_text(encoding="utf-8")
        text = re.sub(r"<option\b[^>]*/>", "", text)
        text = re.sub(r"\s*<keyframe>.*?</keyframe>", "", text, flags=re.DOTALL)
        local_model.write_text(text, encoding="utf-8")
        t = self.task
        ox, oy, oz = t.get("object_start", [0.48, -0.20, 0.425])
        tx, ty, tz = t.get("destination_start", [0.57, 0.22, 0.405])
        table_x, table_y, table_z = t.get("table_position", [0.48, 0, 0.38])
        sx, sy, sz = t.get("object_half_size", [0.022, 0.022, 0.025])
        rx, ry, rz = t.get("tray_half_size", [0.13, 0.13, 0.005])
        obj = t.get("object_name", "red_block")
        dest = t.get("destination_name", "blue_tray")
        scene = f"""<mujoco model="TrainingOwned_{self.robot_name}_Manipulation">
<include file="{local_model.as_posix()}"/>
<option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
<worldbody>
<geom name="floor" type="plane" pos="0 0 0" size="2 2 0.1"/>
<body name="table" pos="{table_x} {table_y} {table_z}"><geom type="box" size="0.33 0.58 0.02"/></body>
<body name="{obj}" pos="{ox} {oy} {oz}"><freejoint name="{obj}_free"/><geom name="{obj}_geom" type="box" size="{sx} {sy} {sz}" mass="0.08" rgba="0.88 0.08 0.06 1" friction="1.2 0.02 0.002" condim="4"/></body>
<body name="{dest}" pos="{tx} {ty} {tz}"><geom type="box" size="{rx} {ry} {rz}"/><geom type="box" pos="{rx} 0 0.025" size="0.005 {ry} 0.025"/><geom type="box" pos="-{rx} 0 0.025" size="0.005 {ry} 0.025"/><geom type="box" pos="0 {ry} 0.025" size="{rx} 0.005 0.025"/><geom type="box" pos="0 -{ry} 0.025" size="{rx} 0.005 0.025"/></body>
</worldbody></mujoco>"""
        scene_path = tmp / "training_scene.xml"
        scene_path.write_text(scene, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

    def _id(self, kind, name):
        import mujoco

        idx = mujoco.mj_name2id(self.model, kind, name)
        if idx < 0:
            raise KeyError(f"Training model element missing: {name}")
        return int(idx)

    def _joint_data(self):
        import mujoco

        qids = []
        dofs = []
        aids = []
        lows = []
        highs = []
        for name in self.profile["arm_joint_names"]:
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = next(
                (
                    a
                    for a in range(self.model.nu)
                    if int(self.model.actuator_trnid[a, 0]) == jid
                ),
                None,
            )
            if aid is None:
                raise RuntimeError(
                    f"No actuator found for configured arm joint {name!r}."
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
        return qids, dofs, aids, np.asarray(lows), np.asarray(highs)

    def _positions(self):
        import mujoco

        names = (
            self.profile["end_effector_body"],
            self.task.get("object_name", "red_block"),
            self.task.get("destination_name", "blue_tray"),
        )
        return tuple(
            self.data.xpos[self._id(mujoco.mjtObj.mjOBJ_BODY, n)].copy() for n in names
        )

    def _obs(self):
        qids, dofs, _, _, _ = self._joint_data()
        hand, obj, dest = self._positions()
        return np.concatenate(
            [self.data.qpos[qids], self.data.qvel[dofs], hand, obj, dest, obj - dest]
        ).astype(np.float32)

    def _set_gripper(self, values):
        import mujoco

        for name, value in zip(self.profile.get("gripper_actuators", []), values):
            self.data.ctrl[self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = float(value)

    def reset(self, *, seed=None, options=None):
        import mujoco

        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        qids, dofs, aids, lows, highs = self._joint_data()
        for qid, val, aid in zip(qids, self.profile["home_positions"], aids):
            self.data.qpos[qid] = float(val)
            self.data.ctrl[aid] = float(val)
        for name, value in self.profile.get("gripper_joint_positions", {}).items():
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[int(self.model.jnt_qposadr[jid])] = float(value)
        self._set_gripper(self.profile.get("gripper_open_control", []))
        obj = self.task.get("object_name", "red_block")
        jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_free")
        adr = int(self.model.jnt_qposadr[jid])
        ox, oy, oz = self.task.get("object_start", [0.48, -0.20, 0.425])
        self.data.qpos[adr : adr + 3] = [
            ox + float(self.np_random.uniform(-0.025, 0.025)),
            oy + float(self.np_random.uniform(-0.025, 0.025)),
            oz,
        ]
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {"success": False, "robot": self.robot_name}

    def step(self, action):
        import mujoco

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError(
                f"Action must be a finite {self.action_space.shape[0]}-vector."
            )
        qids, dofs, aids, lows, highs = self._joint_data()
        current = self.data.qpos[qids].copy()
        target = np.clip(
            current + np.clip(action[: self.arm_dof], -1, 1) * 0.08,
            lows + 1e-4,
            highs - 1e-4,
        )
        for aid, val in zip(aids, target):
            self.data.ctrl[aid] = float(val)
        open_values = np.asarray(
            self.profile.get("gripper_open_control", []), dtype=float
        )
        close_values = np.asarray(
            self.profile.get("gripper_closed_control", []), dtype=float
        )
        if self.gripper_dof:
            grip = np.clip(action[self.arm_dof :], -1, 1)
            commands = (grip + 1.0) / 2.0 * (open_values - close_values) + close_values
            self._set_gripper(commands)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        hand, obj, dest = self._positions()
        reach = float(np.linalg.norm(hand - obj))
        goal = float(np.linalg.norm(obj - dest))
        reward = -1.5 * reach - 0.8 * goal - 0.01
        tolerance = self.task.get("placement_tolerance", [0.105, 0.105, 0.085])
        placed = (
            abs(obj[0] - dest[0]) <= float(tolerance[0])
            and abs(obj[1] - dest[1]) <= float(tolerance[1])
            and abs(obj[2] - dest[2]) <= float(tolerance[2])
        )
        if placed:
            reward += 10.0
        truncated = self.steps >= self.episode_steps
        return (
            self._obs(),
            float(reward),
            bool(placed),
            bool(truncated),
            {
                "success": bool(placed),
                "reach_distance": reach,
                "goal_distance": goal,
                "steps": self.steps,
                "robot": self.robot_name,
            },
        )

    def close(self):
        self.data = self.model = None
        if self._tmp:
            self._tmp.cleanup()
            self._tmp = None


PandaPickPlaceEnv = RobotPickPlaceEnv


def make_env(env_id="RoboPilotPickPlace-v0", robot_name=None, **kwargs):
    if env_id not in {
        "RoboPilotPandaPickPlace-v0",
        "panda_pick_place",
        "RoboPilotPickPlace-v0",
        "pick_and_place",
    }:
        return gym.make(env_id, **kwargs)
    return RobotPickPlaceEnv(robot_name=robot_name, **kwargs)
