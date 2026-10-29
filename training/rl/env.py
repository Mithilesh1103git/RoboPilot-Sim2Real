"""Training-owned Gymnasium Panda manipulation environment. No imports from app/."""

import re
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

TRAINING_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = TRAINING_ROOT / "assets" / "robots" / "franka_emika_panda"


class PandaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, episode_steps=250, seed=None):
        super().__init__()
        self.episode_steps = int(episode_steps)
        self._tmp = None
        self.model = self.data = None
        self.steps = 0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(26,), dtype=np.float32
        )
        self._build_model()
        if seed is not None:
            self.reset(seed=seed)

    def _build_model(self):
        import mujoco

        src = ROBOT_DIR / "panda.xml"
        if not src.is_file():
            raise FileNotFoundError(f"Training-owned Panda MJCF missing: {src}")
        self._tmp = tempfile.TemporaryDirectory(prefix="robopilot_training_")
        tmp = Path(self._tmp.name)
        text = src.read_text(encoding="utf-8").replace(
            'meshdir="assets"', f'meshdir="{(ROBOT_DIR/"assets").as_posix()}"'
        )
        text = text.replace('<option integrator="implicitfast"/>', "")
        text = re.sub(r"\s*<keyframe>.*?</keyframe>", "", text, flags=re.DOTALL)
        (tmp / "panda.xml").write_text(text, encoding="utf-8")
        scene = "\n".join(
            [
                '<mujoco model="TrainingOwnedPandaManipulation">',
                '<include file="panda.xml"/>',
                '<option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>',
                "<worldbody>",
                '<geom name="floor" type="plane" pos="0 0 0" size="2 2 0.1"/>',
                '<body name="table" pos="0.48 0 0.38"><geom type="box" size="0.33 0.58 0.02"/></body>',
                '<body name="red_block" pos="0.48 -0.20 0.425"><freejoint name="red_block_free"/><geom name="red_block_geom" type="box" size="0.022 0.022 0.025" mass="0.08" rgba="0.88 0.08 0.06 1" friction="1.2 0.02 0.002" condim="4"/></body>',
                '<body name="blue_tray" pos="0.57 0.22 0.405"><geom name="tray_floor" type="box" size="0.13 0.13 0.005"/><geom name="tray_wall_xp" type="box" pos="0.13 0 0.025" size="0.005 0.13 0.025"/><geom name="tray_wall_xn" type="box" pos="-0.13 0 0.025" size="0.005 0.13 0.025"/><geom name="tray_wall_yp" type="box" pos="0 0.13 0.025" size="0.13 0.005 0.025"/><geom name="tray_wall_yn" type="box" pos="0 -0.13 0.025" size="0.13 0.005 0.025"/></body>',
                "</worldbody></mujoco>",
            ]
        )
        (tmp / "training_scene.xml").write_text(scene, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(tmp / "training_scene.xml"))
        self.data = mujoco.MjData(self.model)

    def _id(self, kind, name):
        import mujoco

        idx = mujoco.mj_name2id(self.model, kind, name)
        if idx < 0:
            raise KeyError(f"Training model element missing: {name}")
        return int(idx)

    def _joints(self):
        import mujoco

        qids = []
        dofs = []
        aids = []
        lows = []
        highs = []
        for i in range(1, 8):
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            qids.append(int(self.model.jnt_qposadr[jid]))
            dofs.append(int(self.model.jnt_dofadr[jid]))
            aids.append(self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}"))
            lows.append(float(self.model.jnt_range[jid, 0]))
            highs.append(float(self.model.jnt_range[jid, 1]))
        return qids, dofs, aids, np.array(lows), np.array(highs)

    def _positions(self):
        import mujoco

        return tuple(
            self.data.xpos[self._id(mujoco.mjtObj.mjOBJ_BODY, n)].copy()
            for n in ("hand", "red_block", "blue_tray")
        )

    def _obs(self):
        qids, dofs, _, _, _ = self._joints()
        hand, block, tray = self._positions()
        return np.concatenate(
            [
                self.data.qpos[qids],
                self.data.qvel[dofs],
                hand,
                block,
                tray,
                block - tray,
            ]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        import mujoco

        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        qids, dofs, aids, lows, highs = self._joints()
        home = np.array([0.0, -0.45, 0.0, -2.2, 0.0, 1.8, 0.8])
        for qid, val in zip(qids, home):
            self.data.qpos[qid] = val
        for i in (1, 2):
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")
            self.data.qpos[self.model.jnt_qposadr[jid]] = 0.04
        for aid, val in zip(aids, home):
            self.data.ctrl[aid] = val
        self.data.ctrl[self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")] = 255.0
        # Domain randomization is applied to the actual simulator body pose.
        free_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "red_block_free")
        qadr = int(self.model.jnt_qposadr[free_id])
        self.data.qpos[qadr : qadr + 3] = [
            0.48 + float(self.np_random.uniform(-0.025, 0.025)),
            -0.20 + float(self.np_random.uniform(-0.025, 0.025)),
            0.425,
        ]
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {"success": False}

    def step(self, action):
        import mujoco

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite 8-vector.")
        qids, dofs, aids, lows, highs = self._joints()
        current = self.data.qpos[qids].copy()
        target = np.clip(
            current + np.clip(action[:7], -1, 1) * 0.08, lows + 1e-4, highs - 1e-4
        )
        for aid, val in zip(aids, target):
            self.data.ctrl[aid] = float(val)
        self.data.ctrl[self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")] = float(
            (np.clip(action[7], -1, 1) + 1) * 127.5
        )
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        hand, block, tray = self._positions()
        reach = float(np.linalg.norm(hand - block))
        goal = float(np.linalg.norm(block - tray))
        reward = -1.5 * reach - 0.8 * goal - 0.01
        placed = (
            abs(block[0] - tray[0]) < 0.105
            and abs(block[1] - tray[1]) < 0.105
            and 0.405 <= block[2] <= 0.49
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
            },
        )

    def close(self):
        self.data = self.model = None
        if self._tmp:
            self._tmp.cleanup()
            self._tmp = None


def make_env(env_id="RoboPilotPandaPickPlace-v0", **kwargs):
    if env_id not in {"RoboPilotPandaPickPlace-v0", "panda_pick_place"}:
        return gym.make(env_id, **kwargs)
    return PandaPickPlaceEnv(**kwargs)
