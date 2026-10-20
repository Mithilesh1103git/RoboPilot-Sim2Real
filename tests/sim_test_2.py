import mujoco.viewer
import numpy as np
import time
import urllib.request

# 1. Base URL path targeting Google DeepMind's MuJoCo Menagerie repo
RAW_URL = "https://githubusercontent.com"


def fetch_bytes(url):
    """Helper to download remote files directly into python memory bytes."""
    with urllib.request.urlopen(url) as response:
        return response.read()


print("Fetching assets over the network...")

# 2. Download the core text XML environments
scene_xml_string = fetch_bytes(RAW_URL + "scene.xml").decode("utf-8")
panda_xml_bytes = fetch_bytes(RAW_URL + "panda.xml")

# 3. List of required 3D mesh assets used by the Franka Panda model
mesh_names = [
    "link0.obj",
    "link1.obj",
    "link2.obj",
    "link3.obj",
    "link4.obj",
    "link5.obj",
    "link6.obj",
    "link7.obj",
    "hand.obj",
    "finger.obj",
]

# 4. Bind the downloaded contents directly into the Virtual File System map
assets_dict = {"panda.xml": panda_xml_bytes}
for mesh in mesh_names:
    vfs_key = f"assets/{mesh}"
    assets_dict[vfs_key] = fetch_bytes(RAW_URL + vfs_key)

print("Compiling model from memory VFS...")
# 5. Build and launch the simulation environment
model = mujoco.MjModel.from_xml_string(scene_xml_string, assets=assets_dict)
data = mujoco.MjData(model)

N_STEPS = 5

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer running successfully!")
    while viewer.is_running():
        step_start = time.time()

        # Basic control wave
        t = data.time
        for joint_idx in range(7):
            data.ctrl[joint_idx] = 0.5 * np.sin(2.0 * np.pi * 0.2 * t + joint_idx)

        mujoco.mj_step(model, data, nstep=N_STEPS)
        viewer.sync()

        time_until_next_step = (model.opt.timestep * N_STEPS) - (
            time.time() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
