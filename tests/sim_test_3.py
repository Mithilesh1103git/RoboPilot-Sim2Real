import mujoco.viewer
import numpy as np
import os
import time

# Point to your local panda.xml file
LOCAL_ASSETS_PATH = r"C:\Users\mithi_w\mydata\git_repos\personal_repos\RoboPilot\assets"
MODEL_PATH = os.path.abspath(
    os.path.join(LOCAL_ASSETS_PATH, "franka_emika_panda", "panda_table_glass.xml")
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print(f"Loading Franka Panda model from: {MODEL_PATH}")

# FIX: Remove the 'assets=assets' argument entirely.
# MuJoCo will resolve the paths relative to the XML file location automatically.
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# 2. Define a simple sinusoidal joint controller
def control_callback(model, data):
    t = data.time

    # Send a gentle sine wave to the first three arm joints
    data.ctrl[0] = 0.5 * np.sin(t * 2.0)
    data.ctrl[1] = 0.3 * np.cos(t * 1.5)
    data.ctrl[2] = 0.5 * np.sin(t * 1.0)

    # Leave the remaining arm joints stable (holding zero)
    data.ctrl[3:7] = 0.0

    # Actuator 7 is the finger/gripper control.
    # Max control range for this actuator configuration is typically 255.0
    data.ctrl[7] = 255.0


# Set the controller callback
mujoco.set_mjcb_control(control_callback)

# 3. Launch the interactive visualizer loop
print("Launching MuJoCo Viewer. Press ESC to close.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    while viewer.is_running():
        step_start = time.time()

        # Step the physics simulation forward
        mujoco.mj_step(model, data)

        # Sync the viewer with the current simulation state
        viewer.sync()

        # Maintain real-time execution speeds
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
